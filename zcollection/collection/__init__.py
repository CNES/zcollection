# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Collection of Zarr groups
=========================
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Literal, NoReturn, Sequence
import datetime
import functools
import io
import json
import logging
import os
import types
import warnings

import dask.bag.core
import dask.distributed
import dask.utils
import fsspec
import xarray

from .. import (
    dask_utils,
    dataset,
    fs_utils,
    merging,
    meta,
    partitioning,
    storage,
    sync,
)
from .abc import PartitionFilter, ReadOnlyCollection
from .callable_objects import UpdateCallable, WrappedPartitionCallable
from .detail import (
    _insert,
    _try_infer_callable,
    _wrap_update_func,
    _wrap_update_func_with_overlap,
)

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)


def _write_immutable_dataset(
    zds: dataset.Dataset,
    axis: str,
    path: str,
    fs: fsspec.AbstractFileSystem,
    synchronizer: sync.Sync,
) -> None:
    """Write the immutable dataset.

    Args:
        zds: The dataset to write.
        axis: The partitioning axis.
        path: The path to the immutable dataset.
        fs: The file system that the partition is stored on.
    """
    immutable_dataset = zds.select_variables_by_dims((axis, ), predicate=False)
    assert len(immutable_dataset.variables) != 0, (
        'The dataset to insert does not contain any variable '
        'that is not split.')
    _LOGGER.info('Creating the immutable dataset: %s', path)
    storage.write_zarr_group(immutable_dataset,
                             path,
                             fs,
                             synchronizer,
                             distributed=False)


def _infer_callable(
    collection: Collection,
    func: UpdateCallable,
    filters: PartitionFilter | None,
    delayed,
    selected_variables: Iterable[str] | None,
    *args,
    **kwargs,
) -> tuple[str, ...]:
    try:
        one_partition: str = next(collection.partitions(filters=filters))
    except StopIteration:
        return tuple()

    with collection.synchronizer:
        zds: dataset.Dataset = storage.open_zarr_group(
            one_partition,
            collection.fs,
            delayed=delayed,
            selected_variables=selected_variables)
    func_result: dict[str, Any]
    func_result = _try_infer_callable(func, zds,
                                      collection.partition_properties.dim,
                                      *args, **kwargs)
    unknown_variables: set[str] = set(func_result) - set(
        collection.metadata.variables.keys())
    if len(unknown_variables):
        raise ValueError(f'Unknown variables: {unknown_variables}')
    return tuple(func_result)


class Collection(ReadOnlyCollection):
    """This class manages a collection of files in Zarr format stored in a set
    of subdirectories. These subdirectories split the data, by cycles or dates
    for example, in order to optimize access and updates, deletion or addition
    of new data.

    Args:
        axis: The axis of the collection. This is the dimension along which the
            data is partitioned.
        ds: The dataset containing the collection. This dataset is used to
            create the metadata of the collection, which is used to validate
            datasets that are inserted in the collection.
        partition_handler: The partitioning strategy for the collection. This
            is an instance of a subclass of
            :py:class:`zarr_partitioning.PartitionHandler`.
        partition_base_dir: The base directory for the collection. This is the
            directory where the subdirectories containing the partitioned data
            are stored.
        mode: The mode of the collection. This can be either 'r' (read-only) or
            'w' (write). In read-only mode, the collection can only be read and
            no data can be inserted or modified. In write mode, the collection
            can be read and modified.
        filesystem: The filesystem to use for the collection. This is an
            instance of a subclass of :py:class:`fsspec.AbstractFileSystem`.
        synchronizer: The synchronizer to use for the collection. This is an
            instance of a subclass of
            :py:class:`zarr_synchronizer.Synchronizer`.

    Raises:
        ValueError:
            If the axis does not exist in the dataset, if the partition key is
            not defined in the dataset or if the access mode is not supported.

    Notes:
        Normally, this class is not instantiated directly but through the
        :py:meth:`create_collection <zcollection.create_collection>` and
        :py:meth:`open_collection <zcollection.open_collection>` methods of this
        library.
    """

    def __init__(
        self,
        axis: str,
        ds: meta.Dataset,
        partition_handler: partitioning.Partitioning,
        partition_base_dir: str,
        *,
        mode: Literal['r', 'w'] | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> None:
        super().__init__(axis=axis,
                         ds=ds,
                         partition_handler=partition_handler,
                         partition_base_dir=partition_base_dir,
                         mode=mode,
                         filesystem=filesystem,
                         synchronizer=synchronizer)

        if self.mode == 'r':
            # pylint: disable=method-hidden
            # These methods are overloaded when the collection is opened in
            # readonly.
            self._read_only_mode()
            # pylint: enable=method-hidden
        else:
            self._write_config(skip_if_exists=True)

    def __str__(self) -> str:
        return (f'<{self.__class__.__name__} '
                f'filesystem={self.fs.__class__.__name__!r}, '
                f'partition_base_dir={self.partition_properties.dir!r}'
                f'mode={self.mode!r}>')

    @staticmethod
    def _unsupported_operation(*args, **kwargs) -> NoReturn:
        """Raise an exception if the operation is not supported."""
        raise io.UnsupportedOperation('not writable')

    def _read_only_mode(self) -> None:
        """Set the unsupported methods to raise an exception when the
        collection is opened in read-only mode."""
        # Set each unsupported method to raise an exception.
        for item in [
                'add_variable',
                'drop_partitions',
                'drop_variable',
                'insert',
                'update',
        ]:
            assert hasattr(self, item), f'{item} is not a known method.'
            setattr(self, item,
                    types.MethodType(Collection._unsupported_operation, self))

    def _write_config(self, skip_if_exists: bool = False) -> None:
        """Write the configuration file."""
        base_dir: str = self.partition_properties.dir
        config: str = self._config(base_dir)
        exists: bool = self.fs.exists(config)

        if exists and skip_if_exists:
            return

        _LOGGER.info(
            "Updating collection's configuration: %s"
            if exists else 'Creating the collection: %s', config)

        self.fs.makedirs(base_dir, exist_ok=True)

        params = {
            'axis': self.axis,
            'dataset': self.metadata.get_config(),
            'partitioning': self.partitioning.get_config(),
        }

        with self.fs.open(config, mode='w') as stream:
            json.dump(params, stream, indent=4)  # type: ignore[arg-type]

    def is_readonly(self) -> bool:
        """Return True if the collection is read-only."""
        return self.mode == 'r'

    @classmethod
    def from_config(
        cls,
        path: str,
        *,
        mode: Literal['r', 'w'] | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> Collection:
        """Open a Collection described by a configuration file.

        Args:
            path: The path to the configuration file.
            mode: The mode of the collection. This can be either 'r'
                (read-only) or 'w' (write).
            filesystem: The filesystem to use for the collection. This is an
                instance of a subclass of
                :py:class:`fsspec.AbstractFileSystem`.
            synchronizer: The synchronizer to use for the collection. This is
                an instance of a subclass of
                :py:class:`zarr_synchronizer.Synchronizer`.

        Returns:
            The collection.

        Raises:
            ValueError:
                If the provided directory does not contain a valid collection
                configuration file.
        """
        _LOGGER.info('Opening collection: %r', path)
        fs: fsspec.AbstractFileSystem = fs_utils.get_fs(filesystem)
        config: str = cls._config(path)
        if not fs.exists(config):
            raise ValueError(f'zarr collection not found at path {path!r}')
        with fs.open(config) as stream:
            data: dict[str, Any] = json.load(stream)
        return Collection(
            data['axis'],
            meta.Dataset.from_config(data['dataset']),
            partitioning.get_codecs(data['partitioning']),
            path,
            mode=mode or 'r',
            filesystem=fs,
            synchronizer=synchronizer,
        )

    # pylint: disable=method-hidden
    def insert(
        self,
        ds: xarray.Dataset | dataset.Dataset,
        *,
        merge_callable: merging.MergeCallable | None = None,
        npartitions: int | None = None,
        validate: bool = False,
    ) -> None:
        """Insert a dataset into the collection.

        Args:
            ds: The dataset to insert. It can be either an xarray.Dataset or a
                dataset.Dataset object.
            merge_callable: A function to use to merge the existing data set
                already stored in partitions with the new partitioned data. If
                None, the new partitioned data overwrites the existing
                partitioned data.
            npartitions: The maximum number of partitions to process in
                parallel. By default, partitions are processed one by one.

                .. note::

                    When inserting partitions, Dask parallelizes the writing of
                    each partition across its workers. Additionally, the writing
                    of variables within a partition is parallelized on the
                    worker responsible for inserting that partition, using
                    multiple threads. If you're using a single Dask worker,
                    partition insertion will happen sequentially and changing
                    this parameter will have no effect.

            validate: Whether to validate dataset metadata before insertion
                or not.

        Raises:
            ValueError:
                If the dataset does not match the definition of the collection.

        Warns:
            UserWarning:
                If two different partitions use the same file (chunk), the
                library that handles the storage of chunked arrays (HDF5,
                NetCDF, Zarr, etc.) must be compatible with concurrent access.

        Notes:
            Each worker will process a set of independent partitions. However,
            be careful, two different partitions can use the same file (chunk),
            therefore, the library that handles the storage of chunked arrays
            (HDF5, NetCDF, Zarr, etc.) must be compatible with concurrent
            access.
        """
        # pylint: disable=method-hidden
        if isinstance(ds, xarray.Dataset):
            ds = dataset.Dataset.from_xarray(ds)

        _LOGGER.info('Inserting of a %s dataset in the collection',
                     dask.utils.format_bytes(ds.nbytes))

        missing_variables: tuple[str, ...] = self.metadata.missing_variables(
            ds.metadata())
        for item in missing_variables:
            variable: meta.Variable = self.metadata.variables[item]
            ds.add_variable(variable)

        if validate and ds.metadata() != self.metadata:
            raise ValueError(
                "Provided dataset's metadata do not match the collection's ones"
            )
        ds = ds.set_for_insertion(self.metadata)
        client: dask.distributed.Client = dask_utils.get_client()

        # If the dataset contains variables that should not be partitioned.
        if self._immutable is not None:

            # On the first call, we store the immutable variables in
            # a directory located at the root of the collection.
            if not self.fs.exists(self._immutable):
                _write_immutable_dataset(ds, self.axis, self._immutable,
                                         self.fs, self.synchronizer)

            # Remove the variables that should not be partitioned.
            ds = ds.select_variables_by_dims((self.axis, ))

        # Process the partitions to insert or update by batches to avoid
        # memory issues.
        partitions = tuple(
            self.partitioning.split_dataset(ds, self.partition_properties.dim))

        if npartitions is not None:
            if npartitions < 1:
                raise ValueError('The number of partitions must be positive')
            npartitions = len(partitions) // npartitions + 1

        scattered_ds: Any = client.scatter(ds)
        for sequence in dask_utils.split_sequence(partitions, npartitions):
            futures: list[dask.distributed.Future] = [
                dask_utils.simple_delayed('insert', _insert)(
                    args=partition,  # type: ignore[arg-type]
                    axis=self.axis,
                    zds=scattered_ds,
                    fs=self.fs,
                    merge_callable=merge_callable,
                    partitioning_properties=self.partition_properties,
                ) for partition in sequence
            ]
            storage.execute_transaction(client, self.synchronizer, futures)

    # pylint: disable=method-hidden
    def drop_partitions(
        self,
        *,
        filters: PartitionFilter = None,
        timedelta: datetime.timedelta | None = None,
    ) -> None:
        # pylint: disable=method-hidden
        """Drop the selected partitions.

        Args:
            filters: The predicate used to filter the partitions to drop. To
                get more information on the predicate, see the documentation of
                the :meth:`partitions` method.
            timedelta: Select the partitions created before the specified time
                delta relative to the current time.

        Example:
            >>> collection.drop_partitions(filters="year == 2019")
            >>> collection.drop_partitions(
            ...     timedelta=datetime.timedelta(days=30))
        """
        now: datetime.datetime = datetime.datetime.now()
        client: dask.distributed.Client = dask_utils.get_client()
        folders = list(self.partitions(filters=filters, lock=True))

        # No partition selected, nothing to do.
        if not folders:
            return

        def is_created_before(path: str, now: datetime.datetime,
                              timedelta: datetime.timedelta) -> bool:
            """Return whether the partition was created before the
            timedelta."""
            created: datetime.datetime = self.fs.created(path)
            return now - created > timedelta

        if timedelta is not None:
            folders = list(
                filter(
                    lambda folder: is_created_before(  # type: ignore[arg-type]
                        folder, now, timedelta),
                    folders))

        storage.execute_transaction(
            client, self.synchronizer,
            client.map(self.fs.rm, folders, recursive=True))

        def invalidate_cache(path) -> None:
            """Invalidate the cache."""
            _LOGGER.info('Dropped partition: %s', path)
            self.fs.invalidate_cache(path)

        tuple(map(invalidate_cache, folders))

    # pylint: disable=method-hidden
    def update(
        self,
        func: UpdateCallable,
        /,
        *args,
        delayed: bool = True,
        depth: int = 0,
        filters: PartitionFilter | None = None,
        npartitions: int | None = None,
        selected_variables: list[str] | None = None,
        trim: bool = True,
        **kwargs,
    ) -> None:
        # pylint: disable=method-hidden
        """Update the selected partitions.

        Args:
            func: The function to apply on each partition.
            *args: The positional arguments to pass to the function.
            delayed: Whether to load data in a dask array or not.
            depth: The depth of the overlap between the partitions. Default is
                0 (no overlap). If depth is greater than 0, the function is
                applied on the partition and its neighbors selected by the
                depth. If ``func`` accepts a partition_info as a keyword
                argument, it will be passed a tuple with the name of the
                partitioned dimension and the slice allowing getting in the
                dataset the selected partition.
            filters: The expression used to filter the partitions to update.
            npartitions: The number of partitions to update in parallel. By
                default, it is equal to the number of Dask workers available
                when calling this method.
            selected_variables: A list of variables to load from the collection.
                If None, all variables are loaded.
            trim: Whether to trim ``depth`` items from each partition after
                calling ``func``. Set it to ``False`` if your function does
                this for you.
            **kwargs: The keyword arguments to pass to the function.

        Raises:
            ValueError: If the variables to update are not in the collection.

        Example:
            >>> import dask.array
            >>> import zcollection
            >>> def ones(ds):
            ...     return dict(var2=ds.variables["var1"].values * 0 + 1)
            >>> collection = zcollection.Collection("my_collection", mode="w")
            >>> collection.update(ones)
        """
        if not callable(func):
            raise TypeError('func must be a callable')

        variables: tuple[str, ...]
        variables = _infer_callable(self, func, filters, delayed,
                                    selected_variables, *args, **kwargs)
        if not variables:
            warnings.warn('You are trying to update an empty collection.',
                          category=RuntimeWarning,
                          stacklevel=2)
            return

        # If depth is not 0, the variables updated must be in the selected
        # variables.
        if depth != 0 and selected_variables is not None:
            selected_variables += list(
                set(variables) - set(selected_variables))

        _LOGGER.info('Updating of the (%s) variable in the collection',
                     ', '.join(repr(item) for item in variables))

        selected_partitions = tuple(self.partitions(filters=filters,
                                                    lock=True))

        local_func: WrappedPartitionCallable = _wrap_update_func(
            *args,
            delayed=delayed,
            func=func,
            fs=self.fs,
            immutable=self._immutable,
            selected_variables=selected_variables,
            **kwargs) if depth == 0 else _wrap_update_func_with_overlap(
                *args,
                delayed=delayed,
                depth=depth,
                dim=self.partition_properties.dim,
                func=func,
                fs=self.fs,
                immutable=self._immutable,
                selected_partitions=selected_partitions,
                selected_variables=selected_variables,
                trim=trim,
                **kwargs)

        client: dask.distributed.Client = dask_utils.get_client()

        batches: Iterator[Sequence[str]] = dask_utils.split_sequence(
            selected_partitions, npartitions
            or dask_utils.dask_workers(client, cores_only=True))
        storage.execute_transaction(
            client, self.synchronizer,
            client.map(local_func, tuple(batches), key=func.__name__))

    def drop_variable(
        self,
        variable: str,
    ) -> None:
        """Delete the variable from the collection.

        Args:
            variable: The variable to delete.

        Raises:
            ValueError: If the variable doesn't exist in the collection or is
                used by the partitioning.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="w")
            >>> collection.drop_variable("my_variable")
        """
        _LOGGER.info('Dropping of the %r variable in the collection', variable)
        if variable in self.partitioning.variables:
            raise ValueError(
                f'The variable {variable!r} is part of the partitioning.')
        if variable not in self.metadata.variables:
            raise ValueError(
                f'The variable {variable!r} does not exist in the collection.')
        if self._immutable:
            zds: dataset.Dataset = storage.open_zarr_group(
                self._immutable, self.fs)
            if variable in zds.variables:
                raise ValueError(
                    f'The variable {variable!r} is part of the immutable '
                    'dataset.')
        client: dask.distributed.Client = dask_utils.get_client()
        bag: dask.bag.core.Bag = self._bag_from_partitions(lock=True)
        awaitables: list[
            dask.distributed.Future] = dask.distributed.futures_of(
                bag.map(storage.del_zarr_array, variable, self.fs).persist())
        storage.execute_transaction(client, self.synchronizer, awaitables)
        del self.metadata.variables[variable]
        self._write_config()

    def add_variable(
        self,
        variable: meta.Variable | dataset.Variable,
    ) -> None:
        """Add a variable to the collection.

        Args:
            variable: The variable to add.

        Raises:
            ValueError: if the variable is already part of the collection, it
                doesn't use the partitioning dimension or use a dimension that
                is not part of the dataset.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="w")
            >>> new_variable = meta.Variable(
            ...     name="my_variable",
            ...     dtype=numpy.dtype("int16"),
            ...     dimensions=("num_lines", "num_pixels"),
            ...     fill_value=32267,
            ...     attrs=(dataset.Attribute(name="my_attr", value=0), ),
            ... )
            >>> collection.add_variable(new_variable)
        """
        variable = dataset.get_variable_metadata(variable)
        _LOGGER.info('Adding of the %r variable in the collection',
                     variable.name)
        if self.partition_properties.dim not in variable.dimensions:
            raise ValueError(
                'The new variable must use the partitioning axis.')
        self.metadata.add_variable(variable)
        self._write_config()

        # Remove the attribute from the variable. The attribute will be added
        # from the collection metadata.
        variable = variable.set_for_insertion()

        client: dask.distributed.Client = dask_utils.get_client()

        template: meta.Variable = self.metadata.search_same_dimensions_as(
            variable)
        chunks: dict[str, int] = {
            dim.name: dim.value
            for dim in self.metadata.chunks
        }
        try:
            bag: dask.bag.core.Bag = self._bag_from_partitions(lock=True)
            futures: list[
                dask.distributed.Future] = dask.distributed.futures_of(
                    bag.map(storage.add_zarr_array,
                            variable,
                            template.name,
                            self.fs,
                            chunks=chunks).persist())
            storage.execute_transaction(client, self.synchronizer, futures)
        except Exception:
            self.drop_variable(variable.name)
            raise

    def copy(
        self,
        target: str,
        *,
        filters: PartitionFilter | None = None,
        filesystem: fsspec.AbstractFileSystem | None = None,
        mode: Literal['r', 'w'] = 'w',
        npartitions: int | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> Collection:
        """Copy the collection to a new location.

        Args:
            target: The target location.
            filters: The predicate used to filter the partitions to copy.
            filesystem: The file system to use. If None, the file system of the
                collection is used.
            mode: The mode used to open the collection copied. Default is 'w'.
            npartitions: The number of partitions top copy in parallel. Default
                is number of cores.
            synchronizer: The synchronizer used to synchronize the collection
                copied. Default is None.

        Returns:
            The new collection.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="r")
            >>> collection.copy(target="my_new_collection")
        """
        _LOGGER.info('Copying of the collection to %r', target)
        if filesystem is None:
            filesystem = fs_utils.get_fs(target)
        client: dask.distributed.Client = dask_utils.get_client()
        npartitions = npartitions or dask_utils.dask_workers(client,
                                                             cores_only=True)

        # Sequence of (source, target) to copy split in npartitions
        args = tuple(
            dask_utils.split_sequence(
                [(item,
                  fs_utils.join_path(
                      target,
                      os.path.relpath(item, self.partition_properties.dir)))
                 for item in self.partitions(filters=filters)], npartitions))
        # Copy the selected partitions
        partial = functools.partial(fs_utils.copy_tree,
                                    fs_source=self.fs,
                                    fs_target=filesystem)

        def worker_task(args: Sequence[tuple[str, str]]) -> None:
            """Function call on each worker to copy the partitions."""
            tuple(map(lambda arg: partial(*arg), args))

        client.gather(client.map(worker_task, args))
        # Then the remaining files in the root directory (config, metadata,
        # etc.)
        fs_utils.copy_files([
            item['name']
            for item in self.fs.listdir(self.partition_properties.dir,
                                        detail=True) if item['type'] == 'file'
        ], target, self.fs, filesystem)
        return Collection.from_config(target,
                                      mode=mode,
                                      filesystem=filesystem,
                                      synchronizer=synchronizer)
