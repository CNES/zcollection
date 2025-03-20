# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Collection of Zarr groups
=========================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NoReturn
from collections.abc import Iterable, Iterator, Sequence
import datetime
import functools
import importlib.metadata
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
    variable as zvariable,
)
from ..type_hints import ArrayLike
from .abc import IMMUTABLE, Indexer, PartitionFilter, ReadOnlyCollection
from .callable_objects import UpdateCallable, WrappedPartitionCallable
from .detail import (
    PartitionSlice,
    _insert,
    _try_infer_callable,
    _wrap_update_func,
    _wrap_update_func_with_overlap,
)

__all__ = ('dask_utils', 'dataset', 'fs_utils', 'merging', 'meta',
           'partitioning', 'storage', 'sync', 'Indexer', 'PartitionFilter',
           'ReadOnlyCollection', 'IMMUTABLE', 'UpdateCallable',
           'WrappedPartitionCallable', 'PartitionSlice', '_insert',
           '_try_infer_callable', '_wrap_update_func',
           '_wrap_update_func_with_overlap')

if TYPE_CHECKING:
    from ..variable import abc as variable_abc

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)


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
        return ()

    with collection.synchronizer:
        zds: dataset.Dataset = storage.open_zarr_group(
            dirname=one_partition,
            fs=collection.fs,
            delayed=delayed,
            selected_variables=selected_variables)
    func_result: dict[str, Any]
    func_result = _try_infer_callable(func, zds, collection.dimension, *args,
                                      **kwargs)
    unknown_variables: set[str] = set(func_result) - set(
        collection.metadata.variables.keys())

    if len(unknown_variables):
        raise ValueError(f'Unknown variables: {unknown_variables}')

    return tuple(func_result)


def _check_partition(
    partition: str,
    fs: fsspec.AbstractFileSystem,
    partitioning_strategy: partitioning.Partitioning,
) -> tuple[str, bool]:
    """Check if a given partition is a valid Zarr group.

    Args:
        partition: The partition to check.
        fs: The file system to use.
        partitioning_strategy: The partitioning strategy.

    Returns:
        A tuple containing the partition and a boolean indicating whether it is
        a valid Zarr group.
    """
    try:
        partitioning_strategy.parse(partition)
    except ValueError:
        return partition, False
    return partition, storage.check_zarr_group(partition, fs)


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
            storage.init_zarr_group(dirname=self._immutable, fs=self.fs)

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
                'drop_partitions',
                'add_variable',
                'drop_variable',
                'add_dimension',
                'insert',
                'update',
        ]:
            assert hasattr(self, item), f'{item} is not a known method.'
            setattr(self, item,
                    types.MethodType(Collection._unsupported_operation, self))

    def _write_config(self, skip_if_exists: bool = False) -> None:
        """Write the configuration file."""
        base_dir = self.partition_properties.dir
        config = self._config(base_dir)
        exists = self.fs.exists(config)

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
            'version': importlib.metadata.version('zcollection'),
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
        fs = fs_utils.get_fs(filesystem)
        config = cls._config(path)

        if not fs.exists(config):
            raise ValueError(f'zarr collection not found at path {path!r}')

        with fs.open(config) as stream:
            data: dict[str, Any] = json.load(stream)

        version = data.get('version', '0')
        load_dataset = meta.Dataset.from_config

        if version == '0':
            msg_import = ("Use the 'zcollection.update_deprecated_collection' "
                          'function to update it.')
            warnings.warn(
                message=('This collection needs to be updated and '
                         f'can only be used in read only mode. {msg_import}'),
                category=UserWarning,
                stacklevel=2)
            if mode == 'w':
                raise ValueError(
                    'Collection configuration needs to be updated. '
                    f'{msg_import}')

            load_dataset = meta.Dataset.from_deprecated_config

        collection = Collection(
            axis=data['axis'],
            ds=load_dataset(data['dataset']),
            partition_handler=partitioning.get_codecs(data['partitioning']),
            partition_base_dir=path,
            mode=mode or 'r',
            filesystem=fs,
            synchronizer=synchronizer,
        )
        collection.version = version

        return collection

    # pylint: disable=method-hidden
    def insert(
        self,
        ds: xarray.Dataset | dataset.Dataset,
        *,
        merge_callable: merging.MergeCallable | None = None,
        npartitions: int | None = None,
        distributed: bool = True,
        **kwargs,
    ) -> Iterable[str]:
        """Insert a dataset into the collection.

        Args:
            ds: The dataset to insert. It can be either a xarray.Dataset or a
                dataset.Dataset object.
            merge_callable: A function to use to merge the existing data set
                already stored in partitions with the new partitioned data. If
                None, the new partitioned data overwrites the existing
                partitioned data.
            npartitions: The maximum number of partitions to process in
                parallel. By default, partitions are processed one by one.
            distributed: Whether to use dask or not. Default To True.
            kwargs: Additional keyword arguments passed to the merge callable.

                .. note::

                    When inserting partitions, Dask parallelizes the writing of
                    each partition across its workers. Additionally, the writing
                    of variables within a partition is parallelized on the
                    worker responsible for inserting that partition, using
                    multiple threads. If you're using a single Dask worker,
                    partition insertion will happen sequentially and changing
                    this parameter will have no effect.

        Returns:
            A list of the inserted partitions.

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

        storage.write_zarr_group_missing(
            zds=ds.select_variables_by_dims(
                dims=(self.dimension, ),
                predicate=False,
            ),
            dirname=self._immutable,
            fs=self.fs,
            synchronizer=self.synchronizer,
            distributed=False,
        )

        # Remove the immutable variables.
        ds = ds.select_variables_by_dims((self.dimension, ))

        if not ds.variables:
            # No variable to insert
            return tuple()

        ds = self._set_ds_for_insertion(ds=ds)

        # Process the partitions to insert or update by batches to avoid
        # memory issues.
        partitions = tuple(
            self.partitioning.split_dataset(zds=ds, axis=self.dimension))

        if distributed:
            self._insert_distributed(ds=ds,
                                     partitions=partitions,
                                     npartitions=npartitions,
                                     merge_callable=merge_callable,
                                     **kwargs)
        else:
            self._insert_sequential(ds=ds,
                                    partitions=partitions,
                                    merge_callable=merge_callable,
                                    **kwargs)

        return (fs_utils.join_path(*((self.partition_properties.dir, ) + item))
                for item, _ in partitions)

    def _set_ds_for_insertion(self, ds: dataset.Dataset) -> dataset.Dataset:
        """Create a new dataset ready to be inserted into a collection.

         * Missing collection's variables and dimensions are added
           to the dataset
         * Dimension's size are checked
         * Each variable is checked

        Args:
            ds: Dataset to insert.

        Returns:
            New dataset.
        """
        mds = self.metadata

        ds = dataset.Dataset(variables=[
            self._set_var_for_insertion(var) for var in ds.variables.values()
        ])

        missing_variables = set(mds.missing_variables(
            ds.metadata())) - self.immutable_variables

        for name, dim in mds.dimensions.items():
            if name == self.dimension:
                continue

            if name not in ds.dimensions:
                # Adding missing dimensions properties
                ds.dimensions[name] = dim.value
            elif dim.value != ds.dimensions[name]:
                raise ValueError(
                    f'Inserted dimension {dim.name} has invalid size '
                    f'({ds.dimensions[name]} instead of {dim.value})')

        for item in missing_variables:
            variable = mds.variables[item]
            ds.add_variable(variable)

        ds.copy_properties(ds=mds)

        return ds

    def _set_var_for_insertion(self,
                               variable: variable_abc.T) -> variable_abc.T:
        """Create a new variable ready to be inserted into a collection.
        Variable's dtype and fill_value are checked for consistency.

        Args:
            variable: The variable to insert.

        Returns:
            The variable.
        """
        var_meta = self.metadata.variables.get(variable.name, None)

        if var_meta is None:
            raise ValueError(f"Variable '{variable.name}' is unknown.")

        if variable.dtype != var_meta.dtype:
            raise ValueError(
                f"Variable '{variable.name}' has invalid dtype "
                f"('{variable.dtype}' instead of '{var_meta.dtype}')")

        if variable.fill_value != var_meta.fill_value:
            raise ValueError(
                f"Variable '{variable.name}' has invalid fill_value "
                f"('{variable.fill_value}' instead of '{var_meta.fill_value}')"
            )

        return zvariable.new_variable(cls=type(variable),
                                      name=variable.name,
                                      array=variable.array,
                                      dimensions=variable.dimensions,
                                      attrs=(),
                                      compressor=variable.compressor,
                                      fill_value=variable.fill_value,
                                      filters=variable.filters)

    def _insert_distributed(self, ds: xarray.Dataset | dataset.Dataset,
                            partitions: tuple[PartitionSlice,
                                              ...], npartitions: int | None,
                            merge_callable: merging.MergeCallable | None,
                            **kwargs):
        """Insert a dataset into the collection using dask."""
        if npartitions is not None:
            if npartitions < 1:
                raise ValueError('The number of partitions must be positive')
            npartitions = len(partitions) // npartitions + 1

        client: dask.distributed.Client = dask_utils.get_client()
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
                    **kwargs,
                ) for partition in sequence
            ]
            storage.execute_transaction(client, self.synchronizer, futures)

    def _insert_sequential(self, ds: xarray.Dataset | dataset.Dataset,
                           partitions: tuple[PartitionSlice, ...],
                           merge_callable: merging.MergeCallable | None,
                           **kwargs):
        """Insert a dataset into the collection without using dask."""
        ds = ds.compute()
        for partition in partitions:
            _insert(args=partition,
                    axis=self.axis,
                    zds=ds,
                    fs=self.fs,
                    merge_callable=merge_callable,
                    partitioning_properties=self.partition_properties,
                    distributed=False,
                    **kwargs)

    # pylint: disable=method-hidden
    def drop_partitions(
        self,
        *,
        filters: PartitionFilter = None,
        timedelta: datetime.timedelta | None = None,
        distributed: bool = True,
    ) -> Iterable[str]:
        # pylint: disable=method-hidden
        """Drop the selected partitions.

        Args:
            filters: The predicate used to filter the partitions to drop. To
                get more information on the predicate, see the documentation of
                the :meth:`partitions` method.
            timedelta: Select the partitions created before the specified time
                delta relative to the current time.
            distributed: Whether to use dask or not. Default To True.

        Returns:
            A list of the dropped partitions.

        Example:
            >>> collection.drop_partitions(filters="year == 2019")
            >>> collection.drop_partitions(
            ...     timedelta=datetime.timedelta(days=30))
        """
        now: datetime.datetime = datetime.datetime.now()
        folders = list(self.partitions(filters=filters, lock=True))

        # No partition selected, nothing to do.
        if not folders:
            return folders

        def _is_created_before(_path: str, _now: datetime.datetime,
                               _timedelta: datetime.timedelta) -> bool:
            """Return whether the partition was created before the
            timedelta."""
            created: datetime.datetime = self.fs.created(_path)
            if created.tzinfo is not None:
                created = created.replace(tzinfo=None)
            return _now - created > _timedelta

        if timedelta is not None:
            folders = list(
                filter(
                    lambda _folder: _is_created_before(
                        _path=_folder, _now=now, _timedelta=timedelta),
                    folders))

        if distributed:
            client: dask.distributed.Client = dask_utils.get_client()
            storage.execute_transaction(
                client, self.synchronizer,
                client.map(self.fs.rm, folders, recursive=True))
        else:
            for folder in folders:
                self.fs.rm(path=folder, recursive=True)

        def invalidate_cache(path) -> None:
            """Invalidate the cache."""
            _LOGGER.info('Dropped partition: %s', path)
            self.fs.invalidate_cache(path)

        tuple(map(invalidate_cache, folders))
        return folders

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
        variables: Sequence[str] | None = None,
        distributed: bool = True,
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
            variables: The list of variables updated by the function. If None,
                the variables are inferred by calling the function on the first
                partition. In this case, it is important to ensure that the
                function can be called twice on the same partition without
                side effects. Default is None.
            distributed: Whether to use dask or not. Default To True.
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

        # Delayed has to be False if dask is disabled
        if not distributed:
            delayed = False

        variables = variables or _infer_callable(
            self, func, filters, delayed, selected_variables, *args, **kwargs)

        if not variables:
            warnings.warn('You are trying to update an empty collection.',
                          category=RuntimeWarning,
                          stacklevel=2)
            return

        immutable_update = self.immutable_variables.intersection(variables)

        if immutable_update:
            raise ValueError(f'Immutable variables ({immutable_update}) '
                             'have to be updated using the '
                             "'Collection.update_immutable' method.")

        # If depth is not 0, the variables updated must be in the selected
        # variables.
        if depth != 0 and selected_variables is not None:
            selected_variables += list(
                set(variables) - set(selected_variables))

        _LOGGER.info('Updating of the (%s) variable in the collection',
                     ', '.join(repr(item) for item in variables))

        selected_partitions = tuple(self.partitions(filters=filters,
                                                    lock=True))

        immutable = self._immutable if self.have_immutable else None

        if depth == 0:
            local_func = _wrap_update_func(
                delayed=delayed,
                func=func,
                fs=self.fs,
                immutable=immutable,
                selected_variables=selected_variables)
        else:
            local_func = _wrap_update_func_with_overlap(
                delayed=delayed,
                depth=depth,
                dim=self.dimension,
                func=func,
                fs=self.fs,
                immutable=immutable,
                selected_partitions=selected_partitions,
                selected_variables=selected_variables,
                trim=trim)

        if distributed:
            client: dask.distributed.Client = dask_utils.get_client()

            batches: Iterator[Sequence[str]] = dask_utils.split_sequence(
                sequence=selected_partitions,
                sections=npartitions
                or dask_utils.dask_workers(client=client, cores_only=True))

            storage.execute_transaction(
                client=client,
                synchronizer=self.synchronizer,
                futures=client.map(
                    local_func,
                    tuple(batches),
                    key=func.__name__,
                    func_args=args,
                    func_kwargs=kwargs,
                ),
            )
        else:
            local_func(selected_partitions, args, kwargs)

        tuple(map(self.fs.invalidate_cache, selected_partitions))

    def update_immutable(self, name: str, data: ArrayLike) -> None:
        """Update an immutable variable with provided data..

        Args:
            name: Name of the variable to update.
            data: Immutable variable data.
        """
        dimensions, _ = self.dimensions_properties()
        variable = self.metadata.variables.get(name, None)

        if variable is None:
            raise ValueError(f'Immutable variable ({name}) does not exist.')

        # Validate dimensions size
        for i, dim in enumerate(variable.dimensions):
            if dimensions[dim] != data.shape[i]:
                raise ValueError(
                    f"Added variable '{variable.name}' contains a "
                    f"dimension '{dim}' with an invalid size "
                    f'({data.shape[i]} instead of {dimensions[dim]})')

        storage.update_zarr_array(
            dirname=fs_utils.join_path(self._immutable, variable.name),
            array=data,
            fs=self.fs,
        )

    def add_dimension(self, dimension: meta.Dimension) -> None:
        """Add a dimension to the collection.

        Args:
            dimension: The dimension to add.

        Raises:
            ValueError: if the dimension is already part of the collection.

        Example:
            >>> import zcollection
            >>> import numpy
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="w")
            >>> new_dimension = meta.Dimension(
            ...     name="t",
            ...     value=5,
            ... )
            >>> collection.add_dimension(new_dimension)
        """
        _LOGGER.info(
            'Adding of the %r dimension to the collection'
            ' (size: %r, chunks: %r)', dimension.name, dimension.value,
            dimension.chunks)
        self.metadata.add_dimension(dimension)
        self._write_config()

    def drop_dimension(self, dimension: str) -> None:
        """Drop a dimension from the collection.

        Args:
            dimension: The dimension to drop.

        Raises:
            ValueError: if the dimension is already part of the collection.
        """
        _LOGGER.info('Dropping of the %r dimension from the collection',
                     dimension)

        del self.metadata.dimensions[dimension]
        self._write_config()

    def add_variable(
        self,
        variable: meta.Variable | dataset.Variable,
        distributed: bool = True,
    ) -> None:
        """Add a variable to the collection.

        Args:
            variable: The variable to add.
            distributed: Whether to use dask or not. Default To True.

        Raises:
            ValueError: if the variable is already part of the collection, it
                uses a dimension that is not part of the dataset.

        Example:
            >>> import zcollection
            >>> import numpy
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
        _LOGGER.info('Adding of the %r variable in the collection',
                     variable.name)

        variable = dataset.get_variable_metadata(variable)
        self.metadata.add_variable(variable)
        self._write_config()

        # Attributes are not stored at variable's level
        variable = variable.set_for_insertion()

        try:
            if self.dimension not in variable.dimensions:
                self._add_variable(variable=variable,
                                   partitions=[self._immutable])
            elif distributed:
                self._add_variable_distributed(variable=variable)
            else:
                self._add_variable(variable=variable,
                                   partitions=self.partitions(lock=True))

        except Exception:
            self.drop_variable(variable=variable.name, distributed=distributed)
            raise

    def _add_variable(self, variable: meta.Variable,
                      partitions: Iterable[str]):
        """Add the provided variable to the collection."""
        dimensions, chunks = self.dimensions_properties()

        for partition in partitions:
            storage.add_zarr_array(dirname=partition,
                                   variable=variable,
                                   dimensions=dimensions,
                                   fs=self.fs,
                                   axis=self.axis,
                                   chunks=chunks)

    def _add_variable_distributed(self, variable: meta.Variable):
        """Add the provided variable to the collection using dask."""
        dimensions, chunks = self.dimensions_properties()

        client: dask.distributed.Client = dask_utils.get_client()
        bag: dask.bag.core.Bag = self._bag_from_partitions(lock=True)

        futures: list[dask.distributed.Future] = dask.distributed.futures_of(
            bag.map(storage.add_zarr_array,
                    variable=variable,
                    dimensions=dimensions,
                    fs=self.fs,
                    axis=self.axis,
                    chunks=chunks).persist())

        storage.execute_transaction(client=client,
                                    synchronizer=self.synchronizer,
                                    futures=futures)

    def drop_variable(self, variable: str, distributed: bool = True) -> None:
        """Delete the variable from the collection.

        Args:
            variable: The variable to delete.
            distributed: Whether to use dask or not. Default To True.

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

        if self.dimension not in self.metadata.variables[variable].dimensions:
            # This is an immutable variable.
            storage.del_zarr_array(dirname=self._immutable,
                                   name=variable,
                                   fs=self.fs)
        elif distributed:
            client: dask.distributed.Client = dask_utils.get_client()
            bag: dask.bag.core.Bag = self._bag_from_partitions(lock=True)
            awaitables: list[
                dask.distributed.Future] = dask.distributed.futures_of(
                    bag.map(storage.del_zarr_array, variable,
                            self.fs).persist())
            storage.execute_transaction(client, self.synchronizer, awaitables)
        else:
            for partition in self.partitions(lock=True):
                storage.del_zarr_array(dirname=partition,
                                       name=variable,
                                       fs=self.fs)

        del self.metadata.variables[variable]
        self._write_config()

    def copy(
        self,
        target: str,
        *,
        filters: PartitionFilter | None = None,
        filesystem: fsspec.AbstractFileSystem | None = None,
        mode: Literal['r', 'w'] = 'w',
        npartitions: int | None = None,
        synchronizer: sync.Sync | None = None,
        distributed: bool = True,
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
            distributed: Whether to use dask or not. Default To True.

        Returns:
            The new collection.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="r")
            >>> collection.copy(target="my_new_collection")
        """
        _LOGGER.info('Copying of the collection to %r', target)
        filesystem = filesystem or fs_utils.get_fs(target)

        partitions = self.partitions(filters=filters)

        if distributed:
            client: dask.distributed.Client = dask_utils.get_client()
            npartitions = npartitions or dask_utils.dask_workers(
                client, cores_only=True)

            # Sequence of (source, target) to copy split in npartitions
            args = tuple(
                dask_utils.split_sequence([
                    (item,
                     fs_utils.join_path(
                         target,
                         os.path.relpath(item, self.partition_properties.dir)))
                    for item in partitions
                ], npartitions))
            # Copy the selected partitions
            partial = functools.partial(fs_utils.copy_tree,
                                        fs_source=self.fs,
                                        fs_target=filesystem)

            def worker_task(args: Sequence[tuple[str, str]]) -> None:
                """Function call on each worker to copy the partitions."""
                tuple(map(lambda arg: partial(*arg), args))

            client.gather(client.map(worker_task, args))
        else:
            for source_path in partitions:
                target_path = fs_utils.join_path(
                    target,
                    os.path.relpath(source_path,
                                    self.partition_properties.dir))
                fs_utils.copy_tree(source=source_path,
                                   target=target_path,
                                   fs_source=self.fs,
                                   fs_target=filesystem)

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

    def validate_partitions(self,
                            filters: PartitionFilter | None = None,
                            distributed: bool = True,
                            fix: bool = False) -> list[str]:
        """Validates partitions in the collection by checking if they exist and
        are readable. If `fix` is True, invalid partitions will be removed from
        the collection.

        Args:
            filters: The predicate used to filter the partitions to
                validate. By default, all partitions are validated.
            fix: Whether to fix invalid partitions by removing them from
                the collection.
            distributed: Whether to use dask or not. Default To True.

        Returns:
            A list of invalid partitions.
        """
        partitions = tuple(self.partitions(filters=filters))
        if not partitions:
            return []

        invalid_partitions: list[str] = []

        def _validity_check(_partition, _valid):
            """Check partition validity and add it to invalid partitions if not
            valid."""
            if not _valid:
                warnings.warn(f'Invalid partition: {_partition}',
                              category=RuntimeWarning)
                invalid_partitions.append(_partition)

        if distributed:
            client: dask.distributed.Client = dask_utils.get_client()
            futures: list[dask.distributed.Future] = client.map(
                _check_partition,
                partitions,
                fs=self.fs,
                partitioning_strategy=self.partitioning)

            for item in dask.distributed.as_completed(futures):
                partition, valid = item.result()  # type: ignore
                _validity_check(_partition=partition, _valid=valid)
        else:
            for partition in partitions:
                partition, valid = _check_partition(
                    partition=partition,
                    fs=self.fs,
                    partitioning_strategy=self.partitioning)
                _validity_check(_partition=partition, _valid=valid)

        if fix and invalid_partitions:
            for item in invalid_partitions:
                _LOGGER.info('Removing invalid partition: %s', item)
                self.fs.rm(item, recursive=True)
        return invalid_partitions
