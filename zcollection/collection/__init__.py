# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Collection of Zarr groups
=========================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
import datetime
import functools
import json
import logging
import os
import warnings

import dask.bag.core
import dask.distributed
import dask.utils
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
from ..partitions import FILENAME as INDEX_FILENAME, PartitionIndex
from .abc import (
    IMMUTABLE,
    Indexer,
    PartitionFilter,
    ReadOnlyCollection,
    WritableCollection,
)
from .callable_objects import (
    PartitionInfo,
    UpdateCallable,
    WrappedPartitionCallable,
)
from .detail import (
    PartitionSlice,
    _insert,
    _load_dataset,
    _load_dataset_with_overlap,
    _try_infer_callable,
    infer_variables,
    wrap_update,
)

__all__ = (
    'IMMUTABLE',
    'Indexer',
    'PartitionFilter',
    'PartitionInfo',
    'PartitionSlice',
    'ReadOnlyCollection',
    'UpdateCallable',
    'WrappedPartitionCallable',
    '_insert',
    '_try_infer_callable',
    'dask_utils',
    'dataset',
    'fs_utils',
    'merging',
    'meta',
    'partitioning',
    'storage',
    'sync',
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import fsspec

    from ..type_hints import ArrayLike
    from ..variable import abc as variable_abc

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)


@storage.skip_absent_partition
def _copy_partition(source: str, target: str,
                    fs_source: fsspec.AbstractFileSystem,
                    fs_target: fsspec.AbstractFileSystem) -> None:
    """Copy a partition, skipping an absent source."""
    try:
        fs_utils.copy_tree(source, target, fs_source, fs_target)
    except FileNotFoundError as err:
        if fs_source.exists(source):
            raise
        raise storage.AbsentPartition(source) from err


def _check_partition(
    partition: str,
    fs: fsspec.AbstractFileSystem,
) -> tuple[str, bool]:
    """Check if a given partition is a valid Zarr group.

    Args:
        partition: The partition to check.
        fs: The file system to use.

    Returns:
        A tuple containing the partition and a boolean indicating whether it is
        a valid Zarr group.
    """
    return partition, storage.check_zarr_group(partition, fs)


class Collection(ReadOnlyCollection, WritableCollection):
    """This class manages a collection of files in Zarr format stored in a set
    of subdirectories. These subdirectories split the data, by cycles or dates
    for example, in order to optimize access and updates, deletion or addition
    of new data.

    Args:
        axis: The axis of the collection. This is the dimension along which the
            data is partitioned.
        ds: The dataset containing the collection. This dataset is used to
            create the metadata of the collection, which is used to validate
            datasets that are inserted in the collection. The dimension named
            by ``axis`` grows with each insertion; the size of every other
            dimension is the one this dataset has, and holds for the life of
            the collection.
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
        ds: meta.DatasetSchema,
        partition_handler: partitioning.Partitioning,
        partition_base_dir: str,
        *,
        mode: Literal['r', 'w'] | None = None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> None:
        super().__init__(axis,
                         ds,
                         partition_handler,
                         partition_base_dir,
                         mode=mode,
                         filesystem=filesystem,
                         synchronizer=synchronizer)

        if self.mode == 'r':

            # These methods are overloaded when the collection is opened in
            # readonly.
            self._read_only_mode()

        else:
            with self.synchronizer:
                new_collection = not self.fs.exists(
                    self._config(self.partition_properties.dir))
                storage.init_zarr_group(self._immutable, self.fs)
                self._write_config(skip_if_exists=True)
                if new_collection:
                    self._index.write(self._index.walk())

    def __str__(self) -> str:
        return (f'<{self.__class__.__name__} '
                f'filesystem={self.fs.__class__.__name__!r}, '
                f'partition_base_dir={self.partition_properties.dir!r}'
                f'mode={self.mode!r}>')

    def _read_only_mode(self) -> None:
        """The collection is read-only, so refuse each method that writes."""
        self._refuse_writes((
            'add_dimension',
            'add_variable',
            'drop_dimension',
            'drop_partitions',
            'drop_variable',
            'insert',
            'reindex',
            'update',
            'update_immutable',
        ))

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
            'version': meta.FORMAT_VERSION,
        }

        fs_utils.replace_file(self.fs, config, json.dumps(params, indent=4))

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

        version = meta.format_version(data)
        load_dataset = meta.DatasetSchema.from_config

        if version > meta.FORMAT_VERSION:
            raise ValueError(
                f'This collection is written in version {version} of the '
                'layout. This version of zcollection reads up to version '
                f'{meta.FORMAT_VERSION}.')

        if version == meta.DEPRECATED_FORMAT_VERSION:
            msg_import = (
                "Use the 'zcollection.migration.migrate' function to copy "
                'it into a collection this release writes.')
            warnings.warn(
                message=('This collection was written by an earlier release '
                         f'and can only be read. {msg_import}'),
                category=UserWarning,
                stacklevel=2)
            if mode == 'w':
                raise ValueError('This collection was written by an earlier '
                                 f'release. {msg_import}')

            load_dataset = meta.DatasetSchema.from_deprecated_config

        collection = Collection(
            data['axis'],
            load_dataset(data['dataset']),
            partitioning.get_codecs(data['partitioning']),
            path,
            mode=mode or 'r',
            filesystem=fs,
            synchronizer=synchronizer,
        )
        collection.version = version

        return collection

    def insert(
        self,
        ds: xarray.Dataset | dataset.Dataset,
        *,
        merge_callable: merging.MergeCallable | None = None,
        npartitions: int | None = None,
        delayed: bool | None = None,
        distributed: bool = True,
        **kwargs,
    ) -> tuple[str, ...]:
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
            delayed: Whether the merge callable receives the stored
                partition and the inserted one as Dask arrays. Defaults to
                ``None``, which gives it Dask arrays when ``ds`` holds them.
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
            The path of each partition written, relative to the collection.
            :py:meth:`partitions` reads the same paths, and
            ``selected_partitions`` takes them.

        Raises:
            ValueError:
                If the dataset does not match the definition of the
                collection, or if ``delayed`` is ``True`` and ``ds`` holds no
                Dask array.

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

        if isinstance(ds, xarray.Dataset):
            ds = dataset.Dataset.from_xarray(ds)

        if delayed and not ds.delayed:
            raise ValueError(
                'delayed is True and the dataset holds no Dask array')

        _LOGGER.info('Inserting of a %s dataset in the collection',
                     dask.utils.format_bytes(ds.nbytes))

        ds = self._set_ds_for_insertion(ds=ds)

        storage.write_zarr_group_missing(ds.select_variables_by_dims(
            (self.dimension, ), False),
                                         self._immutable,
                                         self.fs,
                                         self.synchronizer,
                                         distributed=False)

        # Remove the immutable variables.
        ds = ds.select_variables_by_dims((self.dimension, ))

        if not ds.variables:
            # No variable to insert
            return ()

        # Process the partitions to insert or update by batches to avoid
        # memory issues.
        partitions = tuple(
            self.partitioning.split_dataset(zds=ds, axis=self.dimension))

        written: tuple[str, ...] = tuple(
            fs_utils.join_path(*item) for item, _ in partitions)
        with self.synchronizer, self._index.updating():
            if distributed:
                self._insert_distributed(ds,
                                         partitions,
                                         npartitions,
                                         merge_callable,
                                         delayed=delayed,
                                         **kwargs)
            else:
                self._insert_sequential(ds,
                                        partitions,
                                        merge_callable,
                                        delayed=delayed,
                                        **kwargs)
            self._index.add(written)
        return written

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

        for name, dim in mds.dimensions.items():
            if name == self.dimension:
                continue
            if name in ds.dimensions and dim.value != ds.dimensions[name]:
                raise ValueError(
                    f'Inserted dimension {dim.name} has invalid size '
                    f'({ds.dimensions[name]} instead of {dim.value})')

        ds = dataset.Dataset([
            self._set_var_for_insertion(var) for var in ds.variables.values()
        ])

        for name, dim in mds.dimensions.items():
            if name != self.dimension:
                ds.dimensions.setdefault(name, dim.value)

        if self.dimension in ds.dimensions:
            missing_variables = set(mds.missing_variables(
                ds.metadata())) - self.immutable_variables
            for item in missing_variables:
                ds.add_variable(mds.variables[item], filler=True)

        ds.copy_properties(ds=mds)

        return ds

    def _set_var_for_insertion(self,
                               variable: variable_abc.T) -> variable_abc.T:
        """Create a new variable ready to be inserted into a collection.

        The attributes, the compressor, the filters and the fill value are
        taken from the metadata of the collection, which stores and names its
        variables the same way in every partition. The data type and dimensions
        are checked against it, and so is the fill value when the variable
        names one.

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

        if variable.dimensions != var_meta.dimensions:
            raise ValueError(
                f"Variable '{variable.name}' has invalid dimensions "
                f'({variable.dimensions!r} instead of {var_meta.dimensions!r})'
            )

        if variable.fill_value is not None and zvariable.abc.not_equal(
                variable.fill_value, var_meta.fill_value):
            raise ValueError(
                f"Variable '{variable.name}' has invalid fill_value "
                f"('{variable.fill_value}' instead of '{var_meta.fill_value}')"
            )

        return variable.duplicate(attrs=(),
                                  compressor=var_meta.compressor,
                                  fill_value=var_meta.fill_value,
                                  filters=var_meta.filters)

    def _insert_distributed(self,
                            ds: xarray.Dataset | dataset.Dataset,
                            partitions: tuple[PartitionSlice, ...],
                            npartitions: int | None,
                            merge_callable: merging.MergeCallable | None,
                            *,
                            delayed: bool | None = None,
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
                    delayed=delayed,
                    **kwargs,
                ) for partition in sequence
            ]
            storage.execute_transaction(client, sync.NoSync(), futures)

    def _insert_sequential(self,
                           ds: dataset.Dataset,
                           partitions: tuple[PartitionSlice, ...],
                           merge_callable: merging.MergeCallable | None,
                           *,
                           delayed: bool | None = None,
                           **kwargs):
        """Insert a dataset into the collection without using dask."""
        for partition in partitions:
            _insert(args=partition,
                    axis=self.axis,
                    zds=ds,
                    fs=self.fs,
                    merge_callable=merge_callable,
                    partitioning_properties=self.partition_properties,
                    delayed=delayed,
                    distributed=False,
                    **kwargs)

    def drop_partitions(
        self,
        *,
        filters: PartitionFilter = None,
        timedelta: datetime.timedelta | None = None,
        distributed: bool = True,
    ) -> Iterable[str]:
        """Drop the selected partitions.

        Args:
            filters: The predicate used to filter the partitions to drop. To
                get more information on the predicate, see the documentation of
                the :meth:`partitions` method.
            timedelta: Select the partitions created before the specified time
                delta relative to the current time.
                This option can only be used on filesystem supporting
                the ``fs.created()`` method.
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

        if timedelta is not None:

            def _is_created_before(_path: str, _now: datetime.datetime,
                                   _timedelta: datetime.timedelta) -> bool:
                """Return whether the partition was created before the
                timedelta."""
                created: datetime.datetime = self.fs.created(_path)
                if created.tzinfo is not None:
                    created = created.replace(tzinfo=None)
                return _now - created > _timedelta

            folders = list(
                filter(
                    lambda _folder: _is_created_before(
                        _path=_folder, _now=now, _timedelta=timedelta),
                    folders))

        def invalidate_cache(path) -> None:
            """Invalidate the cache."""
            _LOGGER.info('Dropped partition: %s', path)
            self.fs.invalidate_cache(path)

        with self.synchronizer, self._index.updating():
            if distributed:
                client: dask.distributed.Client = dask_utils.get_client()
                storage.execute_transaction(
                    client, sync.NoSync(),
                    client.map(
                        functools.partial(fs_utils.remove_tree, self.fs),
                        folders))
            else:
                for folder in folders:
                    fs_utils.remove_tree(self.fs, folder)

            tuple(map(invalidate_cache, folders))
            self._index.remove(folders)
        return folders

    def _select_update_partitions(
        self,
        filters: PartitionFilter,
        selected_partitions: Iterable[str] | None,
    ) -> tuple[str, ...]:
        """Select the collection partitions for an update."""
        return tuple(
            self.partitions(filters=filters,
                            selected_partitions=selected_partitions,
                            lock=True))

    def _prepare_update(
        self,
        func: UpdateCallable,
        partitions: tuple[str, ...],
        *,
        variables: Sequence[str] | None,
        selected_variables: Iterable[str] | None,
        delayed: bool,
        depth: int,
        trim: bool,
        client: dask.distributed.Client | None,
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
    ) -> WrappedPartitionCallable | None:
        """Validate output names and build the function for partition updates."""
        dim = self.dimension
        immutable = self.immutable_path if self.have_immutable else None
        load = functools.partial(_load_dataset,
                                 delayed,
                                 self.fs,
                                 immutable,
                                 selected_variables=selected_variables)
        if not variables:
            infer = functools.partial(infer_variables,
                                      partitions,
                                      load,
                                      func,
                                      dim,
                                      func_args,
                                      func_kwargs,
                                      self.synchronizer,
                                      partitioning=self.partitioning)
            variables = client.submit(
                infer, pure=False).result() if client else infer()
        if not variables:
            return None
        unknown_variables = set(variables) - set(self.metadata.variables)
        if unknown_variables:
            raise ValueError(f'Unknown variables: {unknown_variables}')
        immutable_update = self.immutable_variables.intersection(variables)
        if immutable_update:
            raise ValueError(f'Immutable variables ({immutable_update}) '
                             'have to be updated using the '
                             "'Collection.update_immutable' method.")
        _LOGGER.info('Updating variable %s',
                     ', '.join(repr(item) for item in variables))
        load_overlap = functools.partial(
            _load_dataset_with_overlap,
            delayed=delayed,
            depth=depth,
            dim=dim,
            axis=self.axis,
            fs=self.fs,
            immutable=immutable,
            partitions=partitions,
            selected_variables=selected_variables) if depth else None
        return wrap_update(func,
                           load,
                           load_overlap=load_overlap,
                           partitioning=self.partitioning,
                           dim=dim,
                           fs=self.fs,
                           metadata=self.metadata,
                           trim=trim)

    def update(
        self,
        func: UpdateCallable,
        /,
        *args,
        depth: int = 0,
        filters: PartitionFilter = None,
        variables: Sequence[str] | None = None,
        selected_variables: Iterable[str] | None = None,
        selected_partitions: Iterable[str] | None = None,
        trim: bool = True,
        npartitions: int | None = None,
        delayed: bool | None = None,
        distributed: bool = True,
        **kwargs,
    ) -> None:
        """Update the selected partitions.

        Args:
            func: The function to apply on each partition.
            *args: The positional arguments to pass to the function.
            depth: The depth of the overlap between the partitions. Default is
                0 (no overlap). If depth is greater than 0, the function is
                applied on the partition and its neighbors selected by the
                depth.
            variables: The list of variables updated by the function. If None,
                the variables are inferred by calling the function on the first
                available selected partition, using the requested loading and
                execution modes. This additional call uses that partition alone,
                including when ``depth`` is greater than 0. Default is None.
            selected_variables: A list of variables to load from the collection.
                If None, all variables are loaded.
            selected_partitions: A list of partitions to load (using the
                partition relative path).
            filters: The expression used to filter the partitions to update.
            trim: Whether to keep only the current partition's values from
                the results returned by ``func`` when using overlap. Set it
                to ``False`` if your function already returns only those values.
            npartitions: The number of partitions to update in parallel. By
                default, it is equal to the number of Dask workers available
                when calling this method.
            delayed: If True, load variables as Dask arrays. If False, load
                variables directly into NumPy arrays. If None, use the value of
                ``distributed``. Default is None.
            distributed: Whether to use dask or not. Default To True.
            **kwargs: The keyword arguments to pass to the function.

        Raises:
            ValueError: If the variables to update are not in the collection.

        Notes:
            Each task loads and writes one partition or overlap window at a
            time. With overlap, neighboring partitions may already contain
            values written by another call. Callbacks that read and rewrite
            the same variables must be idempotent and account for these updates.
            Distributed tasks can run in any order. Inference adds a callback
            invocation whose returned values are discarded. A callback
            declaring ``partition_info`` receives a :class:`zcollection.PartitionInfo`
            on every call, including inference and calls without overlap.
            See :doc:`/callbacks` for examples of overlap, trimming and
            external arguments.

        Example:
            For an existing writable collection with input ``var1`` and
            output ``var2``:

            .. code-block:: python

                def ones(ds):
                    return {'var2': ds['var1'].values * 0 + 1}

                collection.update(ones, variables=('var2', ),
                                  selected_variables=('var1', ))
        """
        self._update(func,
                     *args,
                     depth=depth,
                     filters=filters,
                     variables=variables,
                     selected_variables=selected_variables,
                     selected_partitions=selected_partitions,
                     trim=trim,
                     npartitions=npartitions,
                     delayed=delayed,
                     distributed=distributed,
                     **kwargs)

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
            fs_utils.join_path(self._immutable, variable.name), data, self.fs)

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
            ValueError: If a variable uses the dimension.
            KeyError: If the dimension does not exist.
        """
        used_by = sorted(variable.name
                         for variable in self.metadata.variables.values()
                         if dimension in variable.dimensions)
        if used_by:
            raise ValueError(f'The dimension {dimension!r} is used by '
                             f'the variables {used_by!r}.')

        _LOGGER.info('Dropping of the %r dimension from the collection',
                     dimension)

        del self.metadata.dimensions[dimension]
        self._write_config()

    def add_variable(
        self,
        variable: meta.VariableSchema | dataset.Variable,
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
            >>> new_variable = meta.VariableSchema(
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
        self.metadata.add_variable(variable)
        self._write_config()

        # Attributes are not stored at variable's level
        variable = variable.set_for_insertion()

        try:
            if self.dimension not in variable.dimensions:
                self._add_variable(variable, [self._immutable])
            elif distributed:
                self._add_variable_distributed(variable)
            else:
                self._add_variable(variable, self.partitions(lock=True))

        except Exception:
            self.drop_variable(variable.name, distributed)
            raise

    def _add_variable(self, variable: meta.VariableSchema,
                      partitions: Iterable[str]):
        """Add the provided variable to the collection."""
        dimensions, chunks = self.dimensions_properties()

        for partition in partitions:
            storage.add_zarr_array(partition,
                                   variable,
                                   dimensions,
                                   self.fs,
                                   axis=self.axis,
                                   chunks=chunks)

    def _add_variable_distributed(self, variable: meta.VariableSchema):
        """Add the provided variable to the collection using dask."""
        dimensions, chunks = self.dimensions_properties()

        client: dask.distributed.Client = dask_utils.get_client()
        bag: dask.bag.core.Bag = self._bag_from_partitions(lock=True)

        futures: list[dask.distributed.Future] = dask.distributed.futures_of(
            bag.map(storage.add_zarr_array,
                    variable,
                    dimensions,
                    self.fs,
                    axis=self.axis,
                    chunks=chunks).persist())

        storage.execute_transaction(client, self.synchronizer, futures)

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
            storage.del_zarr_array(self._immutable, variable, self.fs)
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
                storage.del_zarr_array(partition, variable, self.fs)

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
        filesystem.makedirs(target, exist_ok=True)

        index = PartitionIndex(filesystem,
                               fs_utils.normalize_path(filesystem, target),
                               self.partitioning,
                               writable=True)
        with synchronizer or sync.NoSync(), index.updating():
            selected = self.partitions(filters=filters)

            if distributed:
                client: dask.distributed.Client = dask_utils.get_client()
                npartitions = npartitions or dask_utils.dask_workers(
                    client, cores_only=True)

                # Sequence of (source, target) to copy split in npartitions
                args = tuple(
                    dask_utils.split_sequence(
                        [(item,
                          fs_utils.join_path(
                              target,
                              os.path.relpath(item,
                                              self.partition_properties.dir)))
                         for item in selected], npartitions))
                # Copy the selected partitions
                partial = functools.partial(_copy_partition,
                                            fs_source=self.fs,
                                            fs_target=filesystem)

                def worker_task(args: Sequence[tuple[str, str]]) -> None:
                    """Function call on each worker to copy the partitions."""
                    tuple(partial(*arg) for arg in args)

                client.gather(client.map(worker_task, args))
            else:
                for source_path in selected:
                    target_path = fs_utils.join_path(
                        target,
                        os.path.relpath(source_path,
                                        self.partition_properties.dir))
                    _copy_partition(source_path,
                                    target_path,
                                    fs_source=self.fs,
                                    fs_target=filesystem)

            # Then the remaining files in the root directory (config, metadata,
            # etc.). The index of the partitions is left out: the copy holds the
            # partitions selected here, not the ones this collection holds.
            fs_utils.copy_files([
                item['name']
                for item in self.fs.listdir(self.partition_properties.dir,
                                            detail=True)
                if item['type'] == 'file'
                and os.path.basename(item['name']) != INDEX_FILENAME
            ], target, self.fs, filesystem)

            index.add(index.path_of(key) for key in index.walk())

        return Collection.from_config(target,
                                      mode=mode,
                                      filesystem=filesystem,
                                      synchronizer=synchronizer)

    def validate_partitions(self,
                            filters: PartitionFilter = None,
                            distributed: bool = True,
                            fix: bool = False) -> list[str]:
        """List the invalid partitions of the collection.

        A partition is invalid when it cannot be read.

        Args:
            filters: The predicate selecting the partitions to check. By
                default, every partition is checked.
            distributed: Whether to use dask or not. Default To True.
            fix: Whether to delete the invalid partitions and remove them
                from the index.

        Returns:
            The path of each invalid partition.
        """
        partitions = tuple(self.partitions(filters=filters))
        if not partitions:
            return []

        invalid_partitions: list[str] = []

        def _validity_check(partition: str, valid: bool) -> None:
            """Warn about the partition and keep its path when it is not
            valid."""
            if not valid:
                warnings.warn(f'Invalid partition: {partition}',
                              category=RuntimeWarning,
                              stacklevel=2)
                invalid_partitions.append(partition)

        check = functools.partial(_check_partition, fs=self.fs)
        if distributed:
            client = dask_utils.get_client()
            for item in dask.distributed.as_completed(
                    client.map(check, partitions)):
                _validity_check(*item.result())
        else:
            for item in partitions:
                _validity_check(*check(item))

        if fix and invalid_partitions:
            with self.synchronizer, self._index.updating():
                for item in invalid_partitions:
                    _LOGGER.info('Removing invalid partition: %s', item)
                    fs_utils.remove_tree(self.fs, item)
                self._index.remove(invalid_partitions)
        return invalid_partitions
