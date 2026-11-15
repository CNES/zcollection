# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Collection of Zarr groups
=========================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
import contextlib
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
    execution,
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


def _copy_partition(
        paths: tuple[str, str], *, fs_source: fsspec.AbstractFileSystem,
        fs_target: fsspec.AbstractFileSystem) -> execution.Skipped | None:
    """Copy a partition, skipping an absent source with a warning."""
    source, target = paths
    try:
        fs_utils.copy_tree(source, target, fs_source, fs_target)
    except FileNotFoundError:
        if fs_source.exists(source):
            raise
        storage.warn_absent_partition(source)
        return execution.SKIPPED
    return None


def _drop_partition(path: str,
                    fs: fsspec.AbstractFileSystem) -> execution.Skipped | None:
    """Remove the directory of a partition, skipping an absent one."""
    if not fs.exists(path):
        return execution.SKIPPED
    fs_utils.remove_tree(fs, path)
    _LOGGER.info('Dropped partition: %s', path)
    return None


def _add_array(partition: str, *, variable: meta.VariableSchema,
               dimensions: meta.DimensionType, fs: fsspec.AbstractFileSystem,
               axis: str, chunks: dict[str, int]) -> execution.Skipped | None:
    """Create the array of a variable in a partition, skipping an absent
    partition."""
    if not fs.exists(partition):
        return execution.SKIPPED
    storage.add_zarr_array(partition,
                           variable,
                           dimensions,
                           fs,
                           axis=axis,
                           chunks=chunks)
    return None


def _drop_array(partition: str, *, variable: str,
                fs: fsspec.AbstractFileSystem) -> execution.Skipped | None:
    """Remove the array of a variable from a partition, skipping an absent
    partition."""
    if not fs.exists(partition):
        return execution.SKIPPED
    storage.del_zarr_array(partition, variable, fs)
    return None


def _check_partition(partition: str, *, fs: fsspec.AbstractFileSystem,
                     variables: tuple[str, ...]) -> execution.Skipped | None:
    """Check a partition and identify whether it is valid.

    Return ``SKIPPED`` for a valid partition and ``None`` otherwise.

    Args:
        partition: The partition to check.
        fs: The file system storing the partition.
        variables: The arrays required in the partition.
    """
    if storage.check_zarr_group(partition, fs, variables):
        return execution.SKIPPED
    return None


class Collection(ReadOnlyCollection, WritableCollection):
    """Store a dataset as a collection of partitioned Zarr groups.

    Partitioning by date, cycle, or other keys allows individual parts of
    the dataset to be read and updated without processing the whole collection.

    Args:
        axis: The name of the variable used for partitioning. Its first
            dimension is the partitioned dimension.
        ds: The dataset used to define the collection schema. Inserted
            datasets are validated against it. The partitioned dimension
            can vary in size; all other dimensions retain their declared sizes.
        partition_handler: The partitioning scheme, an instance of
            :py:class:`~zcollection.partitioning.abc.Partitioning`.
        partition_base_dir: The base directory where partitions are stored.
        mode: The access mode: ``'r'`` for read-only or ``'w'`` for reading
            and writing. Defaults to ``'w'``.
        filesystem: An :py:class:`fsspec.AbstractFileSystem` instance or a
            file system protocol name. Defaults to the local file system.
        synchronizer: The synchronizer used for the collection, an instance
            of :py:class:`~zcollection.sync.Sync`.

    Raises:
        ValueError: If ``axis`` or a partitioning key is missing from
            the dataset, or if the access mode is unsupported.

    Notes:
        Use :py:func:`~zcollection.create_collection` and
        :py:func:`~zcollection.open_collection` to create or open collections
        instead of instantiating this class directly.
    """
    _DROP_ARRAY = staticmethod(_drop_array)

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

            # Replace these methods when the collection is opened read-only.
            self._read_only_mode()

        else:
            with self.synchronizer:
                new_collection = not self.fs.exists(
                    self._config(self.partition_properties.dir))
                storage.init_zarr_group(self._immutable, self.fs)
                self._write_config(skip_if_exists=True)
                if new_collection:
                    self._index.reindex()

    def __str__(self) -> str:
        return (f'<{self.__class__.__name__} '
                f'filesystem={self.fs.__class__.__name__!r}, '
                f'partition_base_dir={self.partition_properties.dir!r}'
                f'mode={self.mode!r}>')

    @property
    def _reference(self) -> ReadOnlyCollection:
        return self

    def _read_only_mode(self) -> None:
        """Disable methods that modify a read-only collection."""
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
        """Open a collection from its stored configuration.

        Args:
            path: Path to the collection directory.
            mode: Access mode, ``'r'`` for read-only or ``'w'`` for
                reading and writing. Defaults to ``'r'``.
            filesystem: The file system used for the collection, as a
                :py:class:`fsspec.AbstractFileSystem` instance.
            synchronizer: The synchronizer used for the collection, as a
                :py:class:`~zcollection.sync.Sync` instance.

        Returns:
            The opened collection.

        Raises:
            ValueError: If the configuration is missing, its layout is
                newer than supported, or an older collection is opened
                in write mode.
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
        npartitions: int | None = 1,
        delayed: bool | None = None,
        distributed: bool = True,
        stop_on_error: bool = True,
        **kwargs,
    ) -> tuple[str, ...]:
        """Insert a dataset into the collection.

        Existing partitions are replaced unless a merge callable is supplied.

        Args:
            ds: The dataset to insert, as an ``xarray.Dataset`` or a
                :class:`zcollection.dataset.Dataset`.
            merge_callable: Function for combining existing and inserted
                partition data. Defaults to ``None`` (replace the partition).
            npartitions: Maximum number of partition tasks in progress.
                Defaults to 1, or the number of Dask workers when ``None``.
                With ``distributed=False``, partitions run sequentially.
            delayed: Whether the merge callable receives Dask arrays.
                Defaults to ``None``, which follows the representation
                of the input dataset.
            distributed: Whether to use Dask. Defaults to ``True``.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish.
                Defaults to ``True``.
            kwargs: Additional keyword arguments for the merge callable.

        Returns:
            Paths of the written partitions, relative to the collection.
            :py:meth:`partitions` also returns relative paths when called
            with ``relative=True``. These paths can be passed to
            ``selected_partitions``.

        Raises:
            ValueError: If the dataset is incompatible with the collection,
                ``delayed=True`` is used without Dask arrays, or
                ``npartitions`` is below one.
            PartitionErrors: If a partition fails.

        Notes:
            Partitions are processed independently, but their underlying
            data may share a file or chunk. Storage libraries such as
            HDF5, NetCDF, or Zarr must support concurrent access when
            multiple partitions access the same backing resource.
        """

        if isinstance(ds, xarray.Dataset):
            ds = dataset.Dataset.from_xarray(ds)

        if delayed and not ds.delayed:
            raise ValueError(
                'delayed is True and the dataset holds no Dask array')

        _LOGGER.info('Inserting of a %s dataset in the collection',
                     dask.utils.format_bytes(ds.nbytes))

        ds = self._set_ds_for_insertion(ds=ds)

        immutable = ds.select_variables_by_dims((self.dimension, ), False)
        ds = ds.select_variables_by_dims((self.dimension, ))

        with self.synchronizer:
            storage.write_zarr_group_missing(immutable,
                                             self._immutable,
                                             self.fs,
                                             sync.NoSync(),
                                             distributed=False)
            if not ds.variables:
                return ()
            partitions = tuple(
                self.partitioning.split_dataset(zds=ds, axis=self.dimension))
            if not partitions:
                return ()
            client = dask_utils.get_client() if distributed else None
            with self._index.updating(reindex_on_error=True):
                written = execution.run(
                    _insert,
                    partitions,
                    [fs_utils.join_path(*keys) for keys, _ in partitions],
                    client=client,
                    npartitions=npartitions,
                    stop_on_error=stop_on_error,
                    kwargs={
                        'axis': self.axis,
                        'zds':
                        client.scatter(ds, hash=False) if client else ds,
                        'fs': self.fs,
                        'merge_callable': merge_callable,
                        'partitioning_properties': self.partition_properties,
                        'delayed': delayed,
                        'distributed': distributed,
                        'merge_kwargs': kwargs,
                    })
                self._index.add(written)
        return written

    def _set_ds_for_insertion(self, ds: dataset.Dataset) -> dataset.Dataset:
        """Prepare a dataset for insertion into the collection.

        Add missing partitioned variables and dimensions where applicable,
        then check their dimensions and properties against the schema.

        Args:
            ds: The dataset to insert.

        Returns:
            The prepared dataset.
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
        """Prepare a variable for insertion into the collection.

        Check its data type, dimensions, and declared fill value against
        the collection schema. Use the schema's compressor, filters, and
        fill value, and remove the variable's attributes before insertion.

        Args:
            variable: The variable to insert.

        Returns:
            The prepared variable.

        Raises:
            ValueError: If the variable is unknown or its data type,
                dimensions, or fill value is incompatible with the schema.
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

    def drop_partitions(
        self,
        *,
        filters: PartitionFilter = None,
        timedelta: datetime.timedelta | None = None,
        npartitions: int | None = None,
        distributed: bool = True,
        stop_on_error: bool = True,
    ) -> tuple[str, ...]:
        """Remove the partitions selected by ``filters`` and ``timedelta``.

        When both are provided, a partition must match the filter and be
        older than ``timedelta``. Without either, all partitions are removed.

        Args:
            filters: Predicate selecting partitions by their keys. See
                :meth:`partitions` for supported predicates.
            timedelta: Select partitions older than the specified duration,
                relative to the current time. Requires a file system
                implementing ``fs.created()``.
            npartitions: Maximum number of partition tasks in progress.
                Defaults to the number of Dask workers. With
                ``distributed=False``, partitions run sequentially.
            distributed: Whether to use Dask. Defaults to ``True``.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish.
                Defaults to ``True``.

        Returns:
            Paths of the removed partitions, relative to the collection.
            Entries absent from storage are excluded from the result.

        Raises:
            ValueError: If ``npartitions`` is below one and at least one
                partition is selected.
            PartitionErrors: If a partition fails. A subsequent listing
                reports the partitions still present in storage.

        Example:
            >>> collection.drop_partitions(filters="year == 2019")
            >>> collection.drop_partitions(
            ...     timedelta=datetime.timedelta(days=30))
        """
        now = datetime.datetime.now(datetime.UTC)

        def is_created_before(path: str) -> bool:
            """Return whether the partition was created longer than
            `timedelta` ago."""
            # A time without a time zone is taken as local time.
            created = self.fs.created(path).astimezone(datetime.UTC)
            return now - created > cast('datetime.timedelta', timedelta)

        with self.synchronizer:
            paths = list(self.partitions(filters=filters))
            if timedelta is not None:
                paths = [path for path in paths if is_created_before(path)]
            return self._drop_partitions(paths,
                                         npartitions=npartitions,
                                         distributed=distributed,
                                         stop_on_error=stop_on_error)

    def _drop_partitions(self, paths: Sequence[str], *,
                         npartitions: int | None, distributed: bool,
                         stop_on_error: bool) -> tuple[str, ...]:
        """Remove partitions while the caller holds the synchronizer lock.

        Args:
            paths: Absolute paths of the partitions to remove.
            npartitions: Maximum number of partition tasks in progress.
            distributed: Whether to use Dask.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish.

        Returns:
            Paths of the removed partitions, relative to the collection.
        """
        if not paths:
            return ()
        client = dask_utils.get_client() if distributed else None
        with self._index.updating(reindex_on_error=True):
            try:
                dropped = execution.run(
                    _drop_partition,
                    paths, [self._relative_path(path) for path in paths],
                    client=client,
                    npartitions=npartitions,
                    stop_on_error=stop_on_error,
                    kwargs={'fs': self.fs})
            finally:
                # The tasks may run in other processes, so the file system
                # cache of this process can still list the dropped
                # partitions.
                for path in paths:
                    self.fs.invalidate_cache(path)
            # Remove every selected partition from the index, including
            # those already absent from storage and therefore skipped.
            self._index.remove(paths)
        return dropped

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

    def _prepare_update(  # noqa: PLR0913
        self,
        func: UpdateCallable,
        partitions: tuple[str, ...],
        *,
        variables: Sequence[str] | None,
        selected_variables: Iterable[str] | None,
        delayed: bool,
        depth: int,
        pool: tuple[str, ...],
        trim: bool,
        client: dask.distributed.Client | None,
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
    ) -> WrappedPartitionCallable | None:
        """Validate output names and build the function for partition
        updates."""
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
            partitions=pool,
            selected_variables=selected_variables) if depth else None
        return wrap_update(func,
                           load,
                           load_overlap=load_overlap,
                           partitioning=self.partitioning,
                           dim=dim,
                           fs=self.fs,
                           metadata=self.metadata,
                           trim=trim)

    def update(  # noqa: PLR0913
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
        stop_on_error: bool = True,
        **kwargs,
    ) -> tuple[str, ...]:
        """Compute and write variables in selected partitions.

        Args:
            func: Function applied to each selected partition.
            *args: Positional arguments passed to the function.
            depth: Number of neighboring partitions to load on each side.
                Defaults to 0. Neighbors are loaded even when excluded by
                ``filters`` or ``selected_partitions``; only selected
                partitions are written.
            variables: Names of variables to update. If ``None``, call
                ``func`` once on the first available selected partition
                to infer the names, then discard that call's results.
                This call uses the requested loading and execution modes,
                but no overlap. Defaults to ``None``.
            selected_variables: Variables to load. Defaults to all.
            selected_partitions: Partition paths relative to the collection.
            filters: Expression selecting partitions to update.
            trim: Whether to retain only the processed partition's values
                from an overlap result. Set to ``False`` if ``func`` already
                returns only those values.
            npartitions: Maximum number of partition tasks in progress.
                Defaults to the number of Dask workers. With
                ``distributed=False``, partitions run sequentially.
            delayed: Whether to load variables as Dask arrays (``True``)
                or NumPy arrays (``False``). If ``None``, follow
                ``distributed``. Defaults to ``None``.
            distributed: Whether to use Dask. Defaults to ``True``.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish.
                Defaults to ``True``.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Paths of the written partitions, relative to the collection.

        Raises:
            ValueError: If an output variable is absent or immutable,
                or if ``npartitions`` is below one.
            PartitionErrors: If a partition fails.

        Notes:
            Each task loads a partition or overlap window before writing
            the results. With overlap, neighboring partitions may already
            contain values written by another task. Functions that read
            and rewrite the same variables must be idempotent and account
            for these changes. Distributed tasks can run in any order.

            A callback declaring ``partition_info`` receives a
            :class:`zcollection.PartitionInfo` on every call, including
            calls without overlap and the output-inference call.
            See :doc:`/updating` for examples of overlap, trimming, and
            passing additional arguments.

        Example:
            For an existing writable collection with input ``var1`` and
            output ``var2``:

            .. code-block:: python

                def ones(ds):
                    return {'var2': ds['var1'].values * 0 + 1}

                collection.update(ones, variables=('var2', ),
                                  selected_variables=('var1', ))
        """
        return self._update(func,
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
                            stop_on_error=stop_on_error,
                            **kwargs)

    def update_immutable(self, name: str, data: ArrayLike) -> None:
        """Update an immutable variable with the supplied data.

        Args:
            name: Name of the variable to update.
            data: New values for the variable.

        Raises:
            ValueError: If the variable is unknown or the data dimensions
                do not match the collection.
        """
        dimensions, _ = self.dimensions_properties()
        variable = self.metadata.variables.get(name, None)

        if variable is None:
            raise ValueError(f'Immutable variable ({name}) does not exist.')

        # Validate the dimension sizes.
        for i, dim in enumerate(variable.dimensions):
            if dimensions[dim] != data.shape[i]:
                raise ValueError(
                    f"Added variable '{variable.name}' contains a "
                    f"dimension '{dim}' with an invalid size "
                    f'({data.shape[i]} instead of {dimensions[dim]})')

        storage.update_zarr_array(
            fs_utils.join_path(self._immutable, variable.name), data, self.fs)

    def add_dimension(self, dimension: meta.Dimension) -> None:
        """Add a dimension to the collection schema.

        Args:
            dimension: The dimension to add.

        Raises:
            TypeError: If ``dimension`` is not a ``Dimension`` instance.
            ValueError: If the dimension is already declared.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="w")
            >>> new_dimension = zcollection.Dimension(
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
        *,
        npartitions: int | None = None,
        stop_on_error: bool = True,
    ) -> None:
        """Add a variable to the collection.

        The new arrays are initialized with the variable's fill value or, if
        none is defined, the default value for its data type. Use
        :meth:`update` to populate the variable.

        If a partition fails, ZCollection attempts to remove the added
        arrays and revert the schema change. If cleanup also fails, the
        variable remains declared. The exception identifies the partitions
        where its arrays may remain. Call :meth:`drop_variable` before
        reading the collection.

        Args:
            variable: The variable to add.
            distributed: Whether to use Dask. Defaults to ``True``.
            npartitions: The maximum number of partition tasks in progress.
                Defaults to the number of Dask workers. Ignored for an
                immutable variable.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish. Defaults
                to ``True``.

        Raises:
            ValueError: If the name is already used, the variable refers to
                an undeclared dimension, or ``npartitions`` is below one when
                there are partitions to process.
            PartitionErrors: If a partition fails. ``completed`` identifies
                partitions that received the array before cleanup.

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
        _LOGGER.info('Adding the variable %r to the collection', variable.name)
        # Variable attributes stay in the collection schema, not in the
        # partition arrays. They are applied when variables are loaded
        # through load().
        stored = variable.set_for_insertion()
        dimensions, chunks = self.dimensions_properties()

        with self.synchronizer:
            if self.dimension in variable.dimensions:
                paths = list(self.partitions())
            else:
                # An immutable variable has one array, created in the calling
                # process.
                paths = [self._immutable]
                distributed, npartitions = False, None
            client = self._client_for(paths,
                                      distributed=distributed,
                                      npartitions=npartitions)
            self._declare_variable(variable)
            if not paths:
                return
            try:
                execution.run(_add_array,
                              paths,
                              [self._relative_path(path) for path in paths],
                              client=client,
                              npartitions=npartitions,
                              stop_on_error=stop_on_error,
                              kwargs={
                                  'variable': stored,
                                  'dimensions': dimensions,
                                  'fs': self.fs,
                                  'axis': self.axis,
                                  'chunks': chunks,
                              })
            except Exception as error:
                self._undo_addition(variable.name,
                                    paths,
                                    client=client,
                                    npartitions=npartitions,
                                    error=error)
                raise

    def drop_variable(
        self,
        variable: str,
        distributed: bool = True,
        *,
        npartitions: int | None = None,
        stop_on_error: bool = True,
    ) -> None:
        """Remove a variable from the collection.

        If a partition fails, the variable remains declared. Call this
        method again to remove any remaining arrays and finish the removal.

        Args:
            variable: Name of the variable to remove.
            distributed: Whether to use Dask. Defaults to ``True``.
            npartitions: Maximum number of partition tasks in progress.
                Defaults to the number of Dask workers. Ignored for
                immutable variables.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish.
                Defaults to ``True``.

        Raises:
            ValueError: If the variable is absent, used for partitioning,
                or ``npartitions`` is below one when partitions are selected.
            PartitionErrors: If a partition fails.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="w")
            >>> collection.drop_variable("my_variable")
        """
        _LOGGER.info('Removing the variable %r from the collection', variable)
        if variable in self.partitioning.variables:
            raise ValueError(
                f'The variable {variable!r} is part of the partitioning.')
        if variable not in self.metadata.variables:
            raise ValueError(
                f'The variable {variable!r} does not exist in the collection.')

        with self.synchronizer:
            if self.dimension in self.metadata.variables[variable].dimensions:
                paths = list(self.partitions())
            else:
                # An immutable variable has one array, removed in the calling
                # process.
                paths = [self._immutable]
                distributed, npartitions = False, None
            client = self._client_for(paths,
                                      distributed=distributed,
                                      npartitions=npartitions)
            if paths:
                self._drop_arrays(variable,
                                  paths,
                                  client=client,
                                  npartitions=npartitions,
                                  stop_on_error=stop_on_error)
            self._undeclare_variable(variable)

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
        stop_on_error: bool = True,
    ) -> Collection:
        """Copy the collection to a new location.

        The copy includes immutable variables and files at the collection
        root. It writes the configuration and an empty partition index
        before copying selected partitions. If interrupted afterwards,
        the new collection can be opened with the partitions present
        in its storage.

        Args:
            target: Destination, which must be absent or empty.
            filters: Predicate selecting partitions to copy.
            filesystem: Destination file system. Defaults to the file
                system of the source collection.
            mode: Access mode of the new collection. Defaults to ``'w'``.
            npartitions: Maximum number of partition tasks in progress.
                Defaults to the number of Dask workers. With
                ``distributed=False``, partitions run sequentially.
            synchronizer: Synchronizer assigned to the new collection.
                Its lock is held throughout the copy. Defaults to ``None``.
            distributed: Whether to use Dask. Defaults to ``True``.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish.
                Defaults to ``True``.

        Returns:
            The new collection.

        Raises:
            ValueError: If the target is not empty, or if ``npartitions``
                is below one when a partition is selected.
            PartitionErrors: If a partition fails. A subsequent listing
                reports the partitions present in the target storage.

        Example:
            >>> import zcollection
            >>> collection = zcollection.open_collection(
            ...     "my_collection", mode="r")
            >>> collection.copy(target="my_new_collection")
        """
        _LOGGER.info('Copying of the collection to %r', target)
        filesystem = filesystem or self.fs
        target = fs_utils.normalize_path(filesystem, target)
        paths = tuple(self.partitions(filters=filters))
        client = self._client_for(paths,
                                  distributed=distributed,
                                  npartitions=npartitions)
        with synchronizer or sync.NoSync():
            if filesystem.exists(target) and filesystem.ls(target):
                raise ValueError(f'The target {target!r} is not empty')
            filesystem.makedirs(target, exist_ok=True)
            if self.fs.exists(self._immutable):
                fs_utils.copy_tree(self._immutable,
                                   fs_utils.join_path(target, IMMUTABLE),
                                   self.fs, filesystem)
            # Copy root-level files except the configuration and index.
            # Both are written separately below.
            root = self.partition_properties.dir
            fs_utils.copy_files([
                item['name'] for item in self.fs.ls(root, detail=True)
                if item['type'] == 'file' and os.path.basename(item['name'])
                not in (INDEX_FILENAME, self.CONFIG)
            ], target, self.fs, filesystem)
            fs_utils.replace_file(filesystem, self._config(target),
                                  self.fs.cat_file(self._config(root)))

            index = PartitionIndex(filesystem,
                                   target,
                                   self.partitioning,
                                   writable=True)
            index.write(())
            if paths:
                names = [self._relative_path(path) for path in paths]
                pairs = [(path, fs_utils.join_path(target, name))
                         for path, name in zip(paths, names, strict=True)]
                with index.updating(reindex_on_error=True):
                    copied = execution.run(_copy_partition,
                                           pairs,
                                           names,
                                           client=client,
                                           npartitions=npartitions,
                                           stop_on_error=stop_on_error,
                                           kwargs={
                                               'fs_source': self.fs,
                                               'fs_target': filesystem
                                           })
                    index.add(copied)

        return Collection.from_config(target,
                                      mode=mode,
                                      filesystem=filesystem,
                                      synchronizer=synchronizer)

    def validate_partitions(
        self,
        filters: PartitionFilter = None,
        distributed: bool = True,
        fix: bool = False,
        *,
        npartitions: int | None = None,
        stop_on_error: bool = True,
    ) -> tuple[str, ...]:
        """Find invalid partitions in the collection.

        A partition is invalid if it cannot be read or lacks an array for
        a declared variable spanning the partitioned dimension.

        A failed :meth:`add_variable` may leave some partitions without
        an array declared in the schema. Call :meth:`drop_variable`
        before using ``fix=True``; otherwise, validation deletes those
        partitions.

        Args:
            filters: Predicate selecting partitions to check. Defaults
                to all partitions.
            distributed: Whether to use Dask. Defaults to ``True``.
            fix: Whether to delete invalid partitions and remove their
                index entries.
            npartitions: Maximum number of partition tasks in progress.
                Defaults to the number of Dask workers.
            stop_on_error: Whether to stop submitting partitions after the
                first failure. Already submitted Dask tasks finish.
                Defaults to ``True``.

        Returns:
            Paths of the invalid partitions, relative to the collection.

        Raises:
            ValueError: If ``npartitions`` is below one when partitions
                are selected for validation.
            PartitionErrors: If a partition cannot be checked or deleted.
                After a failed check, nothing is deleted and ``completed``
                identifies invalid partitions found so far.
        """
        with self.synchronizer if fix else contextlib.nullcontext():
            paths = {
                self._relative_path(path): path
                for path in self.partitions(filters=filters)
            }
            if not paths:
                return ()
            client = self._client_for(list(paths),
                                      distributed=distributed,
                                      npartitions=npartitions)
            variables = tuple(
                name for name, item in self.metadata.variables.items()
                if self.dimension in item.dimensions)
            invalid = execution.run(_check_partition,
                                    list(paths.values()),
                                    list(paths),
                                    client=client,
                                    npartitions=npartitions,
                                    stop_on_error=stop_on_error,
                                    kwargs={
                                        'fs': self.fs,
                                        'variables': variables
                                    })
            for name in invalid:
                warnings.warn(f'Invalid partition: {name}',
                              category=RuntimeWarning,
                              stacklevel=2)
            if fix and invalid:
                self._drop_partitions([paths[name] for name in invalid],
                                      npartitions=npartitions,
                                      distributed=distributed,
                                      stop_on_error=stop_on_error)
        return invalid
