# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
View on a reference collection.
===============================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import contextlib
import copy
import functools
import json
import logging
import warnings

import fsspec

from .. import (
    collection,
    dask_utils,
    dataset,
    execution,
    expression,
    fs_utils,
    meta,
    partitioning,
    sync,
)
from ..collection.abc import WritableCollection, warn_missing_partitions
from ..collection.detail import infer_variables, wrap_update
from ..convenience import collection as convenience
from ..partitions import PartitionIndex
from .detail import (
    ViewReference,
    ViewUpdateCallable,
    _add_array,
    _assert_have_variables,
    _assert_variable_handled,
    _check_partition,
    _deserialize_filters,
    _drop_array,
    _held_one_dataset,
    _load_dataset,
    _load_dataset_with_overlap,
    _repair_partition,
    _serialize_filters,
    _sync,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    import dask.distributed

__all__ = ('View', 'ViewReference')

#: Module logger.
_LOGGER = logging.getLogger(__name__)


class View(WritableCollection):
    """View on a reference collection.

    Args:
        base_dir: Path to the directory where the view is stored.
        view_ref: Access properties for the reference collection.
        ds: The dataset schema for this view, or None for an empty schema.
        filesystem: The file system used to access the view.
        filters: The filters used to select the partitions of the reference. If
            not provided, all partitions are selected.
        synchronizer: The synchronizer used to synchronize the view.
        distributed: Whether to use dask or not. Default To True.
        npartitions: The largest number of partitions written at the same
            time when the view is created. Defaults to the number of Dask
            workers.
        stop_on_error: When the view is created: if True, no partition starts
            after the first failure, except the ones already submitted to
            Dask, which finish. Defaults to True.
        _initialize: Whether to populate the partitions of a new view.
        _deprecated: Whether the view was written by an earlier release. Such
            a view can only be read: ``add_variable``, ``drop_variable``,
            ``reindex``, ``sync`` and ``update`` stop with an
            :py:class:`io.UnsupportedOperation`.

    Note:
        Normally, you should not call this constructor directly. Instead, use
        :func:`create_view <zcollection.create_view>` or :func:`open_view
        <zcollection.open_view>` to create or open a view.
    """
    #: Configuration filename of the view.
    CONFIG: ClassVar[str] = '.view'
    _DROP_ARRAY = staticmethod(_drop_array)

    def __init__(self,
                 base_dir: str,
                 view_ref: ViewReference,
                 *,
                 ds: meta.DatasetSchema | None,
                 filesystem: fsspec.AbstractFileSystem | str | None = None,
                 filters: collection.PartitionFilter = None,
                 synchronizer: sync.Sync | None = None,
                 distributed: bool = True,
                 npartitions: int | None = None,
                 stop_on_error: bool = True,
                 _initialize: bool = True,
                 _deprecated: bool = False) -> None:
        self._fs: fsspec.AbstractFileSystem = fs_utils.get_fs(filesystem)
        #: Path to the directory where the view is stored.
        self.base_dir: str = fs_utils.normalize_path(fs=self.fs, path=base_dir)
        #: The reference collection of the view.
        self.view_ref: collection.Collection = convenience.open_collection(
            view_ref.path, mode='r', filesystem=view_ref.filesystem)
        #: The metadata of the variables handled by the view.
        self._metadata = ds or meta.DatasetSchema([], [], attrs=[])
        self._synchronizer: sync.Sync = synchronizer or sync.NoSync()
        #: The filters used to select the partitions of the reference.
        self.filters = filters
        #: The partitions of the reference the view holds.
        self._index = PartitionIndex(self.fs,
                                     self.base_dir,
                                     self.view_ref.partitioning,
                                     writable=not _deprecated)

        if _deprecated:
            self._refuse_writes((
                'add_variable',
                'drop_variable',
                'reindex',
                'sync',
                'update',
            ))

        if not self.fs.exists(self.base_dir):
            _LOGGER.info('Creating view %s', self)
            self.fs.makedirs(self.base_dir)
            self._write_config()
            self._index.reindex()
            if _initialize:
                _LOGGER.info('Populating view %s', self)
                with self.synchronizer:
                    self._sync_partitions(filters,
                                          distributed=distributed,
                                          npartitions=npartitions,
                                          stop_on_error=stop_on_error)
        else:
            _LOGGER.info('Opening view %s', self)

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """The filesystem holding the view."""
        return self._fs

    @property
    def synchronizer(self) -> sync.Sync:
        """The synchronizer used by view operations."""
        return self._synchronizer

    @property
    def metadata(self) -> meta.DatasetSchema:
        """The schema of the variables owned by the view."""
        return self._metadata

    @property
    def dimension(self) -> str:
        """The partitioned dimension of the reference collection."""
        return self.view_ref.dimension

    @property
    def partitioning(self) -> partitioning.Partitioning:
        """The partitioning strategy of the reference collection."""
        return self.view_ref.partitioning

    @property
    def _partitions_dir(self) -> str:
        return self.base_dir

    @property
    def _reference(self) -> collection.Collection:
        return self.view_ref

    def _sync_partitions(self, filters: collection.PartitionFilter, *,
                         distributed: bool, npartitions: int | None,
                         stop_on_error: bool) -> tuple[str, ...]:
        """Synchronize the partitions of the reference that `filters`
        selects. The caller holds the lock.

        Args:
            filters: The predicate selecting the partitions of the reference.
            distributed: Whether to use Dask.
            npartitions: The largest number of partitions processed at the
                same time.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish.

        Returns:
            The partitions written, relative to the view.
        """
        partitions = tuple(
            self.view_ref.partitions(filters=filters, relative=True))
        _LOGGER.info('%d partitions selected from %s', len(partitions),
                     self.view_ref)
        if not partitions:
            return ()
        client = self._client_for(partitions,
                                  distributed=distributed,
                                  npartitions=npartitions)
        with self._index.updating(reindex_on_error=True):
            written = execution.run(_sync,
                                    partitions,
                                    partitions,
                                    client=client,
                                    npartitions=npartitions,
                                    stop_on_error=stop_on_error,
                                    kwargs=self._partition_kwargs())
            self._index.add(partitions)
        return written

    def _held_partitions(self) -> frozenset[str]:
        """Return the path of each partition the index names.

        Returns:
            The paths, relative to the view.
        """
        return frozenset(
            self._index.path_of(key) for key in self._index.keys())

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'<filesystem={self.fs.__class__.__name__!r}, '
                f'base_dir={self.base_dir!r}>')

    @classmethod
    def _config(cls, base_dir: str) -> str:
        """Returns the configuration path."""
        return fs_utils.join_path(base_dir, cls.CONFIG)

    def _write_config(self) -> None:
        """Write the configuration file for the view."""
        config = self._config(self.base_dir)
        fs: dict[str, Any] = json.loads(self.view_ref.fs.to_json())

        params = {
            'base_dir': self.base_dir,
            'filters': _serialize_filters(self.filters),
            'metadata': self.metadata.get_config(),
            'view_ref': {
                'path': self.view_ref.partition_properties.dir,
                'fs': fs,
            },
            'version': meta.FORMAT_VERSION,
        }

        fs_utils.replace_file(self.fs, config, json.dumps(params, indent=4))

        self.fs.invalidate_cache(config)

    @classmethod
    def from_config(
        cls,
        path: str,
        *,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        synchronizer: sync.Sync | None = None,
    ) -> View:
        """Open a View described by a configuration file.

        Args:
            path: The path to the configuration file.
            filesystem: The filesystem to use for the view.
            synchronizer: The synchronizer to use for the view.

        Returns:
            The view.

        Raises:
            ValueError: If the provided directory does not contain a view.
        """
        _LOGGER.info('Opening view %r', path)
        fs = fs_utils.get_fs(filesystem)
        config = cls._config(path)

        if not fs.exists(config):
            raise ValueError(f'zarr view not found at path {path!r}')

        with fs.open(config) as stream:
            data: dict[str, Any] = json.load(stream)

        version = meta.format_version(data)

        if version > meta.FORMAT_VERSION:
            raise ValueError(
                f'This view is written in version {version} of the layout. '
                'This version of zcollection reads up to version '
                f'{meta.FORMAT_VERSION}.')

        deprecated = version == meta.DEPRECATED_FORMAT_VERSION
        if deprecated:
            warnings.warn(
                message=('This view was written by an earlier release and '
                         'can only be read. Use the '
                         "'zcollection.migration.migrate_view' function to "
                         'copy it into a view this release writes.'),
                category=UserWarning,
                stacklevel=2)

        view_ref: dict[str, Any] = data['view_ref']
        return View(data['base_dir'],
                    ViewReference(
                        view_ref['path'],
                        fsspec.AbstractFileSystem.from_json(
                            json.dumps(view_ref['fs']))),
                    ds=meta.DatasetSchema.from_config(data['metadata']),
                    filesystem=filesystem,
                    filters=_deserialize_filters(data['filters']),
                    synchronizer=synchronizer,
                    _deprecated=deprecated)

    def partitions(
        self,
        *,
        filters: collection.PartitionFilter = None,
        indexer: collection.Indexer | None = None,
        selected_partitions: Iterable[str] | None = None,
    ) -> Iterator[str]:
        """Returns the list of partitions in the view.

        Args:
            filters: The partition filters.
            indexer: The indexer to apply.
            selected_partitions: The paths of the partitions to select, relative
                to the view.

        Returns:
            The list of partitions.
        """
        held: frozenset[str] = self._held_partitions()
        listed = tuple(
            self.view_ref.partitions(filters=filters,
                                     indexer=indexer,
                                     selected_partitions=selected_partitions,
                                     relative=True))
        if selected_partitions is not None:
            missing = [item for item in listed if item not in held]
            if missing:
                warn_missing_partitions(missing)
        return (fs_utils.join_path(self.base_dir, item) for item in listed
                if item in held)

    def add_variable(self,
                     variable: meta.VariableSchema | dataset.Variable,
                     distributed: bool = True,
                     *,
                     npartitions: int | None = None,
                     stop_on_error: bool = True) -> None:
        """Add a variable to the view.

        Every value of the variable is its fill value, or the default value
        of its data type without one, until it is written, for example by
        :meth:`update`.

        If a partition fails, the variable is removed from the view. If that
        removal fails as well, the variable stays, and a note on the error
        names the partitions where its array may remain: call
        :meth:`drop_variable` before reading the view.

        Args:
            variable: The variable to add.
            distributed: Whether to use Dask. Defaults to True.
            npartitions: The largest number of partitions processed at the
                same time. Defaults to the number of Dask workers.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.

        Raises:
            ValueError: If the view or its reference collection already has
                a variable of that name, if the variable lacks the
                partitioned dimension, or if ``npartitions`` is below one
                when there are partitions to process.
            PartitionErrors: If a partition failed. ``completed`` names the
                partitions that received the array before it was removed.

        Example:
            >>> view.add_variable(
            ...    zcollection.meta.VariableSchema(
            ...        "temperature",
            ...        "float32", ("time", "lat", "lon"),
            ...        (zcollection.meta.Attribute("units", "degrees Celsius"),
            ...         zcollection.meta.Attribute("long_name", "temperature")),
            ...        fill_value=-9999.0))
        """
        ref_zcol = self.view_ref
        ref_meta = ref_zcol.metadata

        variable = dataset.get_variable_metadata(variable)
        _LOGGER.info('Adding the variable %r to the view', variable.name)

        if variable.name in ref_meta.variables:
            raise ValueError(f'Variable {variable.name} already exists')

        if ref_zcol.dimension not in variable.dimensions:
            raise ValueError('Immutable variable cannot be added to views.')

        # Attributes are not stored at the variable's level
        stored = variable.set_for_insertion()
        dimensions, chunks = ref_zcol.dimensions_properties()

        with self.synchronizer:
            held = {self._relative_path(path) for path in self.partitions()}
            records = [
                item for item in ref_zcol.iterate_on_records()
                if item[0] in held
            ]
            paths = [
                fs_utils.join_path(self.base_dir, name) for name, _ in records
            ]
            client = self._client_for(paths,
                                      distributed=distributed,
                                      npartitions=npartitions)
            self._declare_variable(variable, set(ref_meta.dimensions))
            if not paths:
                _LOGGER.info('No partitions found, skipping variable creation')
                return
            try:
                execution.run(_add_array,
                              records, [name for name, _ in records],
                              client=client,
                              npartitions=npartitions,
                              stop_on_error=stop_on_error,
                              kwargs={
                                  'base_dir': self.base_dir,
                                  'variable': stored,
                                  'dimensions': dimensions,
                                  'chunks': chunks,
                                  'axis': ref_zcol.axis,
                                  'fs': self.fs,
                                  'reference_fs': ref_zcol.fs,
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
        """Remove a variable from the view.

        If a partition fails, the variable stays in the view; calling this
        method again finishes the removal.

        Args:
            variable: The name of the variable to remove.
            distributed: Whether to use Dask. Defaults to True.
            npartitions: The largest number of partitions processed at the
                same time. Defaults to the number of Dask workers.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.

        Raises:
            ValueError: If the view has no variable of that name, if the
                variable belongs to the reference collection, or if
                ``npartitions`` is below one when there are partitions to
                process.
            PartitionErrors: If a partition failed.

        Example:
            >>> view.drop_variable("temperature")
        """
        _LOGGER.info('Removing the variable %r from the view', variable)
        _assert_variable_handled(self.view_ref.metadata, self.metadata,
                                 variable)

        with self.synchronizer:
            paths = list(self.partitions())
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

    def load(
        self,
        *,
        delayed: bool | None = None,
        filters: collection.PartitionFilter = None,
        indexer: collection.Indexer | None = None,
        selected_variables: Iterable[str] | None = None,
        selected_partitions: Iterable[str] | None = None,
        distributed: bool = True,
    ) -> dataset.Dataset | None:
        """Load the view.

        Args:
            delayed: Whether to load data in a dask array or not.
                Default value is True if distributed is True, False otherwise.
            filters: The predicate used to filter the partitions to select.
                To get more information on the predicate, see the
                documentation of the :meth:`Collection.partitions
                <zcollection.collection.Collection.partitions>` method.
            indexer: The indexer to apply.
            selected_variables: Input variables to load from the view and its
                reference collection. If None, all variables are loaded.
            selected_partitions: The paths of the partitions to select, relative
                to the view.
            distributed: Whether to use dask or not. Default To True.

        Returns:
            The dataset containing the selected partitions. None when
            ``filters``, ``indexer`` or ``selected_partitions`` select no
            data. Without an explicit partition selection, immutable
            variables from the reference can be loaded alone.

        Example:
            >>> view.load()
            >>> view.load(filters="time == '2020-01-01'")
            >>> view.load(filters=lambda x: x["time"] == "2020-01-01")
        """
        if delayed is None:
            delayed = distributed

        array: dataset.Dataset | None = None
        datasets: list[tuple[dataset.Dataset, str]]
        partitions = self.partitions(filters=filters,
                                     indexer=indexer,
                                     selected_partitions=selected_partitions)

        if indexer is not None:
            arguments = tuple(
                collection.abc.build_indexer_args(self.view_ref,
                                                  filters,
                                                  indexer,
                                                  partitions=partitions))
            if len(arguments) == 0:
                return None
        else:
            arguments = tuple((self.view_ref.partitioning.parse(item), [])
                              for item in partitions)

        if distributed:
            client: dask.distributed.Client = dask_utils.get_client()
            futures: list[dask.distributed.Future] = client.map(
                _held_one_dataset,
                arguments,
                base_dir=self.base_dir,
                delayed=delayed,
                fs=self.fs,
                selected_variables=self.view_ref.metadata.select_variables(
                    selected_variables),
                view_ref=client.scatter(self.view_ref),
                variables=self.metadata.select_variables(selected_variables))
            datasets = [
                item for item in client.gather(futures) if item is not None
            ]
        else:
            datasets = [
                item for item in (
                    _held_one_dataset(arg,
                                      base_dir=self.base_dir,
                                      delayed=delayed,
                                      fs=self.fs,
                                      selected_variables=self.view_ref.metadata
                                      .select_variables(selected_variables),
                                      view_ref=self.view_ref,
                                      variables=self.metadata.select_variables(
                                          selected_variables))
                    for arg in arguments) if item is not None
            ]

        # The load function returns the path of each partition beside its
        # dataset. Only the datasets are kept here.
        arrays: list[dataset.Dataset] = [item[0] for item in datasets]

        if arrays:
            array = arrays.pop(0)
            if arrays:
                array = array.concat(arrays, self.view_ref.dimension)

        # An explicit selection with no partition data returns None.
        # Without a selection, immutable variables can still be loaded alone.
        if array is None and (filters is not None or indexer is not None
                              or selected_partitions is not None):
            return None

        array = self.view_ref.merge_immutable(array, selected_variables,
                                              delayed)

        if array is not None:
            metadata: meta.DatasetSchema = copy.deepcopy(
                self.view_ref.metadata)
            metadata.variables.update(self.metadata.variables.items())
            array.fill_attrs(metadata)

            return array

        return array

    def _select_update_partitions(
        self,
        filters: collection.PartitionFilter,
        selected_partitions: Iterable[str] | None,
    ) -> tuple[str, ...]:
        """Select reference partitions held by the view for an update."""
        with self.synchronizer:
            held = self._held_partitions()
        listed = tuple(
            self.view_ref.partitions(filters=filters,
                                     selected_partitions=selected_partitions,
                                     relative=True,
                                     lock=True))
        if selected_partitions is not None:
            missing = [item for item in listed if item not in held]
            if missing:
                warn_missing_partitions(missing)
        return tuple(
            fs_utils.join_path(self.base_dir, item) for item in listed
            if item in held)

    def _prepare_update(  # noqa: PLR0913
        self,
        func: collection.UpdateCallable,
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
    ) -> ViewUpdateCallable | None:
        """Validate output names and build the function for partition updates."""
        dim = self.view_ref.dimension
        load = functools.partial(_load_dataset,
                                 base_dir=self.base_dir,
                                 delayed=delayed,
                                 fs=self.fs,
                                 view_ref=self.view_ref,
                                 metadata=self.metadata,
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
                                      partitioning=self.view_ref.partitioning)
            variables = client.submit(
                infer, pure=False).result() if client else infer()
        if not variables:
            return None
        for name in variables:
            _assert_variable_handled(self.view_ref.metadata, self.metadata,
                                     name)
        _LOGGER.info('Updating variable %s',
                     ', '.join(repr(item) for item in variables))
        load_overlap = functools.partial(
            _load_dataset_with_overlap,
            base_dir=self.base_dir,
            delayed=delayed,
            depth=depth,
            fs=self.fs,
            metadata=self.metadata,
            partitions=pool,
            selected_variables=selected_variables,
            view_ref=self.view_ref) if depth else None
        return wrap_update(func,
                           load,
                           load_overlap=load_overlap,
                           partitioning=self.view_ref.partitioning,
                           dim=dim,
                           fs=self.fs,
                           metadata=self.metadata,
                           trim=trim)

    def update(  # noqa: PLR0913
        self,
        func: collection.UpdateCallable,
        /,
        *args,
        depth: int = 0,
        filters: collection.PartitionFilter = None,
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
        """Update a variable stored in the view.

        Args:
            func: The function to apply to calculate the new values for the
                target variables.
            args: The positional arguments to pass to the function.
            depth: The depth of the overlap between the partitions. Default is
                0 (no overlap). If depth is greater than 0, the function is
                applied on each partition with ``depth`` neighbors on each
                side, whether ``filters`` and ``selected_partitions`` select
                these neighbors or not. Only the selected partitions are
                written.
            variables: The list of variables updated by the function. If None,
                the variables are inferred by calling the function on the first
                available selected partition, using the requested loading and
                execution modes. This additional call uses that partition alone,
                including when ``depth`` is greater than 0. Default is None.
            selected_variables: Input variables to load from the view and its
                reference collection. If None, all variables are loaded.
            selected_partitions: The paths of the partitions to select, relative
                to the view.
            filters: The expression used to filter the partitions to update.
            trim: Whether to keep only the current partition's values from
                the results returned by ``func`` when using overlap. Set it
                to ``False`` if your function already returns only those
                values.
            npartitions: The largest number of partitions updated at the
                same time. Defaults to the number of Dask workers. With
                ``distributed=False``, the partitions are processed one at a
                time.
            delayed: If True, load variables as Dask arrays. If False, load
                variables directly into NumPy arrays. If None, use the value of
                ``distributed``. Default is None.
            distributed: Whether to use dask or not. Default To True.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.
            kwargs: The keyword arguments to pass to the function.

        Returns:
            The path of each partition written, relative to the view.

        Raises:
            ValueError: If the variable does not exist, if the variable
                belongs to the reference collection, or if ``npartitions``
                is below one.
            PartitionErrors: If a partition failed.

        Notes:
            Each task loads and writes one partition or overlap window at a
            time. With overlap, neighboring partitions may already contain
            values written by another call. Callbacks that read and rewrite
            the same variables must be idempotent and account for these updates.
            Distributed tasks can run in any order. Inference adds a callback
            invocation whose returned values are discarded. A callback
            declaring ``partition_info`` receives a :class:`zcollection.PartitionInfo`
            on every call, including inference and calls without overlap.
            See :doc:`/updating` for examples of overlap, trimming and
            external arguments.

        Example:
            For a view with input ``temperature`` and its own output
            ``temperature_kelvin``:

            .. code-block:: python

                def celsius_to_kelvin(ds):
                    return {'temperature_kelvin': ds['temperature'].values + 273.15}

                view.update(celsius_to_kelvin, variables=('temperature_kelvin', ),
                            selected_variables=('temperature', ))
        """
        _assert_have_variables(self.metadata)

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

    def _prepare_map(
        self,
        partitions: tuple[str, ...],
        *,
        delayed: bool,
        depth: int,
        selected_variables: Iterable[str] | None,
    ) -> tuple[Callable[[str], dataset.Dataset],
               Callable[[str], tuple[dataset.Dataset, slice]] | None]:
        """Prepare the loaders for partition mappings."""
        load = functools.partial(_load_dataset,
                                 base_dir=self.base_dir,
                                 delayed=delayed,
                                 fs=self.fs,
                                 view_ref=self.view_ref,
                                 metadata=self.metadata,
                                 selected_variables=selected_variables)
        load_overlap = functools.partial(
            _load_dataset_with_overlap,
            base_dir=self.base_dir,
            delayed=delayed,
            depth=depth,
            fs=self.fs,
            metadata=self.metadata,
            partitions=partitions,
            selected_variables=selected_variables,
            view_ref=self.view_ref) if depth else None

        return load, load_overlap

    def is_synced(self,
                  distributed: bool = True,
                  *,
                  filters: collection.PartitionFilter = None,
                  npartitions: int | None = None,
                  stop_on_error: bool = True) -> bool:
        """Check whether selected reference partitions are synchronized.

        A partition can fail to be checked, for example because its checksum
        array is missing; :meth:`validate_partitions` with ``fix=True``
        writes such a partition again.

        Args:
            distributed: Whether to use dask or not. Default To True.
            filters: The predicate used to select reference partitions, as
                in :meth:`Collection.partitions
                <zcollection.collection.Collection.partitions>`. If None, all
                reference partitions are checked.
            npartitions: The largest number of partitions checked at the
                same time. Defaults to the number of Dask workers.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.

        Returns:
            True if all selected partitions are present in the view and their
            reference axes match, False otherwise.

        Raises:
            ValueError: If ``npartitions`` is below one when there are
                partitions to check.
            PartitionErrors: If a partition could not be checked.
                ``completed`` names the partitions found out of sync.
        """
        partitions = tuple(
            self.view_ref.partitions(filters=filters, relative=True))
        if not partitions:
            return True
        client = self._client_for(partitions,
                                  distributed=distributed,
                                  npartitions=npartitions)
        return not execution.run(_sync,
                                 partitions,
                                 partitions,
                                 client=client,
                                 npartitions=npartitions,
                                 stop_on_error=stop_on_error,
                                 kwargs=self._partition_kwargs()
                                 | {'dry_run': True})

    def sync(
        self,
        filters: collection.PartitionFilter = None,
        distributed: bool = True,
        *,
        npartitions: int | None = None,
        stop_on_error: bool = True,
    ) -> collection.abc.PartitionFilterCallback:
        """Synchronize the view with the underlying collection.

        Args:
            filters: The predicate used to select the partitions to
                synchronize. To get more information on the predicate, see the
                documentation of the :meth:`Collection.partitions
                <zcollection.collection.Collection.partitions>` method.
                If None, all partitions of the reference are synchronized.
                Existing partitions are not removed, even if they are not
                selected by the predicate. The view records the predicate
                once every selected partition is synchronized.
            distributed: Whether to use dask or not. Default To True.
            npartitions: The largest number of partitions processed at the
                same time. Defaults to the number of Dask workers.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.

        Returns:
            A function that can be used as a predicate to get the partitions
            that have been synchronized using the :meth:`View.partitions`
            method.

        Raises:
            ValueError: If ``npartitions`` is below one when there are
                partitions to process.
            PartitionErrors: If a partition failed.
        """
        _LOGGER.info('Synchronizing view %s', self)

        with self.synchronizer:
            written = self._sync_partitions(filters,
                                            distributed=distributed,
                                            npartitions=npartitions,
                                            stop_on_error=stop_on_error)
            if filters is not None:
                recorded = self.filters
                self.filters = filters
                try:
                    self._write_config()
                except BaseException:
                    self.filters = recorded
                    raise

        partition_ids = tuple(
            dict(self.view_ref.partitioning.parse(item)) for item in written)
        return lambda item: item in partition_ids

    def validate_partitions(
        self,
        filters: collection.PartitionFilter = None,
        distributed: bool = True,
        fix: bool = False,
        *,
        npartitions: int | None = None,
        stop_on_error: bool = True,
    ) -> tuple[str, ...]:
        """List the invalid partitions of the view.

        A partition is invalid when it cannot be read, when it lacks the
        array of a variable the view declares or its ``.checksum`` array, or
        when the reference collection no longer has it. A failed
        :meth:`add_variable` can leave declared a variable that some
        partitions lack. Call :meth:`drop_variable` before validating with
        ``fix``, which otherwise writes those partitions again.

        Args:
            filters: The predicate selecting the partitions to check. By
                default, every partition is checked.
            distributed: Whether to use dask or not. Default To True.
            fix: Whether to repair the invalid partitions. When the reference
                collection still has the partition, the view writes it again,
                and its variables hold their fill value. Otherwise the view
                deletes the partition and removes it from the index.
            npartitions: The largest number of partitions processed at the
                same time. Defaults to the number of Dask workers.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.

        Returns:
            The path of each invalid partition, relative to the view.

        Raises:
            ValueError: If ``npartitions`` is below one when there are
                partitions to check.
            PartitionErrors: If a partition could not be checked or repaired.
                After a failed check, nothing is repaired, and ``completed``
                names the invalid partitions found.
        """
        if isinstance(filters, str):
            expr: Any = expression.Expression(filters)
        elif callable(filters):
            expr = filters
        else:
            expr = None

        with self.synchronizer if fix else contextlib.nullcontext():
            sep = self.fs.sep
            partitions = tuple(item for item in self._held_partitions()
                               if self._is_selected(item.split(sep), expr))
            if not partitions:
                return ()
            client = self._client_for(partitions,
                                      distributed=distributed,
                                      npartitions=npartitions)
            invalid = execution.run(_check_partition,
                                    partitions,
                                    partitions,
                                    client=client,
                                    npartitions=npartitions,
                                    stop_on_error=stop_on_error,
                                    kwargs=self._partition_kwargs())
            for name in invalid:
                warnings.warn(f'Invalid partition: {name}',
                              category=RuntimeWarning,
                              stacklevel=2)
            if fix and invalid:
                self._repair_partitions(invalid,
                                        client=client,
                                        npartitions=npartitions,
                                        stop_on_error=stop_on_error)
        return invalid

    def _partition_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments shared by the tasks that process the
        partitions of the view."""
        return {
            'base_dir': self.base_dir,
            'fs': self.fs,
            'view_ref': self.view_ref,
            'metadata': self.metadata
        }

    def _repair_partitions(self, partitions: Sequence[str], *,
                           client: dask.distributed.Client | None,
                           npartitions: int | None,
                           stop_on_error: bool) -> None:
        """Write the invalid partitions again, or delete those the reference
        collection no longer has. The caller holds the lock.

        Args:
            partitions: The invalid partitions, relative to the view.
            client: The Dask client, or ``None``.
            npartitions: The largest number of partitions processed at the
                same time.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish.
        """
        records = [
            (name,
             self.view_ref.fs.exists(
                 fs_utils.join_path(self.view_ref.partition_properties.dir,
                                    name))) for name in partitions
        ]
        with self._index.updating(reindex_on_error=True):
            execution.run(_repair_partition,
                          records,
                          partitions,
                          client=client,
                          npartitions=npartitions,
                          stop_on_error=stop_on_error,
                          kwargs=self._partition_kwargs())
            self._index.remove(
                [name for name, rewrite in records if not rewrite])
