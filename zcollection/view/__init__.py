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
import copy
import functools
import json
import logging
import pathlib
import warnings

import dask.distributed
import fsspec

from .. import (
    collection,
    dask_utils,
    dataset,
    expression,
    fs_utils,
    meta,
    partitioning,
    storage,
    sync,
)
from ..collection.abc import WritableCollection, warn_missing_partitions
from ..collection.detail import infer_variables, wrap_update
from ..convenience import collection as convenience
from ..partitions import PartitionIndex
from .detail import (
    ViewReference,
    ViewUpdateCallable,
    _assert_have_variables,
    _assert_variable_handled,
    _check_partition,
    _create_zarr_array,
    _deserialize_filters,
    _drop_zarr_zarr,
    _held_one_dataset,
    _load_dataset,
    _load_dataset_with_overlap,
    _serialize_filters,
    _sync,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

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

    def __init__(self,
                 base_dir: str,
                 view_ref: ViewReference,
                 *,
                 ds: meta.DatasetSchema | None,
                 filesystem: fsspec.AbstractFileSystem | str | None = None,
                 filters: collection.PartitionFilter = None,
                 synchronizer: sync.Sync | None = None,
                 distributed: bool = True,
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
                self._init_partitions(filters, distributed)
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

    def _init_partitions(self,
                         filters: collection.PartitionFilter,
                         distributed: bool = True) -> None:
        """Initialize the partitions of the view."""
        _LOGGER.info('Populating view %s', self)
        added = tuple(self.view_ref.partitions(filters=filters, relative=True))
        _LOGGER.info('%d partitions selected from %s', len(added),
                     self.view_ref)

        with self.synchronizer, self._index.updating():
            if distributed:
                client: dask.distributed.Client = dask_utils.get_client()
                storage.execute_transaction(
                    client, sync.NoSync(),
                    client.map(_sync,
                               added,
                               base_dir=self.base_dir,
                               view_ref=self.view_ref,
                               fs=self.fs,
                               metadata=self.metadata))
            else:
                for partition in added:
                    _sync(partition,
                          base_dir=self.base_dir,
                          view_ref=self.view_ref,
                          fs=self.fs,
                          metadata=self.metadata)

            self._index.add(added)

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
                     distributed: bool = True) -> None:
        """Add a variable to the view.

        Args:
            variable: The variable to add
            distributed: Whether to use dask or not. Default To True.

        Raises:
            ValueError: If the variable already exists

        Example:
            >>> view.add_variable(
            ...    zcollection.meta.VariableSchema(
            ...        "temperature",
            ...        "float32", ("time", "lat", "lon"),
            ...        (zcollection.meta.Attribute("units", "degrees Celsius"),
            ...         zcollection.meta.Attribute("long_name", "temperature")),
            ...        fill_value=-9999.0))
        """

        # false positive, no code duplication

        ref_zcol = self.view_ref
        ref_meta = ref_zcol.metadata

        variable = dataset.get_variable_metadata(variable)
        _LOGGER.info('Adding variable %r in the view', variable.name)

        if variable.name in ref_meta.variables:
            raise ValueError(f'Variable {variable.name} already exists')

        if ref_zcol.dimension not in variable.dimensions:
            raise ValueError('Immutable variable cannot be added to views.')

        self.metadata.add_variable(variable, set(ref_meta.dimensions))

        existing_partitions = {
            pathlib.Path(path).relative_to(self.base_dir).as_posix()
            for path in self.partitions()
        }

        if existing_partitions:
            args = filter(lambda item: item[0] in existing_partitions,
                          ref_zcol.iterate_on_records())

            # Attributes are not stored at the variable's level
            variable = variable.set_for_insertion()

            try:
                if distributed:
                    self._add_variable_distributed(variable, args)
                else:
                    self._add_variable(variable, args)

            except Exception:
                self.drop_variable(variable.name, distributed, True)
                raise
        else:
            _LOGGER.info('No partitions found, skipping variable creation')

        self._write_config()

    def _add_variable(self, variable: meta.VariableSchema,
                      partitions: Iterator[tuple[str, str]]):
        """Add the provided variable to the collection."""
        dimensions, chunks = self.view_ref.dimensions_properties()

        for partition in partitions:
            _create_zarr_array(
                partition,
                base_dir=self.base_dir,
                variable=variable,
                dimensions=dimensions,
                chunks=chunks,
                axis=self.view_ref.axis,
                fs=self.fs,
                reference_fs=self.view_ref.fs,
            )

    def _add_variable_distributed(self, variable: meta.VariableSchema,
                                  partitions: Iterator[tuple[str, str]]):
        """Add the provided variable to the collection using dask."""
        dimensions, chunks = self.view_ref.dimensions_properties()
        client: dask.distributed.Client = dask_utils.get_client()

        storage.execute_transaction(
            client,
            self.synchronizer,
            client.map(_create_zarr_array,
                       tuple(partitions),
                       base_dir=self.base_dir,
                       variable=variable,
                       dimensions=dimensions,
                       chunks=chunks,
                       axis=self.view_ref.axis,
                       fs=self.fs,
                       reference_fs=self.view_ref.fs),
        )

    def drop_variable(
        self,
        variable: str,
        distributed: bool = True,
        ignore_errors: bool = False,
    ) -> None:
        """Drop a variable from the view.

        Args:
            variable: The name of the variable to drop.
            distributed: Whether to use dask or not. Default To True.
            ignore_errors: Whether to ignore any errors. Default To False.

        Raise:
            ValueError: If the variable does not exist or if the variable
                belongs to the reference collection.

        Example:
            >>> view.drop_variable("temperature")
        """
        _LOGGER.info('Dropping variable %r', variable)
        _assert_variable_handled(self.view_ref.metadata, self.metadata,
                                 variable)

        del self.metadata.variables[variable]
        self._write_config()

        if distributed:
            client: dask.distributed.Client = dask_utils.get_client()
            storage.execute_transaction(
                client,
                self.synchronizer,
                client.map(
                    _drop_zarr_zarr,
                    tuple(self.partitions()),
                    fs=self.fs,
                    variable=variable,
                    ignore_errors=ignore_errors,
                ),
            )
        else:
            for partition in self.partitions():
                _drop_zarr_zarr(partition, self.fs, variable, ignore_errors)

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
                failure, and the partitions in progress finish. Defaults
                to True.
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
                  filters: collection.PartitionFilter = None) -> bool:
        """Check whether selected reference partitions are synchronized.

        Args:
            distributed: Whether to use dask or not. Default To True.
            filters: The predicate used to select reference partitions, as
                in :meth:`zcollection.Collection.partitions`. If None, all
                reference partitions are checked.

        Returns:
            True if all selected partitions are present in the view and their
            reference axes match, False otherwise.
        """
        partitions = tuple(
            self.view_ref.partitions(filters=filters, relative=True))

        if distributed:
            client: dask.distributed.Client = dask_utils.get_client()
            unsynchronized_partition = storage.execute_transaction(
                client, self.synchronizer,
                client.map(_sync,
                           partitions,
                           base_dir=self.base_dir,
                           fs=self.fs,
                           view_ref=self.view_ref,
                           metadata=self.metadata,
                           dry_run=True))
        else:
            unsynchronized_partition = [
                _sync(partition,
                      base_dir=self.base_dir,
                      fs=self.fs,
                      view_ref=self.view_ref,
                      metadata=self.metadata,
                      dry_run=True) for partition in partitions
            ]

        return len(
            tuple(
                filter(lambda item: item is not None,
                       unsynchronized_partition))) == 0

    def sync(
            self,
            filters: collection.PartitionFilter = None,
            distributed: bool = True
    ) -> collection.abc.PartitionFilterCallback:
        """Synchronize the view with the underlying collection.

        Args:
            filters: The predicate used to select the partitions to
                synchronize. To get more information on the predicate, see the
                documentation of the :meth:`zcollection.Collection.partitions`
                method.
                If None, all partitions of the reference are synchronized.
                Existing partitions are not removed, even if they are not
                selected by the predicate.
            distributed: Whether to use dask or not. Default To True.

        Returns:
            A function that can be used as a predicate to get the partitions
            that have been synchronized using the :meth:`View.partitions`
            method.
        """
        _LOGGER.info('Synchronizing view %s', self)

        if filters is not None:
            self.filters = filters
            self._write_config()

        partitions = tuple(
            self.view_ref.partitions(filters=filters, relative=True))

        _LOGGER.info('%d partitions to synchronize', len(partitions))

        with self.synchronizer, self._index.updating():
            if distributed:
                client: dask.distributed.Client = dask_utils.get_client()
                synchronized_partition: list[
                    str | None] = storage.execute_transaction(
                        client, sync.NoSync(),
                        client.map(_sync,
                                   partitions,
                                   base_dir=self.base_dir,
                                   fs=self.fs,
                                   view_ref=self.view_ref,
                                   metadata=self.metadata)) or []
            else:
                synchronized_partition = [
                    _sync(partition,
                          base_dir=self.base_dir,
                          fs=self.fs,
                          view_ref=self.view_ref,
                          metadata=self.metadata) for partition in partitions
                ]
            self._index.add(partitions)

        synced: tuple[str, ...] = tuple(item for item in synchronized_partition
                                        if item is not None)

        partition_ids = tuple(
            dict(self.view_ref.partitioning.parse(item)) for item in synced)
        return lambda item: item in partition_ids

    def validate_partitions(self,
                            filters: collection.PartitionFilter = None,
                            distributed: bool = True,
                            fix: bool = False) -> list[str]:
        """List the invalid partitions of the view.

        A partition is invalid when it cannot be read, or when the reference
        collection no longer has it.

        Args:
            filters: The predicate selecting the partitions to check. By
                default, every partition is checked.
            distributed: Whether to use dask or not. Default To True.
            fix: Whether to repair the invalid partitions. When the reference
                collection still has the partition, the view writes it again.
                Otherwise the view deletes the partition and removes it from
                the index.

        Returns:
            The path of each invalid partition.
        """
        if isinstance(filters, str):
            expr: Any = expression.Expression(filters)
        elif callable(filters):
            expr = filters
        else:
            expr = None

        sep = self.fs.sep
        partitions = tuple(item for item in self._held_partitions()
                           if self._is_selected(item.split(sep), expr))
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

        check = functools.partial(_check_partition,
                                  base_dir=self.base_dir,
                                  fs=self.fs,
                                  view_ref=self.view_ref,
                                  metadata=self.metadata)
        if distributed:
            client = dask_utils.get_client()
            for item in dask.distributed.as_completed(
                    client.map(check, partitions)):
                _validity_check(*item.result())
        else:
            for item in partitions:
                _validity_check(*check(item))

        if fix and invalid_partitions:
            self._fix_partitions(invalid_partitions)
        return [
            fs_utils.join_path(self.base_dir, item)
            for item in invalid_partitions
        ]

    def _fix_partitions(self, partitions: Sequence[str]) -> None:
        """Write the invalid partitions again, or delete them.

        Args:
            partitions: The path of each invalid partition, relative to the
                view.
        """
        orphans: list[str] = []
        with self.synchronizer, self._index.updating():
            for item in partitions:
                path = fs_utils.join_path(self.base_dir, item)
                if self.fs.exists(path):
                    self.fs.rm(path, recursive=True)
                    self.fs.invalidate_cache(path)
                if self.view_ref.fs.exists(
                        fs_utils.join_path(
                            self.view_ref.partition_properties.dir, item)):
                    _LOGGER.info('Writing partition again: %s', item)
                    _sync(item,
                          base_dir=self.base_dir,
                          fs=self.fs,
                          view_ref=self.view_ref,
                          metadata=self.metadata)
                else:
                    _LOGGER.info('Removing invalid partition: %s', item)
                    orphans.append(item)
            if orphans:
                self._index.remove(orphans)
