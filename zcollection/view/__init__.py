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

import dask.array.core
import dask.bag.core
import dask.distributed
import dask.utils
import fsspec

from .. import collection, dask_utils, dataset, fs_utils, meta, storage, sync
from ..collection.detail import (
    execute_update,
    infer_variables,
    warn_empty_update,
    wrap_update,
)
from ..convenience import collection as convenience
from ..partitions import PartitionIndex
from .detail import (
    ViewReference,
    ViewUpdateCallable,
    _assert_have_variables,
    _assert_variable_handled,
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
    from collections.abc import Iterable, Iterator, Sequence

    from ..collection.callable_objects import MapCallable, PartitionCallable

__all__ = ('View', 'ViewReference')

#: Module logger.
_LOGGER = logging.getLogger(__name__)


class View:
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
                 _initialize: bool = True) -> None:
        #: The file system used to access the view (default local file system).
        self.fs: fsspec.AbstractFileSystem = fs_utils.get_fs(filesystem)
        #: Path to the directory where the view is stored.
        self.base_dir: str = fs_utils.normalize_path(fs=self.fs, path=base_dir)
        #: The reference collection of the view.
        self.view_ref: collection.Collection = convenience.open_collection(
            view_ref.path, mode='r', filesystem=view_ref.filesystem)
        #: The metadata of the variables handled by the view.
        self.metadata = ds or meta.DatasetSchema([], [], attrs=[])
        #: The synchronizer used to synchronize the view.
        self.synchronizer: sync.Sync = synchronizer or sync.NoSync()
        #: The filters used to select the partitions of the reference.
        self.filters = filters
        #: The partitions of the reference the view holds.
        self._index = PartitionIndex(self.fs,
                                     self.base_dir,
                                     self.view_ref.partitioning,
                                     writable=True)

        if not self.fs.exists(self.base_dir):
            _LOGGER.info('Creating view %s', self)
            self.fs.makedirs(self.base_dir)
            self._write_config()
            if _initialize:
                self._init_partitions(filters, distributed)
        else:
            _LOGGER.info('Opening view %s', self)

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

        if version == meta.DEPRECATED_FORMAT_VERSION:
            raise ValueError(
                'This view was written by an earlier release. Use the '
                "'zcollection.migration.migrate_view' function to copy it into a view "
                'this release writes.')

        view_ref: dict[str, Any] = data['view_ref']
        return View(data['base_dir'],
                    ViewReference(
                        view_ref['path'],
                        fsspec.AbstractFileSystem.from_json(
                            json.dumps(view_ref['fs']))),
                    ds=meta.DatasetSchema.from_config(data['metadata']),
                    filesystem=filesystem,
                    filters=_deserialize_filters(data['filters']),
                    synchronizer=synchronizer)

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
            selected_partitions: A list of partitions to load (using the
                partition relative path).

        Returns:
            The list of partitions.
        """
        held: frozenset[str] = self._held_partitions()
        return (fs_utils.join_path(self.base_dir, item)
                for item in self.view_ref.partitions(
                    filters=filters,
                    indexer=indexer,
                    selected_partitions=selected_partitions,
                    relative=True) if item in held)

    def variables(
        self,
        selected_variables: Iterable[str] | None = None
    ) -> tuple[dataset.Variable, ...]:
        """Return the variables of the view.

        Args:
            selected_variables: The variables to return. If None, all the
                variables are returned.

        Returns:
            The variables of the view.
        """
        return dataset.get_dataset_variable_properties(self.metadata,
                                                       selected_variables)

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
                       fs=self.fs),
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
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded.
            selected_partitions: A list of partitions to load (using the
                partition relative path).
            distributed: Whether to use dask or not. Default To True.

        Returns:
            The dataset.

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
        return tuple(
            fs_utils.join_path(self.base_dir, item) for item in
            self.view_ref.partitions(filters=filters,
                                     selected_partitions=selected_partitions,
                                     relative=True,
                                     lock=True) if item in held)

    def _prepare_update(
        self,
        func: collection.UpdateCallable,
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
            infer = functools.partial(infer_variables, partitions, load, func,
                                      dim, func_args, func_kwargs,
                                      self.synchronizer)
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
            partitions=partitions,
            selected_variables=selected_variables,
            view_ref=self.view_ref) if depth else None
        return wrap_update(func,
                           load,
                           load_overlap=load_overlap,
                           dim=dim,
                           fs=self.fs,
                           metadata=self.metadata,
                           trim=trim)

    def update(
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
        **kwargs,
    ) -> None:
        """Update a variable stored in the view.

        Args:
            func: The function to apply to calculate the new values for the
                target variables.
            args: The positional arguments to pass to the function.
            depth: The depth of the overlap between the partitions. Default is
                0 (no overlap). If depth is greater than 0, the function is
                applied on the partition and its neighbors selected by the
                depth. If ``func`` accepts ``partition_info`` as a keyword
                argument, it receives a tuple containing the partitioned
                dimension name and a slice selecting the current partition
                within the dataset, excluding its neighbors.
            variables: The list of variables updated by the function. If None,
                the variables are inferred by calling the function on the first
                available selected partition, using the requested loading and
                execution modes. This additional call uses that partition alone,
                including when ``depth`` is greater than 0. Default is None.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded.
            selected_partitions: A list of partitions to load (using the
                partition relative path).
            filters: The expression used to filter the partitions to update.
            trim: Whether to keep only the current partition's values from
                the results returned by ``func`` when using overlap. Set it
                to ``False`` if your function already returns only those
                values.
            npartitions: The number of partitions to update in parallel. By
                default, it is equal to the number of Dask workers available
                when calling this method.
            delayed: If True, load variables as Dask arrays. If False, load
                variables directly into NumPy arrays. If None, use the value of
                ``distributed``. Default is None.
            distributed: Whether to use dask or not. Default To True.
            kwargs: The keyword arguments to pass to the function.

        Raises:
            ValueError: If the variable does not exist or if the variable
                belongs to the reference collection.

        Notes:
            Each task loads and writes one partition or overlap window at a
            time. With overlap, neighboring partitions may already contain
            values written by another call. Callbacks that read and rewrite
            the same variables must be idempotent and account for these updates.
            Distributed tasks can run in any order. Inference adds a callback
            invocation whose returned values are discarded.

        Example:
            >>> def temp_celsius_to_kelvin(
            ...     dataset: zcollection.dataset.Dataset,
            ... ) -> Dict[str, numpy.ndarray]:
            ...     return dict(
            ...         temperature_kelvin=dataset["temperature"].values +
            ...         273.15)
            >>> view.update(temp_celsius_to_kelvin)
        """
        _assert_have_variables(self.metadata)

        if not callable(func):
            raise TypeError('func must be a callable')
        if depth < 0:
            raise ValueError('Depth must be non-negative.')
        if delayed is None:
            delayed = distributed
        if selected_variables is not None:
            selected_variables = tuple(selected_variables)

        partitions = self._select_update_partitions(filters,
                                                    selected_partitions)
        if not partitions:
            warn_empty_update()
            return

        client = dask_utils.get_client() if distributed else None
        wrap_function = self._prepare_update(
            func,
            partitions,
            variables=variables,
            selected_variables=selected_variables,
            delayed=delayed,
            depth=depth,
            trim=trim,
            client=client,
            func_args=args,
            func_kwargs=kwargs)
        if wrap_function is None:
            warn_empty_update()
            return

        execute_update(wrap_function,
                       partitions,
                       client=client,
                       synchronizer=self.synchronizer,
                       name=func.__name__,
                       npartitions=npartitions,
                       func_args=args,
                       func_kwargs=kwargs)
        tuple(map(self.fs.invalidate_cache, partitions))

    # false positive, no code duplication
    def map(
        self,
        func: MapCallable,
        /,
        *args,
        delayed: bool = True,
        filters: collection.PartitionFilter = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Iterable[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the view.

        Args:
            func: The function to apply to every partition of the view.
            *args: The positional arguments to pass to the function.
            delayed: Whether to load data in a dask array or in memory.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`zcollection.Collection.partitions`
                method.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag holding, for each partition, the tuple of its scheme and
            the result of the function. A partition the store no longer
            holds is left out.

        Example:
            For a view defining ``var3`` with the same dimensions as ``var1``
            in its reference collection:

            .. code-block:: python

                bag = view.map(
                    lambda ds: ds['var1'].values + ds['var3'].values)
                for partition, values in bag.compute(scheduler='synchronous'):
                    print(partition, values.tolist())
        """

        def _wrap(
            _partition: str,
            _func: PartitionCallable,
            *_args,
            **_kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any] | None:
            """Wraps the function to apply on the partition.

            Args:
                _partition: The partition to apply the function on.
                _func: The function to apply.
                *_args: The positional arguments to pass to the function.
                **_kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            held = _held_one_dataset(
                (self.view_ref.partitioning.parse(_partition), []),
                base_dir=self.base_dir,
                delayed=delayed,
                fs=self.fs,
                selected_variables=self.view_ref.metadata.select_variables(
                    selected_variables),
                view_ref=self.view_ref,
                variables=self.metadata.select_variables(selected_variables),
                with_immutable=True)
            if held is None:
                return None
            zds, partition = held
            return self.view_ref.partitioning.parse(partition), _func(
                zds, *_args, **_kwargs)

        _assert_have_variables(self.metadata)

        bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
            tuple(self.partitions(filters=filters)), partition_size,
            npartitions)
        return bag.map(_wrap, func, *args,
                       **kwargs).filter(lambda item: item is not None)

    def map_overlap(
        self,
        func: MapCallable,
        /,
        *args,
        depth: int = 1,
        delayed: bool = True,
        filters: collection.PartitionFilter = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Iterable[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the view with some overlap.

        Args:
            func: The function to apply to every partition of the view.
            depth: The depth of the overlap between the partitions. Defaults
                to 1. If depth is greater than 0, the function is
                applied on the partition and its neighbors selected by the
                depth. If ``func`` accepts ``partition_info`` as a keyword
                argument, it receives a tuple containing the partitioned
                dimension name and a slice selecting the current partition
                within the dataset, excluding its neighbors.
            *args: The positional arguments to pass to the function.
            delayed: Whether to load data in a dask array or in memory.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`zcollection.Collection.partitions`
                method.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag holding, for each partition, the tuple of its scheme and
            the result of the function. A partition the store no longer
            holds is left out.

        Example:
            For a view defining ``var3`` with the same dimensions as ``var1``
            in its reference collection:

            Each printed array includes results for the partition and its
            available neighbours.

            .. code-block:: python

                bag = view.map_overlap(
                    lambda ds: ds['var1'].values + ds['var3'].values,
                    depth=1)
                for partition, values in bag.compute(scheduler='synchronous'):
                    print(partition, values.tolist())
        """
        if depth < 0:
            raise ValueError('Depth must be greater than or equal to 0')

        add_partition_info: bool = dask.utils.has_keyword(
            func, 'partition_info')

        def _wrap(
            _partition: str,
            _func: PartitionCallable,
            _partitions: tuple[str, ...],
            _depth: int,
            *_args,
            **_kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any] | None:
            """Wraps the function to apply on the partition.

            Args:
                arguments: The partition scheme and the dataset.
                _func: The function to apply.
                datasets: The datasets to apply the function on.
                _depth: The depth of the overlap between the partitions.
                *_args: The positional arguments to pass to the function.
                **_kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            # _load_dataset_with_overlap warns about an absent partition
            # before raising AbsentPartition.
            try:
                zds, indices = _load_dataset_with_overlap(
                    base_dir=self.base_dir,
                    delayed=delayed,
                    depth=_depth,
                    fs=self.fs,
                    metadata=self.metadata,
                    partition=_partition,
                    partitions=_partitions,
                    selected_variables=selected_variables,
                    view_ref=self.view_ref)
            except storage.AbsentPartition:
                return None

            if add_partition_info:
                _kwargs = _kwargs.copy()
                _kwargs['partition_info'] = (self.view_ref.dimension, indices)

            # Finally, apply the function.
            return (self.view_ref.partitioning.parse(_partition),
                    _func(zds, *_args, **_kwargs))

        _assert_have_variables(self.metadata)

        partitions = tuple(self.partitions(filters=filters))
        bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
            partitions, partition_size, npartitions)
        return bag.map(_wrap, func, partitions, depth, *args,
                       **kwargs).filter(lambda item: item is not None)

    def reindex(self) -> tuple[str, ...]:
        """Build the index of the partitions from the store.

        A partition the index does not name becomes visible to a listing,
        and an entry naming a partition the store does not hold is
        dropped.

        Returns:
            The path of each partition, relative to the view.
        """
        _LOGGER.info('Indexing the partitions of %s', self)
        with self.synchronizer:
            keys = self._index.walk()
            self._index.write(keys)
        return tuple(self._index.path_of(key) for key in keys)

    def is_synced(self, distributed: bool = True) -> bool:
        """Check if the view is synchronized with the underlying collection.

        Args:
            distributed: Whether to use dask or not. Default To True.

        Returns:
            True if the view is synchronized, False otherwise.
        """
        partitions = tuple(self.view_ref.partitions(relative=True))

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
