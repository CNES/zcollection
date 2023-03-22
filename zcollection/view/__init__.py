# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
View on a reference collection.
===============================
"""
from __future__ import annotations

from typing import Any, ClassVar, Iterable, Iterator
import copy
import json
import logging
import pathlib
import warnings

import dask.array.core
import dask.bag.core
import dask.distributed
import dask.utils
import fsspec

from .. import collection, dask_utils, dataset, fs_utils, meta, storage, sync
from ..collection.detail import try_infer_callable
from ..convenience import collection as convenience
from .detail import (
    ViewReference,
    _assert_have_variables,
    _assert_variable_handled,
    _create_zarr_array,
    _deserialize_filters,
    _drop_zarr_zarr,
    _load_datasets_list,
    _load_one_dataset,
    _select_overlap,
    _serialize_filters,
    _sync,
    _wrap_update_func,
    _wrap_update_func_overlap,
    _write_checksum,
)

__all__ = ['View', 'ViewReference']

#: Module logger.
_LOGGER = logging.getLogger(__name__)


class View:
    """View on a reference collection.

    Args:
        base_dir: Path to the directory where the view is stored.
        view_ref: Access properties for the reference view.
        ds: The dataset handled by this view.
        filesystem: The file system used to access the view.
        filters: The filters used to select the partitions of the reference. If
            not provided, all partitions are selected.
        synchronizer: The synchronizer used to synchronize the view.

    .. note::

        Normally, you should not call this constructor directly. Instead, use
        :func:`create_view <zcollection.create_view>` or :func:`open_view
        <zcollection.open_view>` to create or open a view.
    """
    #: Configuration filename of the view.
    CONFIG: ClassVar[str] = '.view'

    def __init__(
        self,
        base_dir: str,
        view_ref: ViewReference,
        *,
        ds: meta.Dataset | None,
        filesystem: fsspec.AbstractFileSystem | str | None = None,
        filters: collection.PartitionFilter = None,
        synchronizer: sync.Sync | None = None,
    ) -> None:
        #: The file system used to access the view (default local file system).
        self.fs = fs_utils.get_fs(filesystem)
        #: Path to the directory where the view is stored.
        self.base_dir = fs_utils.normalize_path(self.fs, base_dir)
        #: The reference collection of the view.
        self.view_ref = convenience.open_collection(
            view_ref.path, mode='r', filesystem=view_ref.filesystem)
        #: The metadata of the variables handled by the view.
        self.metadata = ds or meta.Dataset(
            self.view_ref.metadata.dimensions, variables=[], attrs=[])
        #: The synchronizer used to synchronize the view.
        self.synchronizer = synchronizer or sync.NoSync()
        #: The filters used to select the partitions of the reference.
        self.filters = filters

        if not self.fs.exists(self.base_dir):
            _LOGGER.info('Creating view %s', self)
            self.fs.makedirs(self.base_dir)
            self._write_config()
            self._init_partitions(filters)
        else:
            _LOGGER.info('Opening view %s', self)

    def _init_partitions(self, filters: collection.PartitionFilter) -> None:
        """Initialize the partitions of the view."""
        _LOGGER.info('Populating view %s', self)
        args = tuple(
            map(lambda item: fs_utils.join_path(self.base_dir, item),
                self.view_ref.partitions(filters=filters, relative=True)))
        # When opening an existing view, if the user asks to use new partitions
        # from the reference collection, in this case only the missing
        # partitions are created.
        args = tuple(filter(lambda item: not self.fs.exists(item), args))
        _LOGGER.info('%d partitions selected from %s', len(args),
                     self.view_ref)
        client = dask_utils.get_client()
        storage.execute_transaction(
            client, self.synchronizer,
            client.map(
                _write_checksum,
                tuple(args),
                base_dir=self.base_dir,
                view_ref=self.view_ref,
                fs=self.fs,
            ))

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
        fs = json.loads(self.view_ref.fs.to_json())
        with self.fs.open(config, mode='w') as stream:
            json.dump(
                {
                    'base_dir': self.base_dir,
                    'filters': _serialize_filters(self.filters),
                    'metadata': self.metadata.get_config(),
                    'view_ref': {
                        'path': self.view_ref.partition_properties.dir,
                        'fs': fs,
                    },
                },
                stream,  # type: ignore[arg-type]
                indent=4)
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
            data = json.load(stream)

        view_ref = data['view_ref']
        return View(data['base_dir'],
                    ViewReference(
                        view_ref['path'],
                        fsspec.AbstractFileSystem.from_json(
                            json.dumps(view_ref['fs']))),
                    ds=meta.Dataset.from_config(data['metadata']),
                    filesystem=filesystem,
                    filters=_deserialize_filters(data['filters']),
                    synchronizer=synchronizer)

    def partitions(
        self,
        filters: collection.PartitionFilter = None,
    ) -> Iterator[str]:
        """Returns the list of partitions in the view.

        Args:
            filters: The partition filters.

        Returns:
            The list of partitions.
        """
        return filter(
            self.fs.exists,
            map(lambda item: fs_utils.join_path(self.base_dir, item),
                self.view_ref.partitions(filters=filters, relative=True)))

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
        return collection.variables(self.metadata, selected_variables)

    def add_variable(
        self,
        variable: meta.Variable | dataset.Variable,
    ) -> None:
        """Add a variable to the view.

        Args:
            variable: The variable to add

        Raises:
            ValueError: If the variable already exists

        Example:
            >>> view.add_variable(
            ...    zcollection.meta.Variable(
            ...        "temperature",
            ...        "float32", ("time", "lat", "lon"),
            ...        (zcollection.meta.Attribute("units", "degrees Celsius"),
            ...         zcollection.meta.Attribute("long_name", "temperature")),
            ...        fill_value=-9999.0))
        """
        # pylint: disable=duplicate-code
        # false positive, no code duplication

        variable = dataset.get_variable_metadata(variable)
        _LOGGER.info('Adding variable %r in the view', variable.name)
        if (variable.name in self.view_ref.metadata.variables
                or variable.name in self.metadata.variables):
            raise ValueError(f'Variable {variable.name} already exists')
        client = dask_utils.get_client()
        self.metadata.add_variable(variable)
        template = self.view_ref.metadata.search_same_dimensions_as(variable)

        existing_partitions = {
            pathlib.Path(path).relative_to(self.base_dir).as_posix()
            for path in self.partitions()
        }

        if len(existing_partitions) == 0:
            _LOGGER.info('No partitions found, skipping variable creation')
            return

        args = filter(lambda item: item[0] in existing_partitions,
                      self.view_ref.iterate_on_records(relative=True))
        try:
            storage.execute_transaction(
                client, self.synchronizer,
                client.map(_create_zarr_array,
                           tuple(args),
                           base_dir=self.base_dir,
                           fs=self.fs,
                           template=template.name,
                           variable=variable))
        except Exception:
            storage.execute_transaction(
                client, self.synchronizer,
                client.map(_drop_zarr_zarr,
                           tuple(self.partitions()),
                           fs=self.fs,
                           variable=variable.name,
                           ignore_errors=True))
            raise

        self._write_config()
        # pylint: enable=duplicate-code

    def drop_variable(
        self,
        varname: str,
    ) -> None:
        """Drop a variable from the view.

        Args:
            varname: The name of the variable to drop.

        Raise:
            ValueError: If the variable does not exist or if the variable
                belongs to the reference collection.

        Example:
            >>> view.drop_variable("temperature")
        """
        _LOGGER.info('Dropping variable %r', varname)
        _assert_variable_handled(self.view_ref.metadata, self.metadata,
                                 varname)
        client = dask_utils.get_client()

        variable = self.metadata.variables.pop(varname)
        self._write_config()

        storage.execute_transaction(
            client, self.synchronizer,
            client.map(_drop_zarr_zarr,
                       tuple(self.partitions()),
                       fs=self.fs,
                       variable=variable.name))

    def load(
        self,
        *,
        filters: collection.PartitionFilter = None,
        indexer: collection.Indexer | None = None,
        selected_variables: Iterable[str] | None = None,
    ) -> dataset.Dataset | None:
        """Load the view.

        Args:
            filters: The predicate used to filter the partitions to select.
                To get more information on the predicate, see the
                documentation of the :meth:`Collection.partitions
                <zcollection.collection.Collection.partitions>` method.
            indexer: The indexer to apply.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded.

        Returns:
            The dataset.

        Example:
            >>> view.load()
            >>> view.load(filters="time == '2020-01-01'")
            >>> view.load(filters=lambda x: x["time"] == "2020-01-01")
        """
        _assert_have_variables(self.metadata)
        if indexer is not None:
            arguments = tuple(
                collection.build_indexer_args(self.view_ref,
                                              filters,
                                              indexer,
                                              partitions=self.partitions()))
            if len(arguments) == 0:
                return None
        else:
            arguments = tuple((self.view_ref.partitioning.parse(item), [])
                              for item in self.partitions(filters=filters))

        client = dask_utils.get_client()
        futures = client.map(
            _load_one_dataset,
            arguments,
            base_dir=self.base_dir,
            fs=self.fs,
            selected_variables=self.view_ref.metadata.select_variables(
                selected_variables),
            view_ref=client.scatter(self.view_ref),
            variables=self.metadata.select_variables(selected_variables))

        # The load function returns the path to the partitions and the loaded
        # datasets. Only the loaded datasets are retrieved here and filter None
        # values corresponding to empty partitions.
        arrays: list[dataset.Dataset] = list(
            map(
                lambda item: item[0],  # type: ignore[arg-type]
                filter(lambda item: item is not None,
                       client.gather(futures))))  # type: ignore[arg-type]
        if arrays:
            array = arrays.pop(0)
            if arrays:
                array = array.concat(arrays,
                                     self.view_ref.partition_properties.dim)
            metadata: meta.Dataset = copy.deepcopy(self.view_ref.metadata)
            metadata.variables.update(self.metadata.variables.items())
            array.fill_attrs(metadata)
            return array
        return None

    def update(
        self,
        func: collection.UpdateCallable,
        /,
        *args,
        depth: int = 0,
        filters: collection.PartitionFilter = None,
        partition_size: int | None = None,
        selected_variables: Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        """Update a variable stored int the view.

        Args:
            func: The function to apply to calculate the new values for the
                target variables.
            depth: The depth of the overlap between the partitions. Default is
                0 (no overlap). If depth is greater than 0, the function is
                applied on the partition and its neighbors selected by the
                depth. If ``func`` accepts a partition_info as a keyword
                argument, it will be passed a tuple with the name of the
                partitioned dimension and the slice allowing getting in the
                dataset the selected partition without the overlap.
            filters: The predicate used to filter the partitions to drop.
                To get more information on the predicate, see the
                documentation of the :meth:`Collection.partitions
                <zcollection.collection.Collection.partitions>` method.
            partition_size: The number of partitions to update in a single
                batch. By default 1, which is the same as to map the function to
                each partition. Otherwise, the function is called on a batch of
                partitions.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded. Useful to load only a
                subset of the view.
            args: The positional arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.

        Raises:
            ValueError: If the variable does not exist or if the variable
                belongs to the reference collection.

        Example:
            >>> def temp_celsius_to_kelvin(
            ...     dataset: zcollection.dataset.Dataset,
            ... ) -> Dict[str, numpy.ndarray]:
            ...     return dict(
            ...         temperature_kelvin=dataset["temperature"].values + 273,
            ...         15)
            >>> view.update(update_temperature)
        """
        _assert_have_variables(self.metadata)

        client = dask_utils.get_client()

        datasets_list = tuple(
            _load_datasets_list(client, self.base_dir, self.fs,
                                self.view_ref, self.metadata,
                                self.partitions(filters), selected_variables))

        # If no dataset is selected, we have nothing to do.
        if not datasets_list:
            warnings.warn('The update function is not applied because no '
                          'data is selected with the given filters.')
            return

        func_result = try_infer_callable(
            func, datasets_list[0][0], self.view_ref.partition_properties.dim,
            *args, **kwargs)
        tuple(
            map(
                lambda varname: _assert_variable_handled(
                    self.view_ref.metadata, self.metadata, varname),
                func_result))
        _LOGGER.info('Updating variable %s',
                     ', '.join(repr(item) for item in func_result))

        # Wrap the function to apply to each partition.
        if depth == 0:
            wrap_function = _wrap_update_func(
                func,
                self.fs,
                *args,
                **kwargs,
            )
        else:
            wrap_function = _wrap_update_func_overlap(
                datasets_list,
                depth,
                func,
                self.fs,
                self.view_ref,
                *args,
                **kwargs,
            )

        batchs = dask_utils.split_sequence(
            datasets_list, partition_size
            or dask_utils.dask_workers(client, cores_only=True))
        awaitables = client.map(wrap_function,
                                tuple(batchs),
                                key=func.__name__,
                                base_dir=self.base_dir)
        storage.execute_transaction(client, self.synchronizer, awaitables)

    # pylint: disable=duplicate-code
    # false positive, no code duplication
    def map(
        self,
        func: collection.MapCallable,
        *args,
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
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`zcollection.Collection.partitions`
                method.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded. Useful to load only a
                subset of the view.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag containing the tuple of the partition scheme and the result
            of the function.

        Example:
            >>> futures = view.map(
            ...     lambda x: (x["var1"] + x["var2"]).values)
            >>> for item in futures:
            ...     print(item)
            [1.0, 2.0, 3.0, 4.0]
            [5.0, 6.0, 7.0, 8.0]
        """

        def _wrap(
            arguments: tuple[dataset.Dataset, str],
            func: collection.PartitionCallable,
            *args,
            **kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                arguments: The partition scheme and the dataset.
                func: The function to apply.
                *args: The positional arguments to pass to the function.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            ds, partition = arguments
            return self.view_ref.partitioning.parse(partition), func(
                ds, *args, **kwargs)

        _assert_have_variables(self.metadata)

        client = dask_utils.get_client()
        datasets_list = tuple(
            _load_datasets_list(client, self.base_dir, self.fs,
                                self.view_ref, self.metadata,
                                self.partitions(filters), selected_variables))
        bag = dask.bag.core.from_sequence(datasets_list,
                                          partition_size=partition_size,
                                          npartitions=npartitions)
        return bag.map(_wrap, func, *args, **kwargs)
        # pylint: enable=duplicate-code

    def map_overlap(
        self,
        func: collection.MapCallable,
        depth: int,
        *args,
        filters: collection.PartitionFilter = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Iterable[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the view with some overlap.

        Args:
            func: The function to apply to every partition of the view.
            depth: The depth of the overlap between the partitions. Default is
                0 (no overlap). If depth is greater than 0, the function is
                applied on the partition and its neighbors selected by the
                depth. If ``func`` accepts a partition_info as a keyword
                argument, it will be passed a tuple with the name of the
                partitioned dimension and the slice allowing getting in the
                dataset the selected partition without the overlap.
            *args: The positional arguments to pass to the function.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`zcollection.Collection.partitions`
                method.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the view.
                If None, all variables are loaded. Useful to load only a
                subset of the view.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag containing the tuple of the partition scheme and the result
            of the function.

        Example:
            >>> futures = view.map_overlap(
            ...     lambda x: (x["var1"] + x["var2"]).values,
            ...     depth=1)
            >>> for item in futures:
            ...     print(item)
            [1.0, 2.0, 3.0, 4.0]
        """
        if depth < 0:
            raise ValueError('Depth must be greater than or equal to 0')

        add_partition_info = dask.utils.has_keyword(func, 'partition_info')

        def _wrap(
            arguments: tuple[dataset.Dataset, str],
            func: collection.PartitionCallable,
            datasets_list: tuple[tuple[dataset.Dataset, str]],
            depth: int,
            *args,
            **kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                arguments: The partition scheme and the dataset.
                func: The function to apply.
                datasets: The datasets to apply the function on.
                depth: The depth of the overlap between the partitions.
                *args: The positional arguments to pass to the function.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            ds, indices = _select_overlap(arguments, datasets_list, depth,
                                          self.view_ref)

            if add_partition_info:
                kwargs = kwargs.copy()
                kwargs['partition_info'] = (
                    self.view_ref.partition_properties.dim, indices)

            # Finally, apply the function.
            return (self.view_ref.partitioning.parse(arguments[1]),
                    func(ds, *args, **kwargs))

        _assert_have_variables(self.metadata)

        client = dask_utils.get_client()
        datasets_list = tuple(
            _load_datasets_list(client, self.base_dir, self.fs,
                                self.view_ref, self.metadata,
                                self.partitions(filters), selected_variables))
        bag = dask.bag.core.from_sequence(datasets_list,
                                          partition_size=partition_size,
                                          npartitions=npartitions)
        return bag.map(_wrap, func, datasets_list, depth, *args, **kwargs)

    def is_synced(self) -> bool:
        """Check if the view is synchronized with the underlying collection.

        Returns:
            True if the view is synchronized, False otherwise.
        """
        partitions = tuple(self.view_ref.partitions(relative=True))
        client = dask_utils.get_client()
        unsynchronized_partition = storage.execute_transaction(
            client, self.synchronizer,
            client.map(_sync,
                       partitions,
                       base_dir=self.base_dir,
                       fs=self.fs,
                       view_ref=self.view_ref,
                       metadata=self.metadata,
                       dry_run=True))
        return len(
            tuple(
                filter(lambda item: item is not None,
                       unsynchronized_partition))) == 0

    def sync(
        self,
        filters: collection.PartitionFilter = None
    ) -> collection.PartitionFilterCallback:
        """Synchronize the view with the underlying collection.

        This method is useful to update the view after a change in the
        underlying collection.

        Args:
            filters: The predicate used to select the partitions to
                synchronize. To get more information on the predicate, see the
                documentation of the :meth:`zcollection.Collection.partitions`
                method.
                If None, the view is synchronized with all the partitions
                already present in the view. If you want to extend the view
                with new partitions, use must provide a predicate that
                selects the new partitions.
                Existing partitions are not removed, even if they are not
                selected by the predicate.

        Returns:
            A function that can be used as a predicate to get the partitions
            that have been synchronized using the :meth:`View.partitions`
            method.
        """
        _LOGGER.info('Synchronizing view %s', self)

        if filters is not None:
            self.filters = filters
            self._write_config()
            self._init_partitions(filters)

        partitions = tuple(self.view_ref.partitions(relative=True))
        _LOGGER.info('%d partitions to synchronize', len(partitions))

        client = dask_utils.get_client()
        synchronized_partition = storage.execute_transaction(
            client, self.synchronizer,
            client.map(_sync,
                       partitions,
                       base_dir=self.base_dir,
                       fs=self.fs,
                       view_ref=self.view_ref,
                       metadata=self.metadata))
        partition_ids = []
        for item in filter(lambda item: item is not None,
                           synchronized_partition):
            partition_ids.append(dict(self.view_ref.partitioning.parse(item)))
        return lambda item: item in partition_ids
