# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
View on a reference collection.
===============================
"""
from typing import (
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import dataclasses
import json
import logging
import pathlib

import dask.array.core
import dask.bag.core
import dask.distributed
import fsspec
import zarr

from . import collection, dataset, meta, storage, sync, utilities
from .convenience import collection as convenience

#: Module logger.
_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ViewReference:
    """Properties of the collection used as reference by a view.

    Args:
        path: Path to the collection.
        filesystem: The file system used to access the reference collection.
    """
    #: Path to the collection.
    path: str
    #: The file system used to access the reference collection.
    filesystem: fsspec.AbstractFileSystem = utilities.get_fs('file')


def _create_zarr_array(args: Tuple[str, zarr.Group], base_dir: str,
                       fs: fsspec.AbstractFileSystem, template: str,
                       variable: meta.Variable) -> None:
    """Create a Zarr array, with fill_value being used as the default value for
    uninitialized portions of the array.

    Args:
        args: Tuple of (path, zarr.Group).
        base_dir: Base directory for the Zarr array.
        fs: The filesystem used to create the Zarr array.
        template: The variable's name is used as a template for the Zarr array
            to determine the shape of the new variable.
        variable: The properties of the variable to create.
    """
    partition, group = args
    data: dask.array.core.Array = dask.array.core.from_zarr(group[template])

    dirname = fs.sep.join((base_dir, partition))
    mapper = fs.get_mapper(fs.sep.join((dirname, variable.name)))
    zarr.full(data.shape,
              chunks=data.chunksize,
              dtype=variable.dtype,
              compressor=variable.compressor,
              fill_value=variable.fill_value,
              store=mapper,
              overwrite=True,
              filters=variable.filters)
    storage.write_zattrs(dirname, variable, fs)
    fs.invalidate_cache(dirname)


def _drop_zarr_zarr(partition: str,
                    fs: fsspec.AbstractFileSystem,
                    variable: str,
                    ignore_errors: bool = False) -> None:
    """Drop a Zarr array.

    Args:
        partition: The partition that contains the array to drop.
        base_dir: Base directory for the Zarr array.
        fs: The filesystem used to delete the Zarr array.
        variable: The name of the variable to drop.
        ignore_errors: If True, ignore errors when dropping the array.
    """
    try:
        fs.rm(fs.sep.join((partition, variable)), recursive=True)
        fs.invalidate_cache(partition)
    # pylint: disable=broad-except
    # We don't want to fail on errors.
    except Exception:
        if not ignore_errors:
            raise
    # pylint: enable=broad-except


def _load_one_dataset(
    args: Tuple[Tuple[Tuple[str, int], ...], List[slice]],
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    selected_variables: Optional[Iterable[str]],
    view_ref: collection.Collection,
    variables: Sequence[str],
) -> Optional[Tuple[dataset.Dataset, str]]:
    """Load a dataset from a partition stored in the reference collection and
    merge it with the variables defined in this view.

    Args:
        args: Tuple containing the partition's keys and its indexer.
        base_dir: Base directory of the view.
        fs: The file system used to access the variables in the view.
        selected_variables: The list of variable to retain from the view
            reference.
        view_ref: The view reference.
        variables: The variables to retain from the view

    Returns:
        The dataset and the partition's path.
    """
    partition_scheme, slices = args
    partition = view_ref.partitioning.join(partition_scheme, fs.sep)
    ds = storage.open_zarr_group(
        view_ref.fs.sep.join((view_ref.partition_properties.dir, partition)),
        view_ref.fs, selected_variables)
    if ds is None:
        return None

    # If the user has not selected any variables in the reference view. In this
    # case, the dataset is built from all the variables selected in the view.
    if len(ds.dimensions) == 0:
        return dataset.Dataset(
            [
                storage.open_zarr_array(
                    zarr.open(  # type: ignore[arg-type]
                        fs.get_mapper(
                            fs.sep.join((base_dir, partition, variable))),
                        mode='r',
                    ),
                    variable) for variable in variables
            ],
            ds.attrs), partition

    _ = {
        ds.add_variable(item.metadata(), item.array)  # type: ignore[arg-type]
        for item in (
            storage.open_zarr_array(
                zarr.open(  # type: ignore[arg-type]
                    fs.get_mapper(fs.sep.join((base_dir, partition,
                                               variable))),
                    mode='r',
                ),
                variable) for variable in variables)
    }

    # Apply indexing if needed.
    if len(slices):
        dim = view_ref.partition_properties.dim
        ds_list: List[dataset.Dataset] = []
        _ = {
            ds_list.append(  # type: ignore[func-returns-value]
                ds.isel({dim: indexer}))
            for indexer in slices
        }
        ds = ds_list.pop(0)
        if ds_list:
            ds = ds.concat(ds_list, dim)
    return ds, partition


def _assert_variable_handled(reference: meta.Dataset, view: meta.Dataset,
                             variable: str) -> None:
    """Assert that a variable belongs to a view.

    Args:
        reference: The reference dataset.
        view: The view dataset.
        variable: The variable to check.
    """
    if variable in reference.variables:
        raise ValueError(f'Variable {variable} is read-only')
    if variable not in view.variables:
        raise ValueError(f'Variable {variable} does not exist')


def _load_datasets_list(
    client: dask.distributed.Client,
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    metadata: meta.Dataset,
    partitions: Iterable[str],
    selected_variables: Optional[Iterable[str]] = None,
) -> Iterator[Tuple[dataset.Dataset, str]]:
    """Load datasets from a list of partitions.

    Args:
        client: The client used to load the datasets.
        base_dir: Base directory of the view.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
        metadata: The metadata of the dataset.
        partitions: The list of partitions to load.
        selected_variables: The list of variable to retain from the view

    Returns:
        The datasets and their paths.
    """
    arguments: Tuple[Tuple[Tuple[Tuple[str, int], ...], List], ...] = tuple(
        (view_ref.partitioning.parse(item), []) for item in partitions)
    futures = client.map(
        _load_one_dataset,
        arguments,
        base_dir=base_dir,
        fs=fs,
        selected_variables=view_ref.metadata.select_variables(
            keep_variables=selected_variables),
        view_ref=client.scatter(view_ref),
        variables=metadata.select_variables(selected_variables))

    return filter(lambda item: item is not None,
                  client.gather(futures))  # type: ignore[arg-type]


def _assert_have_variables(metadata: meta.Dataset) -> None:
    """Assert that the current view has variables.

    Args:
        metadata: The metadata of the dataset.
    """
    if not metadata.variables:
        raise ValueError('The view has no variables')


class View:
    """View on a reference collection.

    Args:
        base_dir: Path to the directory where the view is stored.
        view_ref: Access properties for the reference view.
        ds: The dataset handled by this view.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.
    """
    #: Configuration filename of the view.
    CONFIG: ClassVar[str] = '.view'

    def __init__(
        self,
        base_dir: str,
        view_ref: ViewReference,
        *,
        ds: Optional[meta.Dataset] = None,
        filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
        synchronizer: Optional[sync.Sync] = None,
    ) -> None:
        #: The file system used to access the view (default local file system).
        self.fs = utilities.get_fs(filesystem)
        #: Path to the directory where the view is stored.
        self.base_dir = utilities.normalize_path(self.fs, base_dir)
        #: The reference collection of the view.
        self.view_ref = convenience.open_collection(
            view_ref.path, mode='r', filesystem=view_ref.filesystem)
        #: The metadata of the variables handled by the view.
        self.metadata = ds or meta.Dataset(
            self.view_ref.metadata.dimensions, variables=[], attrs=[])
        #: The synchronizer used to synchronize the view.
        self.synchronizer = synchronizer or sync.NoSync()

        if not self.fs.exists(self.base_dir):
            _LOGGER.info('Creating view %s', self)
            self.fs.makedirs(self.base_dir)
            self._write_config()

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'<filesystem={self.fs.__class__.__name__!r}, '
                f'base_dir={self.base_dir!r}>')

    @classmethod
    def _config(cls, base_dir: str, fs: fsspec.AbstractFileSystem) -> str:
        """Returns the configuration path."""
        return fs.sep.join((base_dir, cls.CONFIG))

    def _write_config(self) -> None:
        """Write the configuration file for the view."""
        config = self._config(self.base_dir, self.fs)
        fs = json.loads(self.view_ref.fs.to_json())
        with self.fs.open(config, mode='w') as stream:
            json.dump(
                dict(base_dir=self.base_dir,
                     metadata=self.metadata.get_config(),
                     view_ref=dict(path=self.view_ref.partition_properties.dir,
                                   fs=fs)),
                stream,  # type: ignore[arg-type]
                indent=4)
        self.fs.invalidate_cache(config)

    @classmethod
    def from_config(
        cls,
        path: str,
        *,
        filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
        synchronizer: Optional[sync.Sync] = None,
    ) -> 'View':
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
        fs = utilities.get_fs(filesystem)
        config = cls._config(path, fs)
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
            map(lambda item: self.fs.sep.join((self.base_dir, item)),
                self.view_ref.partitions(filters=filters, relative=True)))

    def variables(
        self,
        selected_variables: Optional[Iterable[str]] = None
    ) -> Tuple[dataset.Variable, ...]:
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
        variable: Union[meta.Variable, dataset.Variable],
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
        client = utilities.get_client()
        self.metadata.add_variable(variable)
        template = self.view_ref.metadata.search_same_dimensions_as(variable)

        existing_partitions: Iterable[str] = tuple(self.partitions())

        # If the view already contains variables, you only need to modify the
        # existing partitions.
        if existing_partitions:
            existing_partitions = {
                pathlib.Path(path).relative_to(self.base_dir).as_posix()
                for path in existing_partitions
            }
            args: Any = filter(lambda item: item[0] in existing_partitions,
                               self.view_ref.iterate_on_records(relative=True))
        else:
            args = self.view_ref.iterate_on_records(relative=True)

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
        client = utilities.get_client()

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
        indexer: Optional[collection.Indexer] = None,
        selected_variables: Optional[Iterable[str]] = None,
    ) -> Optional[dataset.Dataset]:
        """Load the view.

        Args:
            filters: The predicate used to filter the partitions to drop.
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

        client = utilities.get_client()
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
        arrays: List[dataset.Dataset] = list(
            map(
                lambda item: item[0],  # type: ignore[arg-type]
                filter(lambda item: item is not None,
                       client.gather(futures))))  # type: ignore[arg-type]
        if arrays:
            array = arrays.pop(0)
            if arrays:
                array = array.concat(arrays,
                                     self.view_ref.partition_properties.dim)
            return array
        return None

    def update(
        self,
        func: collection.UpdateCallable,
        /,
        *args,
        filters: collection.PartitionFilter = None,
        partition_size: Optional[int] = None,
        selected_variables: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:
        """Update a variable stored int the view.

        Args:
            func: The function to apply to calculate the new values for the
                target variables.
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
        client = utilities.get_client()

        datasets_list = tuple(
            _load_datasets_list(client, self.base_dir, self.fs,
                                self.view_ref, self.metadata,
                                self.partitions(filters), selected_variables))

        func_result = func(datasets_list[0][0], *args, **kwargs)
        tuple(
            map(
                lambda varname: _assert_variable_handled(
                    self.view_ref.metadata, self.metadata, varname),
                func_result))
        _LOGGER.info('Updating variable %s',
                     ', '.join(repr(item) for item in func_result))

        def wrap_function(parameters: Iterable[Tuple[dataset.Dataset, str]],
                          base_dir: str) -> None:
            """Wrap the function to be applied to the dataset."""
            for ds, partition in parameters:
                # Applying function on partition's data
                dictionary = func(ds, *args, **kwargs)
                tuple(
                    storage.
                    update_zarr_array(  # type: ignore[func-returns-value]
                        dirname=self.fs.sep.join((base_dir, partition,
                                                  varname)),
                        array=array,
                        fs=self.fs,
                    ) for varname, array in dictionary.items())

        batchs = utilities.split_sequence(
            datasets_list, partition_size
            or utilities.dask_workers(client, cores_only=True))
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
        partition_size: Optional[int] = None,
        npartitions: Optional[int] = None,
        selected_variables: Optional[Iterable[str]] = None,
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
            arguments: Tuple[dataset.Dataset, str],
            func: collection.PartitionCallable,
            *args,
            **kwargs,
        ) -> Tuple[Tuple[Tuple[str, int], ...], Any]:
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

        client = utilities.get_client()
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
        partition_size: Optional[int] = None,
        npartitions: Optional[int] = None,
        selected_variables: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the view with some overlap.

        Args:
            func: The function to apply to every partition of the view.
            depth: The depth of the overlap between the partitions.
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

        def _wrap(
            arguments: Tuple[dataset.Dataset, str],
            func: collection.PartitionCallable,
            datasets_list: Tuple[Tuple[dataset.Dataset, str]],
            depth: int,
            *args,
            **kwargs,
        ) -> Tuple[Tuple[Tuple[str, int], ...], slice, Any]:
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
            # pylint: disable=too-many-locals
            # The local function is not taken into account for counting
            # locals.
            _, partition = arguments
            where = next(ix for ix, item in enumerate(datasets_list)
                         if item[1] == partition)

            # Search for the overlapping partitions
            selected_datasets = [
                datasets_list[ix]
                for ix in range(where - depth, where + depth + 1)
                if 0 <= ix < len(datasets_list)
            ]

            # Compute the slice of the given partition.
            start = 0
            indices = slice(0, 0, None)
            for ds, current_partition in datasets_list:
                size = ds[self.view_ref.axis].size
                indices = slice(start, start + size, None)
                if partition == current_partition:
                    break
                start += size

            # Build the dataset for the selected partitions.
            groups = [ds for ds, _ in selected_datasets]
            ds = groups.pop(0)
            ds.concat(groups, self.view_ref.partition_properties.dim)

            # Finally, apply the function.
            return (self.view_ref.partitioning.parse(partition), indices,
                    func(ds, *args, **kwargs))
            # pylint: enable=too-many-locals

        _assert_have_variables(self.metadata)

        client = utilities.get_client()
        datasets_list = tuple(
            _load_datasets_list(client, self.base_dir, self.fs,
                                self.view_ref, self.metadata,
                                self.partitions(filters), selected_variables))
        bag = dask.bag.core.from_sequence(datasets_list,
                                          partition_size=partition_size,
                                          npartitions=npartitions)
        return bag.map(_wrap, func, datasets_list, depth, *args, **kwargs)
