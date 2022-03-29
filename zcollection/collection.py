# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Collection of Zarr groups
=========================
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)
import dataclasses
import io
import itertools
import json
import logging
import pathlib
import types

import dask
import dask.array
import dask.bag
import dask.distributed
import dask.utils
import fsspec
import numpy
import xarray
import zarr
import zarr.storage

from . import (
    dataset,
    expression,
    merging,
    meta,
    partitioning,
    storage,
    sync,
    utilities,
)

#: Function type to load and call a callback function of type
#: :class:`PartitionCallable`.
WrappedPartitionCallable = Callable[[str], None]

#: Type of functions filtering the partitions.
PartitionFilterCallback = Callable[[Dict[str, int]], bool]

#: Type of argument to filter the partitions.
PartitionFilter = Optional[Union[str, PartitionFilterCallback]]

#: Indexer's type.
Indexer = Iterable[Tuple[Tuple[Tuple[str, int], ...], slice]]

#: Module logger.
_LOGGER = logging.getLogger(__name__)


#: pylint: disable=too-few-public-methods
class PartitionCallable(Protocol):
    """Protocol for partition callables.

    A partition callable is a function that accepts a dataset and returns a
    result.
    """

    def __call__(self, ds: dataset.Dataset, *args, **kwargs) -> Any:
        """Call the partition function.

        Args:
            ds: Dataset to partition.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the partition function.
        """
        ...  # pragma: no cover


class MapCallable(Protocol):
    """Protocol for map callables.

    A callable map is a function that accepts a data set, a list of arguments,
    a dictionary of keyword arguments and returns a result.
    """

    def __call__(self, ds: dataset.Dataset, *args, **kwargs) -> Any:
        """Call the map function.

        Args:
            ds: Dataset to map.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the map function.
        """
        ...  # pragma: no cover

    #: pylint: enable=too-few-public-methods


@dataclasses.dataclass(frozen=True)
class PartitioningProperties:
    """
    Properties of a partition.
    """
    #: The base directory of the partition.
    dir: str
    #: The name of the partitioning dimension.
    dim: str


def _wrap_update_func(
    func: PartitionCallable,
    fs: fsspec.AbstractFileSystem,
    variable: str,
    *args,
    **kwargs,
) -> WrappedPartitionCallable:
    """
    Wrap an update function taking a partition's dataset as input and
    returning variable's values as a numpy array.

    Args:
        func: Function to apply on each partition.
        fs: File system on which the Zarr dataset is stored.
        variable: Name of the variable to update.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The wrapped function that takes a partition's dataset and a variable
        name as input and returns the variable's values as a numpy array.
    """

    def wrap_function(partition: str) -> None:
        # Applying function on partition's data
        array = func(storage.open_zarr_group(partition, fs), *args, **kwargs)

        storage.update_zarr_array(dirname=fs.sep.join((partition, variable)),
                                  array=array,
                                  fs=fs)

    return wrap_function


def _insert(
    args: Tuple[Tuple[str, ...], Dict[str, slice]],
    axis: str,
    ds: dataset.Dataset,
    fs: fsspec.AbstractFileSystem,
    merge_callable: Optional[merging.MergeCallable],
    partitioning_properties: PartitioningProperties,
) -> None:
    """Insert or update a partition in the collection.

    Args:
        args: Tuple containing the partition's name and its slice.
        axis: The axis to merge on.
        ds: The dataset to process.
        fs: The file system that the partition is stored on.
        merge_callable: The merge callable.
        partitioning_properties: The partitioning properties.
    """
    partition, indexer = args
    dirname = fs.sep.join((partitioning_properties.dir, ) + partition)

    # If the consolidated zarr metadata does not exist, we consider the
    # partition as empty.
    if fs.exists(fs.sep.join((dirname, ".zmetadata"))):
        # The current partition already exists, so we need to merge
        # the dataset.
        merging.perform(ds.isel(indexer), dirname, axis, fs,
                        partitioning_properties.dim, merge_callable)
        return

    # The current partition does not exist, so we need to create
    # it and insert the dataset.
    try:
        zarr.storage.init_group(store=fs.get_mapper(dirname))

        # The synchronization is done by the caller.
        storage.write_zarr_group(ds.isel(indexer), dirname, fs, sync.NoSync())
    except:  # noqa: E722
        # If the construction of the new dataset fails, the created
        # partition is deleted, to guarantee the integrity of the
        # collection.
        fs.rm(dirname, recursive=True)
        fs.invalidate_cache(dirname)
        raise


def _load_and_apply_indexer(
    args: Tuple[Tuple[Tuple[str, int], ...], List[slice]],
    fs: fsspec.AbstractFileSystem,
    partition_handler: partitioning.Partitioning,
    partition_properties: PartitioningProperties,
    selected_variables: Optional[Iterable[str]],
) -> List[dataset.Dataset]:
    """Load a partition and apply its indexer.

    Args:
        args: Tuple containing the partition's name and its indexer.
        fs: The file system that the partition is stored on.
        partition_handler: The partitioning handler.
        partition_properties: The partitioning properties.
        selected_variable: The selected variables to load.

    Returns:
        The list of loaded datasets.
    """
    partition_scheme, items = args
    partition = fs.sep.join((partition_properties.dir,
                             partition_handler.join(partition_scheme, fs.sep)))

    ds = storage.open_zarr_group(partition, fs, selected_variables)
    arrays = []
    _ = {
        arrays.append(ds.isel({partition_properties.dim: indexer}))
        for indexer in items
    }
    return arrays


class Collection:
    """
    This class manages a collection of files in Zarr format stored in a set
    of subdirectories. These subdirectories split the data, by cycles or dates
    for example, in order to optimize access and updates, deletion or addition
    of new data.

    Args:
        axis: The axis of the collection.
        ds: The dataset containing the collection.
        partition_handler: The partitioning strategy for the collection.
        partition_base_dir: The base directory for the collection.
        mode: The mode of the collection.
        filesystem: The filesystem to use for the collection.
        synchronizer: The synchronizer to use for the collection.

    Raises:
        ValueError: If the axis does not exist in the dataset, if the
            partition key is not defined in the dataset or if the access mode
            is not supported.
    """
    #: Configuration filename of the collection.
    CONFIG: ClassVar[str] = ".zcollection"

    def __init__(
        self,
        axis: str,
        ds: meta.Dataset,
        partition_handler: partitioning.Partitioning,
        partition_base_dir: str,
        *,
        mode: Optional[str] = None,
        filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
        synchronizer: Optional[sync.Sync] = None,
    ) -> None:
        if axis not in ds.variables:
            raise ValueError(
                f"The variable {axis!r} is not defined in the dataset.")

        for varname in partition_handler.variables:
            if varname not in ds.variables:
                raise ValueError(
                    f"The partitioning key {varname!r} is not defined in "
                    "the dataset.")

        mode = mode or "w"
        if mode not in ("r", "w"):
            raise ValueError(f"The mode {mode!r} is not supported.")

        #: The axis of the collection.
        self.axis = axis
        #: The metadata that describes the dataset handled by the collection.
        self.metadata = ds
        #: The file system used to read/write the collection.
        self.fs = utilities.get_fs(filesystem)
        #: The partitioning strategy used to split the data.
        self.partitioning = partition_handler
        #: The partitioning properties (base directory and dimension).
        self.partition_properties = PartitioningProperties(
            partition_base_dir.rstrip(self.fs.sep),
            ds.variables[axis].dimensions[0],
        )
        #: The access mode of the collection.
        self.mode = mode
        #: The synchronizer used to synchronize the modifications.
        self.synchronizer = synchronizer or sync.NoSync()

        self._write_config(skip_if_exists=True)

        if mode == "r":
            # pylint: disable=method-hidden
            # These methods are overloaded when the collection is opened in
            # readonly.
            for item in [
                    "add_variable",
                    "drop_partitions",
                    "drop_variable",
                    "insert",
                    "update",
            ]:
                assert hasattr(self, item), f"{item} is not a known method."
                setattr(
                    self, item,
                    types.MethodType(Collection._unsupported_operation, self))
            # pylint: enable=method-hidden

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"<filesystem={self.fs.__class__.__name__!r}, "
                f"partition_base_dir={self.partition_properties.dir!r}, "
                f"mode={self.mode!r}>")

    @staticmethod
    def _unsupported_operation(*args, **kwargs):
        """Raise an exception if the operation is not supported."""
        raise io.UnsupportedOperation("not writable")

    @classmethod
    def _config(cls, partition_base_dir: str,
                fs: fsspec.AbstractFileSystem) -> str:
        """Return the configuration path"""
        return fs.sep.join((partition_base_dir, cls.CONFIG))

    def _write_config(self, skip_if_exists: bool = False) -> None:
        """Write the configuration file
        """
        base_dir = self.partition_properties.dir
        config = self._config(base_dir, self.fs)
        exists = self.fs.exists(config)

        message = ("Creating the collection: %s"
                   if exists else "Updating collection's configuration: %s")
        _LOGGER.info(message, config)
        if skip_if_exists and exists:
            return

        self.fs.makedirs(base_dir, exist_ok=True)

        params = dict(axis=self.axis,
                      dataset=self.metadata.get_config(),
                      partitioning=self.partitioning.get_config())

        with self.fs.open(config, mode="w") as stream:
            json.dump(params, stream, indent=4)  # type: ignore

    @classmethod
    def from_config(
        cls,
        path: str,
        *,
        mode: Optional[str] = None,
        filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
        synchronizer: Optional[sync.Sync] = None,
    ) -> "Collection":
        """Open a Collection described by a configuration file.

        Args:
            path: The path to the configuration file.
            mode: The mode of the collection.
            filesystem: The filesystem to use for the collection.
            synchronizer: The synchronizer to use for the collection.

        Returns:
            The collection.

        Raises:
            ValueError: If the provided directory does not contain a collection.
        """
        _LOGGER.info("Opening collection: %r", path)
        fs = utilities.get_fs(filesystem)
        config = cls._config(path, fs)
        if not fs.exists(config):
            raise ValueError(f"zarr collection not found at path {path!r}")
        with fs.open(config) as stream:
            data = json.load(stream)
        return Collection(
            data["axis"],
            meta.Dataset.from_config(data["dataset"]),
            partitioning.get_codecs(data["partitioning"]),
            path,
            mode=mode or "r",
            filesystem=fs,
            synchronizer=synchronizer,
        )

    def _is_selected(
        self,
        partition: Sequence[str],
        expr: Optional[Callable[[Dict[str, int]], bool]],
    ) -> bool:
        """Return whether the partition is selected.

        Args:
            partition: The partition to check.
            expr: The expression used to filter the partition.

        Returns:
            Whether the partition is selected.
        """
        if expr is not None:
            variables = dict(self.partitioning.parse("/".join(partition)))
            return expr(variables)
        return True

    # pylint: disable=method-hidden
    def insert(
        self,
        ds: Union[xarray.Dataset, dataset.Dataset],
        *,
        merge_callable: Optional[merging.MergeCallable] = None,
        parallel_tasks: Optional[int] = None,
    ) -> None:
        """Insert a dataset into the collection.

        Args:
            ds: The dataset to insert.
            merge_callable: The function to use to merge the existing data set
                already stored in partitions with the new partitioned data. If
                None, the new partitioned data overwrites the existing
                partitioned data.
            parallel_tasks: The maximum number of parallel tasks to use.
                By default, the number of available cores of the dask cluster
                is used.

        .. warning::

            Each worker will process a set of independent partitions. However,
            be careful, two different partitions can use the same file (chunk),
            therefore, the library that handles the storage of Dask arrays
            (HDF5, NetCDF, Zarr, etc.) must be compatible with concurrent
            access.

        Raises:
            ValueError: If the dataset mismatched the definition of the
                collection.
        """
        # pylint: disable=method-hidden
        if isinstance(ds, xarray.Dataset):
            ds = dataset.Dataset.from_xarray(ds)

        _LOGGER.info("Inserting of a %s dataset in the collection",
                     dask.utils.format_bytes(ds.nbytes))

        missing_variables = self.metadata.missing_variables(ds.metadata())
        for item in missing_variables:
            variable = self.metadata.variables[item]
            ds.add_variable(variable)

        client = utilities.get_client()

        utilities.calculation_stream(
            _insert,
            self.partitioning.split_dataset(ds, self.partition_properties.dim),
            max_workers=parallel_tasks,
            axis=self.axis,
            ds=client.scatter(ds),
            fs=self.fs,
            merge_callable=merge_callable,
            partitioning_properties=self.partition_properties)

    def _relative_path(self, path: str) -> str:
        """Return the relative path to the collection.

        Args:
            path: The path to the dataset.

        Returns:
            The relative path to the collection.
        """
        return pathlib.Path(path).relative_to(
            self.partition_properties.dir).as_posix()

    def partitions(
        self,
        *,
        filters: PartitionFilter = None,
        relative: bool = False,
    ) -> Iterator[str]:
        """List the partitions of the collection.

        Args:
            filters: The predicate used to filter the partitions to load. If
                the predicate is a string, it is a valid python expression to
                filter the partitions, using the partitioning scheme as
                variables. If the predicate is a function, it is a function
                that takes the partition scheme as input and returns a boolean.
            relative: Whether to return the relative path.

        Returns:
            The list of partitions.

        Example:
            >>> tuple(collection.partitions(
            ...     filters="year == 2019 and month == 1"))
            ('year=2019/month=01/day=01', 'year=2019/month=01/day=02/', ...)
            >>> tuple(collection.partitions(
            ...     filters=lambda x: x["year"] == 2019 and x["month"] == 1))
            ('year=2019/month=01/day=01', 'year=2019/month=01/day=02/', ...)
        """
        if isinstance(filters, str):
            expr = expression.Expression(filters)
        elif callable(filters):
            expr = filters
        else:
            expr = None

        base_dir = self.partition_properties.dir
        sep = self.fs.sep

        for item in self.partitioning.list_partitions(self.fs, base_dir):
            # Filtering on partition names
            partitions = item.replace(base_dir, "")
            entry = partitions.split(sep)

            if self._is_selected(entry, expr):
                yield self._relative_path(item) if relative else item

    # pylint: disable=method-hidden
    def drop_partitions(
        self,
        *,
        filters: PartitionFilter = None,
    ) -> None:
        # pylint: disable=method-hidden
        """Drop the selected partitions.

        Args:
            filters: The predicate used to filter the partitions to drop.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.

        Example:
            >>> collection.drop_partitions(filters="year == 2019")
        """
        client = utilities.get_client()
        folders = list(self.partitions(filters=filters))
        storage.execute_transaction(
            client, self.synchronizer,
            client.map(self.fs.rm, folders, recursive=True))
        for item in folders:
            _LOGGER.info("Dropped partition: %s", item)
            self.fs.invalidate_cache(item)

    def map(
        self,
        func: MapCallable,
        *args,
        filters: PartitionFilter = None,
        bag_partition_size: Optional[int] = None,
        bag_npartitions: Optional[int] = None,
        **kwargs,
    ) -> dask.bag.Bag:
        """Map a function over the partitions of the collection.

        Args:
            func: The function to apply to every partition of the collection.
            *args: The positional arguments to pass to the function.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            bag_partition_size: The length of each bag partition.
            bag_npartitions: The number of desired bag partitions.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag containing the tuple of the partition scheme and the result
            of the function.

        Example:
            >>> futures = collection.map(
            ...     lambda x: (x["var1"] + x["var2"]).values)
            >>> for item in futures:
            ...     print(item)
            [1.0, 2.0, 3.0, 4.0]
            [5.0, 6.0, 7.0, 8.0]
        """

        def _wrap(
            partition: str,
            func: PartitionCallable,
            *args,
            **kwargs,
        ) -> Tuple[Tuple[Tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                func: The function to apply.
                partition: The partition to apply the function on.
                *args: The positional arguments to pass to the function.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            ds = storage.open_zarr_group(partition, self.fs)
            return self.partitioning.parse(partition), func(
                ds, *args, **kwargs)

        bag = dask.bag.from_sequence(self.partitions(filters=filters),
                                     partition_size=bag_partition_size,
                                     npartitions=bag_npartitions)
        return bag.map(_wrap, func, *args, **kwargs)

    def load(
        self,
        *,
        filters: PartitionFilter = None,
        indexer: Optional[Indexer] = None,
        selected_variables: Optional[Iterable[str]] = None,
    ) -> Optional[dataset.Dataset]:
        """Load the selected partitions.

        Args:
            filters: The predicate used to filter the partitions to load.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            indexer: The indexer to apply.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.

        Returns:
            The dataset containing the selected partitions.

        .. warning::

            If you select variables to load from the collection, do not insert
            the returned dataset otherwise all skipped variables will be reset
            with fill values.

        Example:
            >>> collection = ...
            >>> collection.load(
            ...     filters="year=2019 and month=3 and day % 2 == 0")
            >>> collection.load(
            ...     filters=lambda keys: keys["year"] == 2019 and
            ...     keys["month"] == 3 and keys["day"] % 2 == 0)
        """
        client = utilities.get_client()
        arrays: List[dataset.Dataset] = []
        if indexer is None:
            selected_partitions = tuple(self.partitions(filters=filters))
            if len(selected_partitions) == 0:
                return None

            # No indexer, so the dataset is loaded directly for each
            # selected partition.
            bag = dask.bag.from_sequence(self.partitions(filters=filters),
                                         npartitions=utilities.dask_workers(
                                             client, cores_only=True))
            arrays = bag.map(storage.open_zarr_group,
                             fs=self.fs,
                             selected_variables=selected_variables).compute()
        else:
            # Build an indexer dictionary between the partition scheme and
            # indexer.
            indexers_map: Dict[Tuple[Tuple[str, int], ...], List[slice]] = {}
            _ = {
                indexers_map.setdefault(partition_scheme, []).append(indexer)
                for partition_scheme, indexer in indexer
            }
            # Filter the selected partitions
            selected_partitions = set(indexers_map) & set(
                (self.partitioning.parse(item)
                 for item in self.partitions(filters=filters)))
            if len(selected_partitions) == 0:
                return None

            # For each provided partition scheme, retrieves the corresponding
            # indexer.
            args = ((item, indexers_map[item])
                    for item in sorted(selected_partitions))
            bag = dask.bag.from_sequence(args,
                                         npartitions=utilities.dask_workers(
                                             client, cores_only=True))

            # Finally, load the selected partitions and apply the indexer.
            arrays = list(
                itertools.chain.from_iterable(
                    bag.map(
                        _load_and_apply_indexer,
                        fs=self.fs,
                        partition_handler=self.partitioning,
                        partition_properties=self.partition_properties,
                        selected_variables=selected_variables,
                    ).compute()))

        array = arrays.pop(0)
        if arrays:
            array = array.concat(arrays, self.partition_properties.dim)
        return array

    # pylint: disable=method-hidden
    def update(
        self,
        func: PartitionCallable,
        variable: str,
        /,
        *args,
        filters: Optional[str] = None,
        **kwargs,
    ) -> None:
        # pylint: disable=method-hidden
        """Update the selected partitions.

        Args:
            func: The function to apply on each partition.
            variable: The variable to update.
            *args: The positional arguments to pass to the function.
            filters: The expression used to filter the partitions to update.
            **kwargs: The keyword arguments to pass to the function.

        Example:
            >>> import dask.array
            >>> import zcollection
            >>> def ones(ds):
            ...     return ds.variables["var1"].values * 0 + 1
            >>> collection = zcollection.Collection("my_collection", mode="w")
            >>> collection.update(ones, "var2")
        """
        _LOGGER.info("Updating of the %r variable in the collection", variable)
        client = utilities.get_client()

        local_func = _wrap_update_func(func=func,
                                       fs=self.fs,
                                       variable=variable,
                                       *args,
                                       **kwargs)
        awaitables = client.map(local_func,
                                tuple(self.partitions(filters=filters)))
        storage.execute_transaction(client, self.synchronizer, awaitables)

    def _bag_from_partitions(self) -> dask.bag.Bag:
        """Return a dask bag from the partitions.

        Returns:
            The dask bag.
        """
        partitions = [*self.partitions()]
        return dask.bag.from_sequence(seq=partitions,
                                      npartitions=len(partitions))

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
        _LOGGER.info("Dropping of the %r variable in the collection", variable)
        if variable in self.partitioning.variables:
            raise ValueError(
                f"The variable {variable!r} is part of the partitioning.")
        if variable not in self.metadata.variables:
            raise ValueError(
                f"The variable {variable!r} does not exist in the collection.")
        bag = self._bag_from_partitions()
        bag.map(storage.del_zarr_array, variable, self.fs).compute()
        del self.metadata.variables[variable]
        self._write_config()

    def add_variable(self, variable: meta.Variable) -> None:
        """Add a variable to the collection.

        Args:
            variable: The variable to add.

        Raises:
            ValueError: if the variable is already part of the collection, it
                doesn’t use the partitioning dimension or use a dimension that
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
        _LOGGER.info("Adding of the %r variable in the collection",
                     variable.name)
        if self.partition_properties.dim not in variable.dimensions:
            raise ValueError(
                "The new variable must use the partitioning axis.")
        self.metadata.add_variable(variable)
        self._write_config()

        template = self.metadata.search_same_dimensions_as(variable)
        try:
            bag = self._bag_from_partitions()
            bag.map(storage.add_zarr_array, variable, template.name,
                    self.fs).compute()
        except Exception:
            self.drop_variable(variable.name)
            raise

    def iterate_on_records(
        self,
        *,
        relative: bool = False,
    ) -> Iterator[Tuple[str, zarr.Group]]:
        """Iterate over the partitions and the zarr groups.

        Args:
            relative: If True, the paths are relative to the base directory.

        Returns
            The iterator over the partitions and the zarr groups.
        """
        for item in self.partitions():
            yield (
                self._relative_path(item) if relative else item,
                zarr.open_consolidated(  # type: ignore
                    self.fs.get_mapper(item)))

    def variables(
        self,
        selected_variables: Optional[Iterable[str]] = None
    ) -> Tuple[dataset.Variable]:
        """Return the variables of the collection.

        Args:
            selected_variables: The variables to return. If None, all the
                variables are returned.

        Returns:
            The variables of the collection.
        """
        selected_variables = selected_variables or self.metadata.variables.keys(
        )
        return tuple(
            dataset.Variable(
                v.name, numpy.ndarray((0, ) * len(v.dimensions), v.dtype),
                v.dimensions, v.attrs, v.compressor, v.fill_value, v.filters)
            for k, v in self.metadata.variables.items()
            if k in selected_variables)


def create_collection(
    axis: str,
    ds: Union[xarray.Dataset, dataset.Dataset],
    partition_handler: partitioning.Partitioning,
    partition_base_dir: str,
    **kwargs,
) -> Collection:
    """Create a collection.

    Args:
        axis: The axis to use for the collection.
        ds: The dataset to use.
        partition_handler: The partitioning handler to use.
        partition_base_dir: The base directory to use for the partitions.
        **kwargs: Additional parameters are passed through to the constructor
            of the class :py:class:`Collection`.

    Example:
        >>> import xarray as xr
        >>> import zcollection
        >>> data = xr.Dataset({
        ...     "a": xr.DataArray([1, 2, 3]),
        ...     "b": xr.DataArray([4, 5, 6])
        ... })
        >>> collection = zcollection.create_collection(
        ...     "a", data,
        ...     zcollection.partitioning.Sequence(("a", )),
        ...     "/tmp/my_collection")

    Returns:
        The collection.
    """
    if isinstance(ds, xarray.Dataset):
        ds = dataset.Dataset.from_xarray(ds)
    return Collection(axis,
                      ds.metadata(),
                      partition_handler,
                      partition_base_dir,
                      mode="w",
                      **kwargs)


# pylint: disable=redefined-builtin
def open_collection(path: str,
                    *,
                    mode: Optional[str] = None,
                    **kwargs) -> Collection:
    """Open a collection.

    Args:
        path: The path to the collection.
        mode: The mode to open the collection.
        **kwargs: Additional parameters are passed through the method
            :py:meth:`zcollection.collection.Collection.from_config`.
    Returns:
        The collection.

    Example:
        >>> import zcollection
        >>> collection = zcollection.open_collection(
        ...     "/tmp/mycollection", mode="r")
    """
    return Collection.from_config(path, mode=mode, **kwargs)
    # pylint: enable=redefined-builtin
