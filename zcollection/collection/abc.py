# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Base classes for collections.
=============================
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
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import dataclasses
import itertools
import pathlib

import dask.bag.core
import dask.distributed
import dask.utils
import fsspec
import zarr

from .. import (
    dask_utils,
    dataset,
    expression,
    fs_utils,
    meta,
    partitioning,
    storage,
    sync,
)
from .callable_objects import MapCallable, PartitionCallable
from .detail import (
    PartitioningProperties,
    _load_and_apply_indexer,
    _load_dataset,
    _load_dataset_with_overlap,
)

#: Type of functions filtering the partitions.
PartitionFilterCallback = Callable[[Dict[str, int]], bool]

#: Type of argument to filter the partitions.
PartitionFilter = Optional[Union[str, PartitionFilterCallback]]

#: Indexer's type.
Indexer = Iterable[Tuple[Tuple[Tuple[str, int], ...], slice]]

#: Indexer arguments' type.
IndexerArgs = Tuple[Tuple[Tuple[str, int], ...], List[slice]]

#: Name of the directory storing the immutable dataset.
_IMMUTABLE = '.immutable'


def build_indexer_args(
    collection: ReadOnlyCollection,
    filters: PartitionFilter,
    indexer: Indexer,
    *,
    partitions: Iterable[str] | None = None,
) -> Iterator[IndexerArgs]:
    """Build the arguments for the indexer.

    Args:
        collection: The collection to index.
        filters: The partition filters.
        indexer: The indexer.
        partitions: The partitions to index. If None, all the partitions
            are indexed.

    Returns:
        An iterator containing the arguments for the indexer.
    """
    # Build an indexer dictionary between the partition scheme and
    # indexer.
    indexers_map: dict[tuple[tuple[str, int], ...], list[slice]] = {}
    _ = {
        indexers_map.setdefault(  # type: ignore[func-returns-value]
            partition_scheme, []).append(indexer)
        for partition_scheme, indexer in indexer
    }
    # Filter the selected partitions
    partitions = partitions or collection.partitions(filters=filters)
    selected_partitions = set(indexers_map) & {
        collection.partitioning.parse(item)
        for item in partitions
    }

    # For each provided partition scheme, retrieves the corresponding
    # indexer.
    return ((item, indexers_map[item]) for item in sorted(selected_partitions))


def _immutable_path(
    zds: meta.Dataset,
    partition_properties: PartitioningProperties,
) -> str | None:
    """Return the immutable path of the dataset.

    Args:
        zds: The dataset to process.
        partition_properties: The partitioning properties.

    Returns:
        The immutable path of the dataset containing data that are immutable
        relative to the partitioning or None if the dataset does not contain
        immutable data.
    """
    return fs_utils.join_path(
        partition_properties.dir, _IMMUTABLE) if zds.select_variables_by_dims(
            (partition_properties.dim, ), predicate=False) else None


@dataclasses.dataclass(frozen=True)
class CollectionProperties:
    """This class contains the properties of a collection."""
    #: The axis of the collection.
    axis: str

    #: The metadata that describes the dataset handled by the collection.
    metadata: meta.Dataset

    #: The partitioning strategy used to split the data.
    partition_strategy: partitioning.Partitioning

    #: The partitioning properties (base directory and dimension).
    partition: PartitioningProperties


@dataclasses.dataclass(frozen=True)
class CollectionSettings:
    """This class contains the settings of a collection."""
    #: The mode of access of the collection.
    mode: Literal['r', 'w']

    #: The file system used to read/write the collection.
    filesystem: fsspec.AbstractFileSystem

    #: The synchronizer used to synchronize the modifications.
    synchronizer: sync.Sync

    def __post_init__(self) -> None:
        if self.mode not in ('r', 'w'):
            raise ValueError(f'The mode {self.mode!r} is not supported.')


class ReadOnlyCollection:
    """Collection base class, offering read-only access to a collection of
    datasets.

    The arguments of the constructor are detailed in the documentation
    of the parent class :py:class:`zarr_collection.Collection`.
    """
    #: Configuration filename of the collection.
    CONFIG: ClassVar[str] = '.zcollection'

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
        if axis not in ds.variables:
            raise ValueError(
                f'The variable {axis!r} is not defined in the dataset.')

        for varname in partition_handler.variables:
            if varname not in ds.variables:
                raise ValueError(
                    f'The partitioning key {varname!r} is not defined in '
                    'the dataset.')

        self._settings = CollectionSettings(mode or 'w',
                                            fs_utils.get_fs(filesystem),
                                            synchronizer or sync.NoSync())

        self._properties = CollectionProperties(
            axis, ds, partition_handler,
            PartitioningProperties(
                fs_utils.normalize_path(self._settings.filesystem,
                                        partition_base_dir),
                ds.variables[axis].dimensions[0]))

        #: The path to the dataset that contains the immutable data relative
        #: to the partitioning.
        self._immutable: str | None = _immutable_path(
            ds, self._properties.partition)

    @property
    def axis(self) -> str:
        """Return the axis of the collection."""
        return self._properties.axis

    @property
    def metadata(self) -> meta.Dataset:
        """Return the metadata of the collection."""
        return self._properties.metadata

    @property
    def partitioning(self) -> partitioning.Partitioning:
        """Return the partitioning strategy of the collection."""
        return self._properties.partition_strategy

    @property
    def partition_properties(self) -> PartitioningProperties:
        """Return the partitioning properties of the collection."""
        return self._properties.partition

    @property
    def mode(self) -> Literal['r', 'w']:
        """Return the mode of the collection."""
        return self._settings.mode

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """Return the filesystem of the collection."""
        return self._settings.filesystem

    @property
    def synchronizer(self) -> sync.Sync:
        """Return the synchronizer of the collection."""
        return self._settings.synchronizer

    @property
    def immutable(self) -> bool:
        """Return True if the collection contains immutable data relative to
        the partitioning."""
        return self._immutable is not None

    @classmethod
    def _config(cls, partition_base_dir: str) -> str:
        """Return the configuration path."""
        return fs_utils.join_path(partition_base_dir, cls.CONFIG)

    def is_locked(self) -> bool:
        """Return True if the collection is locked."""
        return self.synchronizer.is_locked()

    def _is_selected(
        self,
        partition: Sequence[str],
        expr: Callable[[dict[str, int]], bool] | None,
    ) -> bool:
        """Return whether the partition is selected.

        Args:
            partition: The partition to check.
            expr: The expression used to filter the partition.

        Returns:
            Whether the partition is selected.
        """
        return True if expr is None else expr(
            dict(self.partitioning.parse('/'.join(partition))))

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
        lock: bool = False,
        filters: PartitionFilter = None,
        relative: bool = False,
    ) -> Iterator[str]:
        """List the partitions of the collection.

        Args:
            lock: Whether to lock the collection or not to avoid listing
                partitions while the collection is being modified.
            filters: The predicate used to filter the partitions to load. If
                the predicate is a string, it is a valid python expression to
                filter the partitions, using the partitioning scheme as
                variables. If the predicate is a function, it is a function that
                takes the partition scheme as input and returns a boolean.
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
            expr: Any = expression.Expression(filters)
        elif callable(filters):
            expr = filters
        else:
            expr = None

        base_dir: str = self.partition_properties.dir
        sep: str = self.fs.sep

        if lock:
            with self.synchronizer:
                partitions: Iterable[str] = tuple(
                    self.partitioning.list_partitions(self.fs, base_dir))
        else:
            partitions = self.partitioning.list_partitions(self.fs, base_dir)

        yield from (self._relative_path(item) if relative else item
                    for item in partitions
                    if (item != self._immutable and self._is_selected(
                        item.replace(base_dir, '').split(sep), expr)))

    # pylint: disable=duplicate-code
    # false positive, no code duplication
    def map(
        self,
        func: MapCallable,
        /,
        *args,
        filters: PartitionFilter = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Sequence[str] | None = None,
        delayed: bool = True,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the collection.

        Args:
            func: The function to apply to every partition of the collection.
            *args: The positional arguments to pass to the function.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
            delayed: Whether to load the data lazily or not.
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
            selected_variables: Sequence[str] | None,
            delayed: bool,
            *args,
            **kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                func: The function to apply.
                partition: The partition to apply the function on.
                selected_variables: The list of variables to retain from the
                    partition.
                *args: The positional arguments to pass to the function.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            zds: dataset.Dataset = _load_dataset(delayed, self.fs,
                                                 self._immutable, partition,
                                                 selected_variables)
            return self.partitioning.parse(partition), func(
                zds, *args, **kwargs)

        if not callable(func):
            raise TypeError('func must be a callable')

        bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
            self.partitions(filters=filters),
            partition_size=partition_size,
            npartitions=npartitions)
        return bag.map(_wrap, func, selected_variables, delayed, *args,
                       **kwargs)
        # pylint: enable=duplicate-code

    def map_overlap(
        self,
        func: MapCallable,
        /,
        *args,
        delayed: bool = True,
        depth: int = 1,
        filters: PartitionFilter = None,
        partition_size: int | None = None,
        npartition: int | None = None,
        selected_variables: Sequence[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the collection with some
        overlap.

        Args:
            func: The function to apply to every partition of the collection.
                If ``func`` accepts a partition_info as a keyword
                argument, it will be passed a tuple with the name of the
                partitioned dimension and the slice allowing getting in the
                dataset the selected partition without the overlap.
            *args: The positional arguments to pass to the function.
            delayed: Whether to load the data lazily or not.
            depth: The depth of the overlap between the partitions. Defaults
                to 1.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            partition_size: The length of each bag partition.
            npartition: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A bag containing the tuple of the partition scheme and the result
            of the function.

        Example:
            >>> futures = collection.map_overlap(
            ...     lambda x: (x["var1"] + x["var2"]).values,
            ...     depth=1)
            >>> for item in futures:
            ...     print(item)
            [1.0, 2.0, 3.0, 4.0]
            [5.0, 6.0, 7.0, 8.0]
        """
        if not callable(func):
            raise TypeError('func must be a callable')

        add_partition_info: bool = dask.utils.has_keyword(
            func, 'partition_info')

        def _wrap(
            partition: str,
            *args,
            delayed: bool,
            depth: int,
            partitions: tuple[str, ...],
            selected_variables: Sequence[str] | None,
            wrapped_func: PartitionCallable,
            **kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                *args: The positional arguments to pass to the function.
                delayed: Whether to load the data lazily or not.
                depth: The depth of the overlap between the partitions.
                partition: The partition to apply the function on.
                partitions: The partitions to apply the function on.
                selected_variables: The list of variables to retain from the
                    partition.
                wrapped_func: The function to apply.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            zds: dataset.Dataset
            indices: slice

            zds, indices = _load_dataset_with_overlap(
                delayed=delayed,
                depth=depth,
                dim=self.partition_properties.dim,
                fs=self.fs,
                immutable=self._immutable,
                partition=partition,
                partitions=partitions,
                selected_variables=selected_variables)

            if add_partition_info:
                kwargs = kwargs.copy()
                kwargs['partition_info'] = (self.partition_properties.dim,
                                            indices)

            # Finally, apply the function.
            return (self.partitioning.parse(partition),
                    wrapped_func(zds, *args, **kwargs))

        partitions = tuple(self.partitions(filters=filters))
        bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
            partitions, partition_size=partition_size, npartitions=npartition)
        return bag.map(_wrap,
                       *args,
                       delayed=delayed,
                       depth=depth,
                       partitions=partitions,
                       selected_variables=selected_variables,
                       wrapped_func=func,
                       **kwargs)

    def load(
        self,
        *,
        delayed: bool = True,
        filters: PartitionFilter = None,
        indexer: Indexer | None = None,
        selected_variables: Iterable[str] | None = None,
    ) -> dataset.Dataset | None:
        """Load the selected partitions.

        Args:
            delayed: Whether to load data in a dask array or not.
            filters: The predicate used to filter the partitions to load. To
                get more information on the predicate, see the documentation of
                the :meth:`partitions` method.
            indexer: The indexer to apply.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.

        Returns:
            The dataset containing the selected partitions, or None if no
            partitions were selected.

        Warning:
            If you select variables to load from the collection, do not insert
            the returned dataset otherwise all skipped variables will be reset
            with fill values.

        Example:
            >>> collection = ...
            >>> collection.load(
            ...     filters="year == 2019 and month == 3 and day % 2 == 0")
            >>> collection.load(
            ...     filters=lambda keys: keys["year"] == 2019 and
            ...     keys["month"] == 3 and keys["day"] % 2 == 0)
        """
        client: dask.distributed.Client = dask_utils.get_client()
        arrays: list[dataset.Dataset]
        if indexer is None:
            selected_partitions = tuple(self.partitions(filters=filters))
            if len(selected_partitions) == 0:
                return None

            # No indexer, so the dataset is loaded directly for each
            # selected partition.
            bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
                self.partitions(filters=filters),
                npartitions=dask_utils.dask_workers(client, cores_only=True))
            arrays = bag.map(storage.open_zarr_group,
                             delayed=delayed,
                             fs=self.fs,
                             selected_variables=selected_variables).compute()
        else:
            # Build the indexer arguments.
            args = tuple(build_indexer_args(self, filters, indexer))
            if len(args) == 0:
                return None

            bag = dask.bag.core.from_sequence(
                args,
                npartitions=dask_utils.dask_workers(client, cores_only=True))

            # Finally, load the selected partitions and apply the indexer.
            arrays = list(
                itertools.chain.from_iterable(
                    bag.map(
                        _load_and_apply_indexer,
                        delayed=delayed,
                        fs=self.fs,
                        partition_handler=self.partitioning,
                        partition_properties=self.partition_properties,
                        selected_variables=selected_variables,
                    ).compute()))

        array: dataset.Dataset = arrays.pop(0)
        if arrays:
            array = array.concat(arrays, self.partition_properties.dim)
        if self._immutable:
            array.merge(
                storage.open_zarr_group(self._immutable,
                                        self.fs,
                                        delayed=delayed,
                                        selected_variables=selected_variables))
        array.fill_attrs(self.metadata)
        return array

    def _bag_from_partitions(
        self,
        filters: PartitionFilter | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Return a dask bag from the partitions.

        Args:
            filters: The predicate used to filter the partitions to load.
            kwargs: The keyword arguments to pass to the method
                :meth:`partitions`.

        Returns:
            The dask bag.
        """
        partitions: list[str] = [*self.partitions(filters=filters, **kwargs)]
        return dask.bag.core.from_sequence(seq=partitions,
                                           npartitions=len(partitions))

    def iterate_on_records(
        self,
        *,
        relative: bool = False,
    ) -> Iterator[tuple[str, zarr.Group]]:
        """Iterate over the partitions and the zarr groups.

        Args:
            relative: If True, the paths are relative to the base directory.

        Returns
            The iterator over the partitions and the zarr groups.
        """
        yield from (
            (
                self._relative_path(item) if relative else item,
                zarr.open_consolidated(
                    self.fs.get_mapper(item),  # type: ignore
                    mode='r',
                )) for item in self.partitions())

    def variables(
        self,
        selected_variables: Iterable[str] | None = None
    ) -> tuple[dataset.Variable, ...]:
        """Return the variables of the collection.

        Args:
            selected_variables: The variables to return. If None, all the
                variables are returned.

        Returns:
            The variables of the collection.
        """
        return dataset.get_dataset_variable_properties(self.metadata,
                                                       selected_variables)
