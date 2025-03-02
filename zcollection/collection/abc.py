# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Base classes for collections.
=============================
"""
from __future__ import annotations

from typing import Any, ClassVar, Literal, Optional
from collections.abc import Callable, Iterable, Iterator, Sequence
import dataclasses
import importlib.metadata
import itertools
import pathlib

import dask.bag.core
import dask.distributed
import dask.utils
import fsspec

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
PartitionFilterCallback = Callable[[dict[str, int]], bool]

#: Type of argument to filter the partitions.
PartitionFilter = Optional[str | PartitionFilterCallback]

#: Indexer's type.
Indexer = Iterable[tuple[tuple[tuple[str, int], ...], slice]]

#: Indexer arguments' type.
IndexerArgs = tuple[tuple[tuple[str, int], ...], list[slice]]

#: Name of the directory storing the immutable dataset.
IMMUTABLE = '.immutable'


def list_partitions_from_indexer(
    indexer: Indexer,
    partition_handler: partitioning.Partitioning,
    base_dir: str,
    sep: str,
) -> Iterator[str]:
    """List the partitions from the indexer.

    Args:
        indexer: The indexer.
        partition_handler: The partitioning strategy.
        base_dir: The base directory.
        sep: The separator.

    Returns:
        The list of partitions.
    """
    keys = sorted({key for key, _ in indexer})
    yield from (sep.join((base_dir, partition_handler.join(key, sep)))
                for key in keys)


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
        partition_properties.dir, IMMUTABLE) if zds.select_variables_by_dims(
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

    @property
    def dimension(self) -> str:
        """The name of the partitioning dimension."""
        return self.partition.dim


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

        self._settings = CollectionSettings(
            mode=mode or 'w',
            filesystem=fs_utils.get_fs(filesystem),
            synchronizer=synchronizer or sync.NoSync())

        self._properties = CollectionProperties(
            axis=axis,
            metadata=ds,
            partition_strategy=partition_handler,
            partition=PartitioningProperties(
                dir=fs_utils.normalize_path(fs=self._settings.filesystem,
                                            path=partition_base_dir),
                dim=ds.variables[axis].dimensions[0]))

        #: The path to the dataset that contains the immutable data relative
        #: to the partitioning.
        self._immutable: str | None = _immutable_path(
            zds=ds, partition_properties=self._properties.partition)

        self.version = importlib.metadata.version('zcollection')

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

    def _normalize_partitions(self,
                              partitions: Iterable[str]) -> Iterable[str]:
        """Normalize the provided list of partitions to include the full
        partition's path.

        Args:
            partitions: The list of partitions to normalize.

        Returns:
            The list of partitions.
        """
        return filter(
            self.fs.exists,
            map(
                lambda partition: self.fs.sep.join(
                    (self.partition_properties.dir, partition)),
                sorted(set(partitions))))

    def dimensions_properties(self) -> tuple[dict[str, int], dict[str, int]]:
        """Extract dimension properties (size and chunks).

        Returns:
            A tuple of dictionaries containing the dimensions associated
            to their size and the dimensions associated to their chunks.
        """
        chunks: dict[str, int] = {}
        dimensions: dict[str, int] = {}

        for dim in self.metadata.dimensions.values():
            chunks[dim.name] = dim.chunks

            if dim.name != self._properties.dimension:
                dimensions[dim.name] = dim.value

        return dimensions, chunks

    def partitions(
        self,
        *,
        filters: PartitionFilter = None,
        indexer: Indexer | None = None,
        selected_partitions: Iterable[str] | None = None,
        relative: bool = False,
        lock: bool = False,
    ) -> Iterator[str]:
        """List the partitions of the collection.

        Args:
            filters: The predicate used to filter the partitions to load. If
                the predicate is a string, it is a valid python expression to
                filter the partitions, using the partitioning scheme as
                variables. If the predicate is a function, it is a function that
                takes the partition scheme as input and returns a boolean.
            indexer: The indexer to apply.
            selected_partitions: A list of partitions to load (using the
                partition relative path).
            relative: Whether to return the relative path.
            lock: Whether to lock the collection or not to avoid listing
                partitions while the collection is being modified.

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
        if selected_partitions is not None:
            partitions: Iterable[str] = self._normalize_partitions(
                partitions=selected_partitions)
        else:
            if lock:
                with self.synchronizer:
                    partitions = tuple(
                        self.partitioning.list_partitions(self.fs, base_dir))
            else:
                partitions = self.partitioning.list_partitions(
                    self.fs, base_dir)

        if indexer is not None:
            # List of partitions existing in the indexer and partitions list
            partitions = list(partitions)
            partitions = [
                p for p in list_partitions_from_indexer(
                    indexer=indexer,
                    partition_handler=self.partitioning,
                    base_dir=self.partition_properties.dir,
                    sep=self.fs.sep) if p in partitions
            ]

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
        delayed: bool = True,
        filters: PartitionFilter = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Sequence[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Map a function over the partitions of the collection.

        Args:
            func: The function to apply to every partition of the collection.
            *args: The positional arguments to pass to the function.
            delayed: Whether to load the data lazily or not.
            filters: The predicate used to filter the partitions to process.
                To get more information on the predicate, see the
                documentation of the :meth:`partitions` method.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
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
            _partition: str,
            _func: PartitionCallable,
            _selected_variables: Sequence[str] | None,
            _delayed: bool,
            *_args,
            **_kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                _func: The function to apply.
                _partition: The partition to apply the function on.
                _selected_variables: The list of variables to retain from the
                    partition.
                *_args: The positional arguments to pass to the function.
                **_kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            zds: dataset.Dataset = _load_dataset(
                delayed=_delayed,
                fs=self.fs,
                immutable=self._immutable,
                partition=_partition,
                selected_variables=_selected_variables)
            return self.partitioning.parse(_partition), _func(
                zds, *_args, **_kwargs)

        if not callable(func):
            raise TypeError('func must be a callable')

        bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
            self.partitions(filters=filters),
            partition_size=partition_size,
            npartitions=npartitions)

        return bag.map(_wrap,
                       _func=func,
                       _selected_variables=selected_variables,
                       _delayed=delayed,
                       *args,
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
            _partition: str,
            _delayed: bool,
            _depth: int,
            _partitions: tuple[str, ...],
            _selected_variables: Sequence[str] | None,
            _wrapped_func: PartitionCallable,
            *_args,
            **_kwargs,
        ) -> tuple[tuple[tuple[str, int], ...], Any]:
            """Wraps the function to apply on the partition.

            Args:
                *_args: The positional arguments to pass to the function.
                _delayed: Whether to load the data lazily or not.
                _depth: The depth of the overlap between the partitions.
                _partition: The partition to apply the function on.
                _partitions: The partitions to apply the function on.
                _selected_variables: The list of variables to retain from the
                    partition.
                _wrapped_func: The function to apply.
                **_kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function.
            """
            zds: dataset.Dataset
            indices: slice

            zds, indices = _load_dataset_with_overlap(
                delayed=_delayed,
                depth=_depth,
                dim=self.partition_properties.dim,
                fs=self.fs,
                immutable=self._immutable,
                partition=_partition,
                partitions=_partitions,
                selected_variables=_selected_variables)

            if add_partition_info:
                _kwargs = _kwargs.copy()
                _kwargs['partition_info'] = (self.partition_properties.dim,
                                             indices)

            # Finally, apply the function.
            return (self.partitioning.parse(_partition),
                    _wrapped_func(zds, *_args, **_kwargs))

        partitions = tuple(self.partitions(filters=filters))
        bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
            partitions, partition_size=partition_size, npartitions=npartition)

        return bag.map(_wrap,
                       _delayed=delayed,
                       _depth=depth,
                       _partitions=partitions,
                       _selected_variables=selected_variables,
                       _wrapped_func=func,
                       *args,
                       **kwargs)

    def load(
        self,
        *,
        delayed: bool = True,
        filters: PartitionFilter = None,
        indexer: Indexer | None = None,
        selected_variables: Iterable[str] | None = None,
        selected_partitions: Iterable[str] | None = None,
        distributed: bool = True,
    ) -> dataset.Dataset | None:
        """Load collection's data, respecting filters, indexer, and selected
        partitions constraints.

        Args:
            delayed: Whether to load data in a dask array or not.
            filters: The predicate used to filter the partitions to load. To
                get more information on the predicate, see the documentation of
                the :meth:`partitions` method.
            indexer: The indexer to apply.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
            selected_partitions: A list of partitions to load (using the
                partition relative path).
            distributed: Whether to use dask or not. Default To True.

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
        # Delayed has to be True if dask is disabled
        if not distributed:
            delayed = False

        if indexer is None:
            arrays = self._load_partitions(
                delayed=delayed,
                filters=filters,
                selected_variables=selected_variables,
                selected_partitions=selected_partitions,
                distributed=distributed)
        else:
            arrays = self._load_partitions_indexer(
                indexer=indexer,
                delayed=delayed,
                filters=filters,
                selected_variables=selected_variables,
                selected_partitions=selected_partitions,
                distributed=distributed)

        if arrays is None:
            return None

        array: dataset.Dataset = arrays.pop(0)
        if arrays:
            array = array.concat(arrays, self.partition_properties.dim)
        if self._immutable:
            array.merge(
                storage.open_zarr_group(dirname=self._immutable,
                                        fs=self.fs,
                                        delayed=delayed,
                                        selected_variables=selected_variables))
        array.fill_attrs(self.metadata)
        return array

    def _load_partitions(
        self,
        *,
        delayed: bool = True,
        filters: PartitionFilter = None,
        selected_variables: Iterable[str] | None = None,
        selected_partitions: Iterable[str] | None = None,
        distributed: bool = True,
    ) -> list[dataset.Dataset] | None:
        """Load collection's partitions, respecting filters, and selected
        partitions constraints.

        Args:
            delayed: Whether to load data in a dask array or not.
            filters: The predicate used to filter the partitions to load. To
                get more information on the predicate, see the documentation of
                the :meth:`partitions` method.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
            selected_partitions: A list of partitions to load (using the
                partition relative path).
            distributed: Whether to use dask or not. Default To True.

        Returns:
            The list of dataset for each partition, or None if no
            partitions were selected.
        """
        # No indexer, so the dataset is loaded directly for each
        # selected partition.
        selected_partitions = tuple(
            self.partitions(filters=filters,
                            selected_partitions=selected_partitions))

        if len(selected_partitions) == 0:
            return None

        if distributed:
            client = dask_utils.get_client()
            bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
                selected_partitions,
                npartitions=dask_utils.dask_workers(client, cores_only=True))
            arrays = bag.map(storage.open_zarr_group,
                             delayed=delayed,
                             fs=self.fs,
                             selected_variables=selected_variables).compute()
        else:
            arrays = [
                storage.open_zarr_group(dirname=partition,
                                        delayed=delayed,
                                        fs=self.fs,
                                        selected_variables=selected_variables)
                for partition in selected_partitions
            ]

        return arrays

    def _load_partitions_indexer(
        self,
        *,
        indexer: Indexer,
        delayed: bool = True,
        filters: PartitionFilter = None,
        selected_variables: Iterable[str] | None = None,
        selected_partitions: Iterable[str] | None = None,
        distributed: bool = True,
    ) -> list[dataset.Dataset] | None:
        """Load collection's partitions, respecting filters, indexer, and
        selected partitions constraints.

        Args:
            indexer: The indexer to apply.
            delayed: Whether to load data in a dask array or not.
            filters: The predicate used to filter the partitions to load. To
                get more information on the predicate, see the documentation of
                the :meth:`partitions` method.
            selected_variables: A list of variables to retain from the
                collection. If None, all variables are kept.
            selected_partitions: A list of partitions to load (using the
                partition relative path).
            distributed: Whether to use dask or not. Default To True.

        Returns:
            The list of dataset for each partition, or None if no
            partitions were selected.
        """
        # We're going to reuse the indexer variable, so ensure it is
        # an iterable not a generator.
        indexer = tuple(indexer)

        # Build the indexer arguments.
        partitions = self.partitions(selected_partitions=selected_partitions,
                                     filters=filters,
                                     indexer=indexer)
        args = tuple(
            build_indexer_args(collection=self,
                               filters=filters,
                               indexer=indexer,
                               partitions=partitions))
        if len(args) == 0:
            return None

        # Finally, load the selected partitions and apply the indexer.
        if distributed:
            client = dask_utils.get_client()
            bag = dask.bag.core.from_sequence(
                args,
                npartitions=dask_utils.dask_workers(client, cores_only=True))

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
        else:
            arrays = list(
                itertools.chain.from_iterable([
                    _load_and_apply_indexer(
                        args=a,
                        delayed=delayed,
                        fs=self.fs,
                        partition_handler=self.partitioning,
                        partition_properties=self.partition_properties,
                        selected_variables=selected_variables) for a in args
                ]))

        return arrays

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

    def iterate_on_records(self) -> Iterator[tuple[str, str]]:
        """Iterate over the relative and absolute partitions' path.

        Returns     The iterator over the relative and absolute
        partitions' path.
        """
        yield from ((self._relative_path(item), item)
                    for item in self.partitions())

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
