# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Base classes for collections.
=============================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, Optional
import abc
from collections.abc import Callable, Iterable, Iterator, Sequence
import contextlib
import dataclasses
import functools
import io
import itertools
import logging
import pathlib
import types

import dask.bag.core
import dask.distributed
import numpy

from .. import (
    dask_utils,
    dataset,
    execution,
    expression,
    fs_utils,
    meta,
    partitioning,
    storage,
    store,
    sync,
)
from ..partitions import PartitionIndex
from ..variable.abc import not_equal
from .detail import (
    PartitioningProperties,
    _load_and_apply_indexer,
    _load_dataset,
    _load_dataset_with_overlap,
    map_partitions,
    note_failed_removal,
    prepare_partition_operation,
    update_partition_from,
    warn_empty_update,
)

if TYPE_CHECKING:
    import fsspec

    from .callable_objects import (
        MapCallable,
        UpdateCallable,
        WrappedPartitionCallable,
    )


def warn_missing_partitions(paths: Iterable[str]) -> None:
    """Log a warning for paths in ``selected_partitions`` that do not
    correspond to existing partitions.

    Args:
        paths: The paths excluded from the selection.
    """
    _LOGGER.warning(
        'The selected paths %s name no partition of the collection or the '
        'view and are left out.', ', '.join(repr(path) for path in paths))


#: Callable type for partition filters.
PartitionFilterCallback = Callable[[dict[str, int]], bool]

#: Type accepted for partition filters.
PartitionFilter = Optional[str | PartitionFilterCallback]

#: Indexer type.
Indexer = Iterable[tuple[tuple[tuple[str, int], ...], slice]]

#: Type of indexer arguments.
IndexerArgs = tuple[tuple[tuple[str, int], ...], list[slice]]

#: Name of the directory storing the immutable dataset.
IMMUTABLE = '.immutable'

#: Module logger.
_LOGGER = logging.getLogger(__name__)


def list_partitions_from_indexer(
    indexer: Indexer,
    partition_handler: partitioning.Partitioning,
    base_dir: str,
    sep: str,
) -> Iterator[str]:
    """List partitions selected by the indexer.

    Args:
        indexer: The indexer.
        partition_handler: The partitioning strategy.
        base_dir: The base directory.
        sep: The separator.

    Returns:
        The selected partitions.
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
    """Prepare arguments for the indexer.

    Args:
        collection: The collection to index.
        filters: The partition filters.
        indexer: The indexer.
        partitions: The partitions to index. If None, all the partitions
            are indexed.

    Returns:
        An iterator of indexer arguments.
    """
    # Map partitioning keys to the indexer.
    indexers_map: dict[tuple[tuple[str, int], ...], list[slice]] = {}
    _ = {
        indexers_map.setdefault(  # type: ignore[func-returns-value]
            partition_scheme, []).append(indexer)
        for partition_scheme, indexer in indexer
    }
    # Apply the partition filters.
    partitions = partitions or collection.partitions(filters=filters)
    selected_partitions = set(indexers_map) & {
        collection.partitioning.parse(item)
        for item in partitions
    }

    # Retrieve the matching indexer for each partitioning scheme.
    return ((item, indexers_map[item]) for item in sorted(selected_partitions))


@storage.skip_absent_partition
def _open_held_group(
    partition: str,
    fs: fsspec.AbstractFileSystem,
    *,
    delayed: bool,
    selected_variables: Iterable[str] | None,
) -> dataset.Dataset:
    """Read a partition.

    Args:
        partition: The path of the partition.
        fs: The file system storing the partition.
        delayed: Whether to load the data lazily.
        selected_variables: The variables to retain.

    Returns:
        The partition dataset, or ``None`` if the partition is absent
        from storage.
    """
    return storage.read_partition(partition,
                                  fs,
                                  delayed=delayed,
                                  selected_variables=selected_variables)


def _immutable_path(partition_properties: PartitioningProperties, ) -> str:
    """Return the path of the immutable variable group.

    Args:
        partition_properties: The partitioning properties.

    Returns:
        The path to the ``.immutable`` group.
    """
    return fs_utils.join_path(partition_properties.dir, IMMUTABLE)


def _check_sizes(ds: meta.DatasetSchema, main_dimension: str) -> None:
    """Reject schemas with unspecified sizes outside the partitioned dimension.

    The check runs only when opening a collection for writing. Older
    collections may record no dimension sizes, so opening them read-only
    skips this validation.

    Args:
        ds: The collection schema.
        main_dimension: The partitioned dimension.

    Raises:
        ValueError: If a dimension other than the partitioned dimension
            has no declared size.
    """
    for dimension in ds.dimensions.values():
        if dimension.name != main_dimension and dimension.value < 0:
            raise ValueError(
                f'The dimension {dimension.name!r} is declared without a '
                f'size. Only {main_dimension!r}, the dimension the '
                'collection is partitioned along, grows as measurements '
                'arrive; every other size holds for the life of the '
                'collection and an insertion is measured against it.')


def _check_fill_values(ds: meta.DatasetSchema) -> None:
    """Reject schemas whose filters change a variable's fill value.

    Only variables with both a fill value and filters are checked. The fill
    value must remain unchanged after encoding and decoding.

    Args:
        ds: The collection schema.

    Raises:
        ValueError: If a variable's fill value changes after passing
            through its filters.
    """
    for item in ds.variables.values():
        if item.fill_value is None or not item.filters:
            continue
        values = numpy.full((1, ) * len(item.dimensions),
                            item.fill_value,
                            dtype=item.dtype)
        decoded = store.roundtrip_filters(values, item.filters)
        read: Any = numpy.asarray(decoded, dtype=item.dtype).reshape(-1)[0]
        if not_equal(read, item.fill_value):
            raise ValueError(
                f'The variable {item.name!r} marks an absence with '
                f'{item.fill_value!s}, which its filters read back as '
                f'{read!s}. The value a variable marks an absence with is '
                'the one its filters decode to.')


@dataclasses.dataclass(frozen=True)
class CollectionProperties:
    """Properties of a collection instance."""
    #: The name of the axis variable.
    axis: str

    #: The collection schema.
    metadata: meta.DatasetSchema

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
    """Settings of a collection instance."""
    #: The collection access mode.
    mode: Literal['r', 'w']

    #: The file system used to read/write the collection.
    filesystem: fsspec.AbstractFileSystem

    #: The synchronizer used for collection writes.
    synchronizer: sync.Sync

    def __post_init__(self) -> None:
        if self.mode not in ('r', 'w'):
            raise ValueError(f'The mode {self.mode!r} is not supported.')


class BaseCollection(abc.ABC):
    """Share partition mappings and access to collection resources."""

    @property
    @abc.abstractmethod
    def fs(self) -> fsspec.AbstractFileSystem:
        """The file system storing the partitions."""
        ...

    @property
    @abc.abstractmethod
    def synchronizer(self) -> sync.Sync:
        """The synchronizer used by collection operations."""
        ...

    @property
    @abc.abstractmethod
    def metadata(self) -> meta.DatasetSchema:
        """The schema of the variables owned by this collection."""
        ...

    @property
    @abc.abstractmethod
    def dimension(self) -> str:
        """The partitioned dimension."""
        ...

    @property
    @abc.abstractmethod
    def partitioning(self) -> partitioning.Partitioning:
        """The partitioning strategy."""
        ...

    @abc.abstractmethod
    def partitions(
        self,
        *,
        filters: PartitionFilter = None,
        indexer: Indexer | None = None,
        selected_partitions: Iterable[str] | None = None,
    ) -> Iterator[str]:
        """List the partition paths selected for this collection."""
        ...

    @abc.abstractmethod
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
        ...

    @property
    @abc.abstractmethod
    def _partitions_dir(self) -> str:
        """The base directory containing the collection's partitions."""
        ...

    def _relative_path(self, path: str) -> str:
        """Return a partition path relative to the collection's base
        directory."""
        return pathlib.PurePosixPath(path).relative_to(
            self._partitions_dir).as_posix()

    def _overlap_pool(self, partitions: Iterable[str]) -> tuple[str, ...]:
        """Return the partitions among which the overlap windows are formed.

        The pool includes all partitions of the collection or view,
        including those not selected for processing.

        Args:
            partitions: The partitions selected for processing.
        """
        return tuple(
            sorted(set(self.partitions()) | set(partitions),
                   key=lambda path: self.partitioning.parse(
                       self._relative_path(path))))

    def is_locked(self) -> bool:
        """Return whether this instance holds the collection lock."""
        return self.synchronizer.is_locked()

    def variables(
        self,
        selected_variables: Iterable[str] | None = None
    ) -> tuple[dataset.Variable, ...]:
        """Return the variables of the collection.

        Args:
            selected_variables: Variables to return. If ``None``, return all
                variables.

        Returns:
            The collection variables.
        """
        return dataset.get_dataset_variable_properties(self.metadata,
                                                       selected_variables)

    def _map(
        self,
        func: MapCallable,
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
        *,
        delayed: bool,
        depth: int,
        filters: PartitionFilter,
        selected_partitions: Iterable[str] | None,
        partition_size: int | None,
        npartitions: int | None,
        selected_variables: Iterable[str] | None,
    ) -> dask.bag.core.Bag:
        """Prepare the partition loaders and build the mapping bag."""
        selected_variables = prepare_partition_operation(
            func, depth, selected_variables)
        partitions = tuple(
            self.partitions(filters=filters,
                            selected_partitions=selected_partitions))
        load, load_overlap = self._prepare_map(
            self._overlap_pool(partitions) if depth else partitions,
            delayed=delayed,
            depth=depth,
            selected_variables=selected_variables)
        return map_partitions(partitions,
                              func,
                              load,
                              load_overlap=load_overlap,
                              partitioning=self.partitioning,
                              dim=self.dimension,
                              partition_size=partition_size,
                              npartitions=npartitions,
                              func_args=func_args,
                              func_kwargs=func_kwargs)

    def map(
        self,
        func: MapCallable,
        /,
        *args,
        delayed: bool = True,
        filters: PartitionFilter = None,
        selected_partitions: Iterable[str] | None = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Iterable[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Apply a function to each selected partition.

        Args:
            func: The function to apply to each partition.
            *args: The positional arguments to pass to the function.
            delayed: Whether to load variables lazily.
            filters: Predicate selecting partitions to process.
                See :meth:`partitions`.
            selected_partitions: Paths to select, relative to the collection
                or view.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: Variables to load. If ``None``, load
                all variables.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A Dask bag of partition keys and function results. Partitions
            missing from storage are excluded.

        Raises:
            TypeError: If ``func`` is not callable.

        Notes:
            Callback exceptions propagate. A callback declaring
            ``partition_info`` receives a :class:`zcollection.PartitionInfo`
            identifying the partition and its slice in the overlap
            window. See :doc:`/computing` for input selection,
            callback arguments, and execution options.

        Example:
            For a collection with ``var1`` and ``var2`` on the same dimensions,
            including the partitioned dimension:

            .. code-block:: python

                bag = collection.map(
                    lambda ds: ds['var1'].values + ds['var2'].values)
                for partition, values in bag.compute(scheduler='synchronous'):
                    print(partition, values.tolist())
        """

        return self._map(func,
                         args,
                         kwargs,
                         delayed=delayed,
                         depth=0,
                         filters=filters,
                         selected_partitions=selected_partitions,
                         partition_size=partition_size,
                         npartitions=npartitions,
                         selected_variables=selected_variables)

    def map_overlap(
        self,
        func: MapCallable,
        /,
        *args,
        delayed: bool = True,
        depth: int = 1,
        filters: PartitionFilter = None,
        selected_partitions: Iterable[str] | None = None,
        partition_size: int | None = None,
        npartitions: int | None = None,
        selected_variables: Iterable[str] | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Apply a function to each partition with neighboring data.

        Args:
            func: The function to apply to each selected partition.
            *args: The positional arguments to pass to the function.
            delayed: Whether to load variables lazily.
            depth: Number of neighboring partitions to load on each side.
                Defaults to 1. Neighbors are loaded even when excluded by
                ``filters`` or ``selected_partitions``.
            filters: Predicate selecting partitions to process.
                See :meth:`partitions`.
            selected_partitions: Paths to select, relative to the collection
                or view.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            selected_variables: Variables to load. If ``None``, load
                all variables.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A Dask bag of partition keys and function results. Partitions
            missing from storage are excluded.

        Raises:
            TypeError: If ``func`` is not callable.
            ValueError: If ``depth`` is negative.

        Notes:
            Callback exceptions propagate. A callback declaring
            ``partition_info`` receives a :class:`zcollection.PartitionInfo`
            identifying the partition and its slice in the overlap
            window. See :doc:`/computing` for input selection,
            callback arguments, and execution options.

        Example:
            For a collection with ``var1`` and ``var2`` on the same dimensions,
            including the partitioned dimension:

            Each printed array includes results for the partition and its
            available neighbours.

            .. code-block:: python

                bag = collection.map_overlap(
                    lambda ds: ds['var1'].values + ds['var2'].values,
                    depth=1)
                for partition, values in bag.compute(scheduler='synchronous'):
                    print(partition, values.tolist())
        """
        return self._map(func,
                         args,
                         kwargs,
                         delayed=delayed,
                         depth=depth,
                         filters=filters,
                         selected_partitions=selected_partitions,
                         partition_size=partition_size,
                         npartitions=npartitions,
                         selected_variables=selected_variables)

    def _is_selected(
        self,
        partition: Sequence[str],
        expr: Callable[[dict[str, int]], bool] | None,
    ) -> bool:
        """Check whether a partition matches the selection.

        Args:
            partition: The partition to check.
            expr: The expression used to filter the partition.

        Returns:
            ``True`` if the partition matches, otherwise ``False``.
        """
        return True if expr is None else expr(
            dict(self.partitioning.parse('/'.join(partition))))


class WritableCollection(BaseCollection):
    """Share partition updates and index reconstruction."""

    _index: PartitionIndex
    #: Task that removes a variable array from a partition.
    _DROP_ARRAY: ClassVar[Callable[..., Any]]

    @abc.abstractmethod
    def _write_config(self) -> None:
        """Write the configuration, which contains the schema."""
        ...

    @contextlib.contextmanager
    def _changing_schema(self) -> Iterator[meta.DatasetSchema]:
        """Write the configuration after a ``with`` block changes the schema.

        If the block or ``_write_config`` raises, the schema's variable
        declarations are restored to their state before the block.

        Yields:
            The schema to modify.
        """
        before = dict(self.metadata.variables)
        try:
            yield self.metadata
            self._write_config()
        except BaseException:
            self.metadata.variables.clear()
            self.metadata.variables.update(before)
            raise

    def _declare_variable(self,
                          variable: meta.VariableSchema,
                          dimensions: set[str] | None = None) -> None:
        """Add a variable to the schema and write the configuration.

        Args:
            variable: The variable to add.
            dimensions: The dimensions the variable may use. Defaults to
                those of the schema.
        """
        with self._changing_schema() as schema:
            schema.add_variable(variable, dimensions)

    def _undeclare_variable(self, name: str) -> None:
        """Remove a variable from the schema and write the configuration.

        Args:
            name: The name of the variable to remove.
        """
        with self._changing_schema() as schema:
            schema.drop_variable(name)

    @staticmethod
    def _client_for(paths: Sequence[str], *, distributed: bool,
                    npartitions: int | None) -> dask.distributed.Client | None:
        """Return the Dask client processing ``paths``, or ``None``.

        Args:
            paths: The partitions to process.
            distributed: Whether to use Dask.
            npartitions: The largest number of partitions processed at the
                same time.

        Raises:
            ValueError: If ``paths`` is not empty and `npartitions` is below
                one.
        """
        if not paths:
            return None
        execution.check_npartitions(npartitions)
        return dask_utils.get_client() if distributed else None

    def _drop_arrays(self, name: str, paths: Sequence[str], *,
                     client: dask.distributed.Client | None,
                     npartitions: int | None, stop_on_error: bool) -> None:
        """Remove a variable array from the selected partitions.

        The caller holds the lock.

        Args:
            name: The name of the variable.
            paths: The partitions.
            client: The Dask client, or ``None``.
            npartitions: The largest number of partitions processed at the
                same time.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish.
        """
        execution.run(self._DROP_ARRAY,
                      paths, [self._relative_path(path) for path in paths],
                      client=client,
                      npartitions=npartitions,
                      stop_on_error=stop_on_error,
                      kwargs={
                          'variable': name,
                          'fs': self.fs
                      })

    def _undo_addition(self, name: str, paths: Sequence[str], *,
                       client: dask.distributed.Client | None,
                       npartitions: int | None, error: BaseException) -> None:
        """Undo a failed variable addition.

        The caller holds the lock.

        Args:
            name: The name of the variable.
            paths: The partitions of the failed addition.
            client: The Dask client, or ``None``.
            npartitions: The largest number of partitions processed at the
                same time.
            error: The original error. A note is added if cleanup fails.
        """
        try:
            self._drop_arrays(name,
                              paths,
                              client=client,
                              npartitions=npartitions,
                              stop_on_error=False)
        except Exception as removal:
            note_failed_removal(
                error, name, removal.failed if isinstance(
                    removal, execution.PartitionErrors) else
                [self._relative_path(path) for path in paths])
            return
        self._undeclare_variable(name)

    @abc.abstractmethod
    def _select_update_partitions(
        self,
        filters: PartitionFilter,
        selected_partitions: Iterable[str] | None,
    ) -> tuple[str, ...]:
        """Select the partition paths to update."""
        ...

    @abc.abstractmethod
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
        """Validate output variables and prepare the partition update
        function."""
        ...

    def _update(  # noqa: PLR0913
        self,
        func: UpdateCallable,
        /,
        *args,
        depth: int,
        filters: PartitionFilter,
        variables: Sequence[str] | None,
        selected_variables: Iterable[str] | None,
        selected_partitions: Iterable[str] | None,
        trim: bool,
        npartitions: int | None,
        delayed: bool | None,
        distributed: bool,
        stop_on_error: bool,
        **kwargs,
    ) -> tuple[str, ...]:
        """Prepare and execute updates, then invalidate the selected paths.

        Returns:
            The path of each partition written, relative to the collection or
            the view.
        """
        selected_variables = prepare_partition_operation(
            func, depth, selected_variables)
        if delayed is None:
            delayed = distributed

        partitions = self._select_update_partitions(filters,
                                                    selected_partitions)
        if not partitions:
            warn_empty_update()
            return ()

        client = dask_utils.get_client() if distributed else None
        wrap_function = self._prepare_update(
            func,
            partitions,
            pool=self._overlap_pool(partitions) if depth else partitions,
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
            return ()

        with self.synchronizer:
            try:
                return execution.run(
                    wrap_function,
                    partitions,
                    [self._relative_path(item) for item in partitions],
                    client=client,
                    npartitions=npartitions,
                    stop_on_error=stop_on_error,
                    kwargs={
                        'func_args': args,
                        'func_kwargs': kwargs
                    })
            finally:
                tuple(map(self.fs.invalidate_cache, partitions))

    @property
    @abc.abstractmethod
    def _reference(self) -> ReadOnlyCollection:
        """Collection that provides the values of `align_on`."""
        ...

    def _variables_to_update(self, zds: dataset.Dataset,
                             align_on: str) -> tuple[str, ...]:
        """Validate the input dataset for :meth:`update_from`.

        Args:
            zds: Dataset containing the new values.
            align_on: Variable identifying the rows of a partition.

        Returns:
            The names of the variables to update.

        Raises:
            ValueError: If `align_on` cannot identify the rows, or the dataset
                lacks a variable it needs or contains one the schema does not
                define along the same dimensions.
        """
        reference = self._reference
        aligned = reference.metadata.variables.get(align_on)
        if aligned is None:
            raise ValueError(
                f'The collection {reference.partition_properties.dir!r} has '
                f'no variable {align_on!r}.')
        if aligned.dimensions != (self.dimension, ):
            raise ValueError(
                f'The variable {align_on!r} must have {self.dimension!r} as '
                'its only dimension.')

        required = (*self.partitioning.variables, align_on)
        missing = [name for name in required if name not in zds.variables]
        if missing:
            raise ValueError('The dataset does not contain the variables '
                             f'{", ".join(repr(name) for name in missing)}.')

        names = tuple(name for name in zds.variables if name not in required)
        unknown = sorted(set(names) - set(self.metadata.variables))
        if unknown:
            raise ValueError('The schema does not define the variables '
                             f'{", ".join(repr(x) for x in unknown)}.')

        for name in names:
            declared = self.metadata.variables[name].dimensions
            if self.dimension not in declared:
                raise ValueError(f'The variable {name!r} does not have the '
                                 f'dimension {self.dimension!r}.')
            if zds.variables[name].dimensions != declared:
                raise ValueError(
                    f'The variable {name!r} has the dimensions '
                    f'{zds.variables[name].dimensions} in the dataset and '
                    f'{declared} in the schema.')
        return names

    def update_from(self,
                    zds: dataset.Dataset,
                    *,
                    align_on: str,
                    npartitions: int | None = None,
                    distributed: bool = True,
                    stop_on_error: bool = True) -> tuple[str, ...]:
        """Update existing partitions using values from a dataset.

        Each input row updates the stored row with the same value of
        ``align_on``. All other rows and variables remain unchanged.

        The input dataset is validated before writing begins. It must contain
        ``align_on`` and the partitioning variables. All variables to update
        must be declared in the schema with matching dimensions that span
        the partitioned dimension, and every row must belong to an existing
        partition. Rows are then matched within each partition. A partition
        with unmatched rows or duplicate ``align_on`` values is left
        unchanged and reported as failed.

        Args:
            zds: Dataset containing the partitioning variables, `align_on`
                and the variables to update.
            align_on: Variable identifying the rows of a partition. A view
                reads it from its reference collection.
            npartitions: The largest number of partitions written at the
                same time. Defaults to the number of Dask workers. With
                ``distributed=False``, the partitions are processed one at a
                time.
            distributed: Whether to use Dask. Defaults to True.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.

        Returns:
            The path of each partition written, relative to the collection or
            the view.

        Raises:
            ValueError: If `align_on` cannot identify the rows, the dataset
                lacks a variable it needs or contains one the schema does not
                declare as above, a row falls in a partition that does not
                exist, or ``npartitions`` is below one.
            PartitionErrors: If a partition failed, for example when the
                values of `align_on` repeat among its stored rows or among
                the dataset rows that fall in it, or when a row of the
                dataset has no match in it.
        """
        names = self._variables_to_update(zds, align_on)
        if not names:
            return ()

        reference = self._reference
        with self.synchronizer:
            existing = frozenset(self.partitions())
            plan: list[tuple[str, str, dict[str, slice]]] = []
            for keys, indexer in self.partitioning.split_dataset(
                    zds=zds, axis=self.dimension):
                partition = fs_utils.join_path(self._partitions_dir, *keys)
                if partition not in existing:
                    raise ValueError(
                        f'The partition {partition!r} does not exist.')
                plan.append(
                    (partition,
                     fs_utils.join_path(reference.partition_properties.dir,
                                        *keys), indexer))
            if not plan:
                return ()
            client = dask_utils.get_client() if distributed else None
            return execution.run(
                update_partition_from,
                plan,
                [self._relative_path(partition) for partition, _, _ in plan],
                client=client,
                npartitions=npartitions,
                stop_on_error=stop_on_error,
                kwargs={
                    'zds': client.scatter(zds, hash=False) if client else zds,
                    'names': names,
                    'align_on': align_on,
                    'dim': self.dimension,
                    'fs': self.fs,
                    'reference_fs': reference.fs,
                })

    @staticmethod
    def _unsupported_operation(*args, **kwargs) -> NoReturn:
        """Reject an operation that modifies the collection."""
        raise io.UnsupportedOperation('not writable')

    def _refuse_writes(self, methods: Iterable[str]) -> None:
        """Bind each named method to :meth:`_unsupported_operation`.

        Args:
            methods: Names of methods that modify the collection.
        """
        for item in methods:
            assert hasattr(self, item), f'{item} is not a known method.'
            setattr(
                self, item,
                types.MethodType(WritableCollection._unsupported_operation,
                                 self))

    @abc.abstractmethod
    def validate_partitions(
        self,
        filters: PartitionFilter = None,
        distributed: bool = True,
        fix: bool = False,
        *,
        npartitions: int | None = None,
        stop_on_error: bool = True,
    ) -> tuple[str, ...]:
        """List partitions that cannot be read.

        Args:
            filters: The predicate selecting the partitions to check. By
                default, every partition is checked.
            distributed: Whether to use Dask. Defaults to ``True``.
            fix: Whether to repair invalid partitions.
            npartitions: The largest number of partitions processed at the
                same time. Defaults to the number of Dask workers.
            stop_on_error: If True, no partition starts after the first
                failure, except the ones already submitted to Dask, which
                finish. Defaults to True.

        Returns:
            Paths of invalid partitions, relative to the collection or view.
        """
        ...

    def reindex(self) -> tuple[str, ...]:
        """Rebuild the index from a listing of the partitions in the store.

        A partition missing from the index becomes visible to `partitions`,
        and an entry for a partition absent from the store is dropped.

        Returns:
            The path of each partition, relative to the collection.
        """
        _LOGGER.info('Indexing the partitions of %s', self)
        with self.synchronizer:
            keys = self._index.reindex()
        return tuple(self._index.path_of(key) for key in keys)


class ReadOnlyCollection(BaseCollection):
    """Base class providing read-only access to a collection.

    The arguments of the constructor are detailed in the documentation
    of :py:class:`~zcollection.collection.Collection`.
    """
    #: Configuration filename of the collection.
    CONFIG: ClassVar[str] = '.zcollection'

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
        if axis not in ds.variables:
            raise ValueError(
                f'The variable {axis!r} is not defined in the dataset.')

        for varname in partition_handler.variables:
            if varname not in ds.variables:
                raise ValueError(
                    f'The partitioning key {varname!r} is not defined in '
                    'the dataset.')

        # Work on a copy when setting the partitioned dimension size.
        main_dimension = ds.variables[axis].dimensions[0]
        ds = meta.DatasetSchema.from_config(data=ds.get_config())
        ds.dimensions[main_dimension].value = -1

        if (mode or 'w') == 'w':
            _check_sizes(ds, main_dimension)
            _check_fill_values(ds)

        self._settings = CollectionSettings(mode or 'w',
                                            fs_utils.get_fs(filesystem),
                                            synchronizer or sync.NoSync())

        self._properties = CollectionProperties(
            axis, ds, partition_handler,
            PartitioningProperties(
                fs_utils.normalize_path(self._settings.filesystem,
                                        partition_base_dir), main_dimension))

        #: Path to the dataset storing immutable variables.
        self._immutable = _immutable_path(self._properties.partition)

        #: The collection's partition index.
        self._index = PartitionIndex(self._settings.filesystem,
                                     self._properties.partition.dir,
                                     partition_handler,
                                     writable=self._settings.mode == 'w')

        #: Version of the layout the collection is written in.
        self.version: int = meta.FORMAT_VERSION

    @property
    def axis(self) -> str:
        """Return the name of the axis variable."""
        return self._properties.axis

    @property
    def dimension(self) -> str:
        """Return the partitioned dimension of the axis variable."""
        return self._properties.dimension

    @property
    def metadata(self) -> meta.DatasetSchema:
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
    def mode(self) -> str:
        """Return the collection access mode."""
        return self._settings.mode

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """Return the collection file system."""
        return self._settings.filesystem

    @property
    def synchronizer(self) -> sync.Sync:
        """Return the synchronizer of the collection."""
        return self._settings.synchronizer

    @property
    def immutable_path(self) -> str:
        """Return the path to the collection's immutable variables."""
        return self._immutable

    @property
    def have_immutable(self) -> bool:
        """Return ``True`` if the collection has immutable variables."""
        return len(self.immutable_variables) > 0

    @property
    def immutable_variables(self) -> set[str]:
        """Return the immutable variables of the collection."""
        return self.metadata.select_variables_by_dims((self.dimension, ),
                                                      False)

    @classmethod
    def _config(cls, partition_base_dir: str) -> str:
        """Return the configuration path."""
        return fs_utils.join_path(partition_base_dir, cls.CONFIG)

    def _indexed_partitions(self) -> tuple[str, ...]:
        """Return the paths of the partitions listed in the index.

        Returns:
            The paths, ordered by the values of their keys.
        """
        base_dir: str = self.partition_properties.dir
        return tuple(
            fs_utils.join_path(base_dir, self._index.path_of(key))
            for key in self._index.keys())

    @property
    def _partitions_dir(self) -> str:
        return self.partition_properties.dir

    def _normalize_partitions(self,
                              partitions: Iterable[str]) -> Iterable[str]:
        """Normalize and sort selected partition paths.

        Prefix relative paths with the collection's base directory,
        remove duplicates, and retain existing partitions only.

        Args:
            partitions: Partition paths to normalize.

        Returns:
            Absolute partition paths, without duplicates and ordered
            by their partition keys.
        """
        # Parse partition keys before sorting them.
        partitions_scheme = sorted(
            {self.partitioning.parse(item)
             for item in partitions})
        paths = [
            self.fs.sep.join((self.partition_properties.dir,
                              self.partitioning.join(partition, self.fs.sep)))
            for partition in partitions_scheme
        ]
        existing = [path for path in paths if self.fs.exists(path)]
        if len(existing) < len(paths):
            warn_missing_partitions([
                self._relative_path(path) for path in paths
                if path not in existing
            ])
        return existing

    def dimensions_properties(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return the sizes and chunk sizes of the collection dimensions.

        Returns:
            A pair of mappings from dimension names to sizes and chunk sizes.
        """
        chunks: dict[str, int] = {}
        dimensions: dict[str, int] = {}

        for dim in self.metadata.dimensions.values():
            chunks[dim.name] = dim.chunks

            if dim.name != self.dimension:
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
        """List the collection's partitions, optionally selecting a subset.

        Args:
            filters: Expression or callable selecting partitions by their
                keys. A string contains a Python expression using the
                partition key names; a callable takes the keys and returns
                a boolean.
            indexer: Indexer used to select partitions.
            selected_partitions: Partition paths relative to the collection.
            relative: Whether to return relative paths. Defaults to
                ``False`` (absolute paths).
            lock: Whether to lock the collection while listing partitions.

        Returns:
            Partition paths ordered by the values of their keys, rather
            than lexicographically (``cycle=2`` precedes ``cycle=10``).

        Example:
            >>> tuple(collection.partitions(
            ...     filters="year == 2019 and month == 1", relative=True))
            ('year=2019/month=01/day=01', 'year=2019/month=01/day=02', ...)
            >>> tuple(collection.partitions(
            ...     filters=lambda x: x["year"] == 2019 and x["month"] == 1,
            ...     relative=True))
            ('year=2019/month=01/day=01', 'year=2019/month=01/day=02', ...)
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
            partitions = self._normalize_partitions(selected_partitions)
        elif lock:
            with self.synchronizer:
                partitions = self._indexed_partitions()
        else:
            partitions = self._indexed_partitions()

        if indexer is not None:
            # Retain partitions present in both lists.
            partitions = list(partitions)
            partitions = [
                p for p in list_partitions_from_indexer(
                    indexer, self.partitioning, self.partition_properties.dir,
                    self.fs.sep) if p in partitions
            ]

        yield from (self._relative_path(item) if relative else item
                    for item in partitions
                    if (item != self._immutable and self._is_selected(
                        item.replace(base_dir, '').split(sep), expr)))

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
        immutable = self._immutable if self.have_immutable else None
        load = functools.partial(_load_dataset,
                                 delayed,
                                 self.fs,
                                 immutable,
                                 selected_variables=selected_variables)
        load_overlap = functools.partial(
            _load_dataset_with_overlap,
            delayed=delayed,
            depth=depth,
            dim=self.dimension,
            axis=self.axis,
            fs=self.fs,
            immutable=immutable,
            partitions=partitions,
            selected_variables=selected_variables) if depth else None

        return load, load_overlap

    def load(
        self,
        *,
        delayed: bool | None = None,
        filters: PartitionFilter = None,
        indexer: Indexer | None = None,
        selected_variables: Iterable[str] | None = None,
        selected_partitions: Iterable[str] | None = None,
        distributed: bool = True,
    ) -> dataset.Dataset | None:
        """Load collection data using partition and variable selections.

        Args:
            delayed: Whether to load variables lazily. Defaults to
                ``distributed`` when unspecified: Dask arrays if ``True``,
                NumPy arrays if ``False``.
            filters: Predicate selecting partitions by key.
                See :meth:`partitions`.
            indexer: The indexer to apply.
            selected_variables: Variables to load. If ``None``, load
                all variables.
            selected_partitions: Partition paths to select, relative to
                the collection.
            distributed: Whether to use Dask. Defaults to ``True``.

        Returns:
            A dataset containing the selected partitions, or ``None`` if an
            explicit selection retains no partition data. With no
            selection, a collection without written partitions can still
            return its immutable variables.

        Warning:
            Do not reinsert a dataset loaded with ``selected_variables``.
            Omitted variables would be reset to their fill values.

        Example:
            >>> collection = ...
            >>> collection.load(
            ...     filters="year == 2019 and month == 3 and day % 2 == 0")
            >>> collection.load(
            ...     filters=lambda keys: keys["year"] == 2019 and
            ...     keys["month"] == 3 and keys["day"] % 2 == 0)
        """
        if delayed is None:
            delayed = distributed

        array: dataset.Dataset | None = None

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

        if arrays is not None:
            array = arrays.pop(0)
            if arrays:
                array = array.concat(arrays, self.dimension)

        # An explicit selection with no partition data returns None.
        # Without a selection, immutable variables can still be loaded alone.
        if array is None and (filters is not None or indexer is not None
                              or selected_partitions is not None):
            return None

        array = self.merge_immutable(array, selected_variables, delayed)

        if array is not None:
            array.fill_attrs(self.metadata)

        return array

    def merge_immutable(self, ds: dataset.Dataset | None,
                        selected_variables: Iterable[str] | None,
                        delayed: bool) -> dataset.Dataset | None:
        """Load immutable variables and add them to a dataset.

        Args:
            ds: Dataset to extend with immutable variables.
            selected_variables: Variables to read.
            delayed: Whether to load variables lazily.

        Returns:
            The dataset including immutable variables, or ``None`` if no
            dataset was provided and no immutable variables were loaded.
        """
        var_im = self.immutable_variables

        if selected_variables is not None:
            var_im = var_im.intersection(selected_variables)

        if not var_im:
            return ds

        ds_im = storage.open_zarr_group(self._immutable,
                                        self.fs,
                                        delayed=delayed,
                                        selected_variables=selected_variables)
        if not ds_im.variables:
            return ds

        ds_im.merge(ds)

        return ds_im

    def _load_partitions(
        self,
        *,
        delayed: bool = True,
        filters: PartitionFilter = None,
        selected_variables: Iterable[str] | None = None,
        selected_partitions: Iterable[str] | None = None,
        distributed: bool = True,
    ) -> list[dataset.Dataset] | None:
        """Load partitions selected by filters or explicit paths.

        Args:
            delayed: Whether to load variables lazily.
            filters: Predicate selecting partitions by key.
                See :meth:`partitions`.
            selected_variables: Variables to load. If ``None``, load
                all variables.
            selected_partitions: Partition paths to select, relative to
                the collection.
            distributed: Whether to use Dask. Defaults to ``True``.

        Returns:
            The partition datasets, or ``None`` if no selected partitions
            are available in storage.
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
                npartitions=dask_utils.dask_workers(client, True))
            arrays = bag.map(_open_held_group,
                             self.fs,
                             delayed=delayed,
                             selected_variables=selected_variables).filter(
                                 lambda item: item is not None).compute()
        else:
            arrays = [
                zds for zds in (
                    _open_held_group(partition,
                                     self.fs,
                                     delayed=delayed,
                                     selected_variables=selected_variables)
                    for partition in selected_partitions) if zds is not None
            ]

        return arrays or None

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
        """Load selected partitions using an indexer.

        Args:
            indexer: The indexer to apply.
            delayed: Whether to load variables lazily.
            filters: Predicate selecting partitions by key.
                See :meth:`partitions`.
            selected_variables: Variables to load. If ``None``, load
                all variables.
            selected_partitions: Partition paths to select, relative to
                the collection.
            distributed: Whether to use Dask. Defaults to ``True``.

        Returns:
            The partition datasets, or ``None`` if no selected partitions
            are available in storage.
        """
        # Materialize the indexer results so they can be reused.
        indexer = tuple(indexer)

        # Build the indexer arguments.
        partitions = self.partitions(selected_partitions=selected_partitions,
                                     filters=filters,
                                     indexer=indexer)
        args = tuple(
            build_indexer_args(self, filters, indexer, partitions=partitions))
        if len(args) == 0:
            return None

        # Load the selected partitions and apply the indexer.
        if distributed:
            client = dask_utils.get_client()
            bag = dask.bag.core.from_sequence(
                args, npartitions=dask_utils.dask_workers(client, True))

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
                        a,
                        delayed=delayed,
                        fs=self.fs,
                        partition_handler=self.partitioning,
                        partition_properties=self.partition_properties,
                        selected_variables=selected_variables) for a in args
                ]))

        return arrays or None

    def _bag_from_partitions(
        self,
        filters: PartitionFilter | None = None,
        **kwargs,
    ) -> dask.bag.core.Bag:
        """Create a Dask bag from selected partitions.

        Args:
            filters: The predicate used to filter the partitions to load.
            kwargs: The keyword arguments to pass to the method
                :meth:`partitions`.

        Returns:
            The Dask bag.
        """
        partitions: list[str] = [*self.partitions(filters=filters, **kwargs)]
        return dask.bag.core.from_sequence(partitions,
                                           npartitions=len(partitions))

    def iterate_on_records(self) -> Iterator[tuple[str, str]]:
        """Iterate over partition paths in relative and absolute forms.

        Yields:
            The relative and absolute path of each partition.
        """
        yield from ((self._relative_path(item), item)
                    for item in self.partitions())
