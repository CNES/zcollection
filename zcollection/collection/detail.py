# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Implementation details.
=======================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import dataclasses
import sys
import time
import traceback
import warnings

import dask.bag.core
import dask.distributed
import dask.utils
import numpy

from .. import dataset, merging, meta, partitioning, store, sync
from ..execution import SKIPPED
from ..fs_utils import join_path
from ..storage import (
    AbsentPartition,
    open_zarr_group,
    partition_dimension_size,
    read_partition,
    update_zarr_array,
    warn_absent_partition,
    write_zarr_group,
)
from .callable_objects import PartitionInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    import types

    import fsspec

    from ..type_hints import NDArray
    from .callable_objects import (
        MapCallable,
        UpdateCallable,
        WrappedPartitionCallable,
    )

#: Partition's type.
PartitionSlice = tuple[tuple[str, ...], dict[str, slice]]

#: Maximum number of tries to remove a directory.
MAX_TRIES: int = 10

#: The largest number of partitions named in the note of a failed removal.
_NAMED_PARTITIONS = 10


@dataclasses.dataclass(frozen=True)
class PartitioningProperties:
    """Properties of a partition."""
    #: The base directory of the partition.
    dir: str
    #: The name of the partitioning dimension.
    dim: str


def _get_slices(variable: meta.VariableSchema, dim: str,
                indices: slice) -> tuple[slice, ...]:
    """Return a tuple of slices that can be used to select the given dimension
    indices.

    Args:
        dim: Dimension to select
        indices: Dimension indices to select

    Returns:
        The slices
    """
    slices: list[slice] = [slice(None)] * len(variable.dimensions)
    slices[variable.dimensions.index(dim)] = indices
    return tuple(slices)


def prepare_partition_operation(
    func: Callable,
    depth: int,
    selected_variables: Iterable[str] | None,
) -> tuple[str, ...] | None:
    """Validate the callback and depth, and retain a reusable variable selection."""
    if not callable(func):
        raise TypeError('func must be a callable')
    if depth < 0:
        raise ValueError('Depth must be non-negative.')
    return tuple(
        selected_variables) if selected_variables is not None else None


def _get_partition_info(
    partition: str,
    handler: partitioning.Partitioning,
    dim: str,
    indices: slice | None = None,
) -> PartitionInfo:
    """Describe the partition keys, relative name and central slice."""
    scheme = handler.parse(partition)
    return PartitionInfo(
        partition=dict(scheme),
        name=handler.join(scheme, '/'),
        dimension=dim,
        indices=indices if indices is not None else slice(None))


def _call_partition(
    func: Callable,
    zds: dataset.Dataset,
    partition_info: PartitionInfo,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
) -> Any:
    """Pass partition information to callbacks that declare the parameter."""
    if dask.utils.has_keyword(func, 'partition_info'):
        func_kwargs = dict(func_kwargs, partition_info=partition_info)
    return func(zds, *func_args, **func_kwargs)


def _try_infer_callable(
    func: Callable,
    zds: dataset.Dataset,
    partition_info: PartitionInfo,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
) -> Any:
    """Try to call a function with the given arguments.

    Args:
        func: Function to call.
        zds: Dataset to pass to the function.
        partition_info: Identity and location of the current partition.
        func_args: Positional arguments to pass to the function.
        func_kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function.
    """
    try:
        return _call_partition(func, zds, partition_info, func_args,
                               func_kwargs)
    except Exception as exc:
        exc_type: type[BaseException] | None
        exc_traceback: types.TracebackType | None
        exc_value: BaseException | None
        exc_type, exc_value, exc_traceback = sys.exc_info()
        execution_tb: str = ''.join(traceback.format_tb(exc_traceback))
        raise RuntimeError(
            f'An error occurred while applying the function {func} with '
            f'the arguments {func_args} and {func_kwargs} to the partition. The '
            f'error is: {exc_type!r}: {exc_value}. The traceback '
            f'is: {execution_tb}') from exc


def aligned_positions(
    stored: NDArray,
    wanted: NDArray,
    *,
    name: str,
    partition: str,
) -> NDArray:
    """Find the rows of a partition matching the wanted values of a variable.

    Args:
        stored: Values of the variable in the partition.
        wanted: Values to locate.
        name: Name of the variable, used in error messages.
        partition: Partition path, used in error messages.

    Returns:
        Position in `stored` of each wanted value.

    Raises:
        ValueError: If the values of `stored` or of `wanted` repeat, or a
            wanted value has no match.
    """
    order = numpy.argsort(stored, kind='stable')
    ordered = stored[order]
    if numpy.any(ordered[1:] == ordered[:-1]):
        raise ValueError(f'The values of {name!r} repeat in the partition '
                         f'{partition!r}.')
    requested = numpy.sort(wanted)
    if numpy.any(requested[1:] == requested[:-1]):
        raise ValueError(f'The values of {name!r} repeat in the dataset rows '
                         f'of the partition {partition!r}.')

    positions = numpy.searchsorted(ordered, wanted)
    # searchsorted returns `ordered.size` for a value greater than every
    # stored one; that position is out of range and counts as no match.
    placed = positions < ordered.size
    found = numpy.zeros(wanted.shape, dtype=bool)
    found[placed] = ordered[positions[placed]] == wanted[placed]

    if not found.all():
        missing = int((~found).sum())
        first = wanted[numpy.flatnonzero(~found)[0]]
        raise ValueError(f'The partition {partition!r} has no match for '
                         f'{missing} of the {name!r} values of the dataset; '
                         f'the first is {first}.')
    return order[positions]


def update_partition_from(
    args: tuple[str, str, dict[str, slice]],
    *,
    zds: dataset.Dataset,
    names: Sequence[str],
    align_on: str,
    dim: str,
    fs: fsspec.AbstractFileSystem,
    reference_fs: fsspec.AbstractFileSystem,
) -> None:
    """Update variables of one partition from a dataset.

    The rows are matched before any variable of the partition is written: if
    a row of the dataset has no match, or the values of `align_on` repeat in
    the partition or in the dataset, the partition keeps its values.

    Args:
        args: Partition path, path of its reference partition, and dataset
            indexer.
        zds: Dataset containing the values to write.
        names: Names of the variables to update.
        align_on: Variable identifying the rows.
        dim: Partitioned dimension.
        fs: File system containing the partition.
        reference_fs: File system containing the reference partition.

    Raises:
        ValueError: If the values of `align_on` repeat in the reference
            partition or in the dataset, or a row of the dataset has no match
            in the reference partition.
    """
    partition, reference, indexer = args
    selected = zds.isel(indexer)
    positions = aligned_positions(store.open_ndarray(
        reference_fs, join_path(reference, align_on)),
                                  selected.variables[align_on].values,
                                  name=align_on,
                                  partition=reference)
    for name in names:
        variable = selected.variables[name]
        # Select `positions` along the partitioned dimension and every
        # element along the others.
        rows: list[Any] = [slice(None)] * len(variable.dimensions)
        rows[variable.dimensions.index(dim)] = positions
        update_zarr_array(join_path(partition, name),
                          variable.array,
                          fs,
                          selection=tuple(rows))


def _update_partition(
    func: UpdateCallable,
    zds: dataset.Dataset,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
    *,
    partition_info: PartitionInfo,
    indices: slice | None,
    fs: fsspec.AbstractFileSystem,
    path: str,
    trim: bool,
    metadata: meta.DatasetSchema,
) -> None:
    """Call the update function and write its results to the target partition.

    With overlap, indices selects the current partition within the dataset.
    Without overlap, indices is None.
    """
    dictionary = _call_partition(func, zds, partition_info, func_args,
                                 func_kwargs)
    for name, array in dictionary.items():
        if trim and indices is not None:
            slices = _get_slices(metadata.variables[name],
                                 partition_info.dimension, indices)
            values = array[slices]  # type: ignore[index]
        else:
            values = array
        update_zarr_array(join_path(path, name), values, fs)


def infer_variables(
    partitions: Sequence[str],
    load: Callable[[str], dataset.Dataset],
    func: UpdateCallable,
    dim: str,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
    synchronizer: sync.Sync,
    *,
    partitioning: partitioning.Partitioning,
) -> tuple[str, ...]:
    """Infer output names from the first available selected partition.

    The callback result stays in the task that loads the partition.
    """
    for partition in partitions:
        try:
            with synchronizer:
                zds = load(partition)
        except AbsentPartition:
            warn_absent_partition(partition)
            continue
        info = _get_partition_info(partition, partitioning, dim)
        return tuple(
            _try_infer_callable(func, zds, info, func_args, func_kwargs))
    return ()


def map_partitions(
    partitions: Sequence[str],
    func: MapCallable,
    load: Callable[[str], dataset.Dataset],
    *,
    load_overlap: Callable[[str], tuple[dataset.Dataset, slice]] | None,
    partitioning: partitioning.Partitioning,
    dim: str,
    partition_size: int | None,
    npartitions: int | None,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
) -> dask.bag.core.Bag:
    """Load each partition or window and map the callback with Dask arguments."""

    def wrap_function(partition: str, /, *args, **kwargs):
        try:
            if load_overlap is None:
                zds = load(partition)
                indices = None
            else:
                zds, indices = load_overlap(partition)
        except AbsentPartition as err:
            warn_absent_partition(str(err))
            return None
        info = _get_partition_info(partition, partitioning, dim, indices)
        return tuple(info.partition.items()), _call_partition(
            func, zds, info, args, kwargs)

    bag = dask.bag.core.from_sequence(partitions, partition_size, npartitions)
    return bag.map(wrap_function, *func_args,
                   **func_kwargs).filter(lambda item: item is not None)


def _load_dataset(
    delayed: bool,
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    partition: str,
    selected_variables: Iterable[str] | None,
) -> dataset.Dataset:
    """Load a dataset from a partition.

    Args:
        delayed: Whether to load the dataset lazily.
        fs: File system on which the Zarr dataset is stored.
        immutable: Name of the immutable directory.
        partition: Name of the partition.
        selected_variables: Name of the variables to load from the dataset.

    Returns:
        The loaded dataset.
    """
    zds = read_partition(partition,
                         fs,
                         delayed=delayed,
                         selected_variables=selected_variables)

    if immutable:
        zds.merge(
            open_zarr_group(immutable,
                            fs,
                            delayed=delayed,
                            selected_variables=selected_variables))
    return zds


def _load_dataset_with_overlap(
    partition: str,
    *,
    delayed: bool,
    depth: int,
    dim: str,
    axis: str,
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    partitions: Sequence[str],
    selected_variables: Iterable[str] | None,
) -> tuple[dataset.Dataset, slice]:
    """Load a dataset from a partition with overlap.

    Args:
        delayed: Whether to load the dataset lazily.
        depth: Depth of the overlap.
        dim: Name of the partitioning dimension.
        axis: Name of the stored axis variable.
        fs: File system on which the Zarr dataset is stored.
        immutable: Name of the immutable directory.
        partition: Name of the partition.
        partitions: List of all partitions.
        selected_variables: Name of the variables to load from the dataset.

    Returns:
        The loaded dataset and the slice to select the data without the
        overlap.
    """

    def calculate_slice(_groups: list[dataset.Dataset],
                        _selected_partitions: list[str]) -> slice:
        """Compute the slice of the selected dataset (without overlap)."""
        start = 0
        indexes = slice(0, 0, None)
        for idx, ds in enumerate(_groups):
            size = partition_dimension_size(dimensions=ds.dimensions,
                                            dim=dim,
                                            axis=join_path(
                                                _selected_partitions[idx],
                                                axis),
                                            fs=fs)
            indexes = slice(start, start + size, None)
            if partition == _selected_partitions[idx]:
                break
            start += size
        return indexes

    where: int = partitions.index(partition)

    # Search for the overlapping partitions
    selected_partitions: list[str] = [
        partitions[ix] for ix in range(where - depth, where + depth + 1)
        if 0 <= ix < len(partitions)
    ]

    # Load the datasets for each selected partition. A neighbour the store
    # does not hold is left out of both lists, which calculate_slice reads
    # in step. The partition itself is the one being mapped over, so its
    # absence is raised.
    groups: list[dataset.Dataset] = []
    held: list[str] = []
    for item in selected_partitions:
        try:
            groups.append(
                read_partition(item,
                               fs,
                               delayed=delayed,
                               selected_variables=selected_variables))
        except AbsentPartition:
            if item == partition:
                raise
            warn_absent_partition(item)
            continue
        held.append(item)
    selected_partitions = held

    # Compute the slice of the given partition.
    indices: slice = calculate_slice(groups, selected_partitions)

    # Build the dataset for the selected partitions.
    zds: dataset.Dataset = groups.pop(0)
    if groups:
        zds = zds.concat(groups, dim)

    if immutable:
        zds.merge(
            open_zarr_group(immutable,
                            fs,
                            delayed=delayed,
                            selected_variables=selected_variables))
    return zds, indices


def note_failed_removal(error: BaseException, variable: str,
                        failed: Iterable[str]) -> None:
    """Add a note to `error` naming where the array of `variable` may
    remain.

    Args:
        error: The error of the addition.
        variable: The name of the variable added.
        failed: The partitions where removing the array failed.
    """
    names = list(failed)
    shown = ', '.join(names[:_NAMED_PARTITIONS])
    if len(names) > _NAMED_PARTITIONS:
        shown += f' and {len(names) - _NAMED_PARTITIONS} others'
    error.add_note(
        f'Removing the variable {variable!r} failed on {shown}, where its '
        'array may remain, so the schema still declares it. Call '
        'drop_variable to remove it.')


def warn_empty_update() -> None:
    """Warn that the update has no selected partitions or output variables."""
    warnings.warn(
        'The update function is not applied because no partitions or output '
        'variables are selected.',
        category=RuntimeWarning,
        stacklevel=3)


def wrap_update(
    func: UpdateCallable,
    load: Callable[[str], dataset.Dataset],
    *,
    load_overlap: Callable[[str], tuple[dataset.Dataset, slice]] | None,
    partitioning: partitioning.Partitioning,
    dim: str,
    fs: fsspec.AbstractFileSystem,
    metadata: meta.DatasetSchema,
    trim: bool,
) -> WrappedPartitionCallable:
    """Return the function that loads one partition or its overlap window,
    calls `func` on it and writes the partition.

    The function returns :data:`zcollection.execution.SKIPPED` for an absent
    partition, after a warning.
    """

    def wrap_function(partition: str, func_args: tuple[Any, ...],
                      func_kwargs: dict[str, Any]) -> Any:
        try:
            if load_overlap is None:
                zds = load(partition)
                indices = None
            else:
                zds, indices = load_overlap(partition)
        except AbsentPartition:
            warn_absent_partition(partition)
            return SKIPPED
        _update_partition(func,
                          zds,
                          func_args,
                          func_kwargs,
                          partition_info=_get_partition_info(
                              partition, partitioning, dim, indices),
                          indices=indices,
                          fs=fs,
                          path=partition,
                          trim=trim,
                          metadata=metadata)
        return None

    wrap_function.__name__ = getattr(func, '__name__', 'update')
    return wrap_function


def _rm(fs: fsspec.AbstractFileSystem, dirname: str) -> None:
    """Remove a directory and its content.

    Args:
        fs: The file system on which the directory is stored.
        dirname: The name of the directory to remove.
    """
    tries = 0
    while tries < MAX_TRIES:
        try:
            fs.rm(dirname, recursive=True)
            fs.invalidate_cache(dirname)
            if not fs.exists(dirname):
                return
        except OSError:
            fs.invalidate_cache(dirname)
        time.sleep(1)
        tries += 1


def _insert(
    args: PartitionSlice,
    /,
    *,
    axis: str,
    zds: dataset.Dataset,
    fs: fsspec.AbstractFileSystem,
    merge_callable: merging.MergeCallable | None,
    partitioning_properties: PartitioningProperties,
    delayed: bool | None = None,
    distributed: bool = True,
    merge_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Insert or update a partition in the collection.

    Args:
        args: Tuple containing the partition's name and its slice.
        axis: The axis to merge on.
        zds: The dataset to process.
        fs: The file system that the partition is stored on.
        merge_callable: The merge callable.
        partitioning_properties: The partitioning properties.
        delayed: Whether the merge callable receives the stored partition
            and the inserted one as Dask arrays. Defaults to ``None``, which
            gives it Dask arrays when the inserted dataset holds them.
        distributed: Whether to use dask or not. Default To True.
        merge_kwargs: Keyword arguments passed to the merge callable.
    """
    partition: tuple[str, ...]
    indexer: dict[str, slice]

    partition, indexer = args
    dirname = join_path(*((partitioning_properties.dir, *partition)))

    # A partition that holds no group is empty.
    if store.contains_group(fs, dirname):
        merging.perform(zds.isel(indexer),
                        dirname,
                        axis,
                        fs,
                        partitioning_properties.dim,
                        delayed=delayed,
                        distributed=distributed,
                        merge_callable=merge_callable,
                        merge_kwargs=merge_kwargs)
        return

    try:
        store.create_group(fs, dirname)

        # The synchronization is done by the caller.
        write_zarr_group(zds.isel(indexer),
                         dirname,
                         fs,
                         sync.NoSync(),
                         distributed=distributed)
    except:
        # If the construction of the new dataset fails, the created
        # partition is deleted to guarantee the integrity of the
        # collection.
        _rm(fs, dirname)
        raise


def _load_and_apply_indexer(
    args: tuple[tuple[tuple[str, int], ...], list[slice]],
    *,
    delayed: bool,
    fs: fsspec.AbstractFileSystem,
    partition_handler: partitioning.Partitioning,
    partition_properties: PartitioningProperties,
    selected_variables: Iterable[str] | None,
) -> list[dataset.Dataset]:
    """Load a partition and apply its indexer.

    Args:
        args: Tuple containing the partition's keys and its indexer.
        fs: The file system that the partition is stored on.
        partition_handler: The partitioning handler.
        partition_properties: The partitioning properties.
        selected_variables: The selected variables to load.

    Returns:
        The list of loaded datasets, empty when the store holds no
        partition at that path.
    """
    items: list[slice]
    partition_scheme: tuple[tuple[str, int], ...]

    partition_scheme, items = args
    partition = join_path(partition_properties.dir,
                          partition_handler.join(partition_scheme, fs.sep))
    try:
        zds = read_partition(partition,
                             fs,
                             delayed=delayed,
                             selected_variables=selected_variables)
    except AbsentPartition:
        warn_absent_partition(partition)
        return []
    return [zds.isel({partition_properties.dim: indexer}) for indexer in items]
