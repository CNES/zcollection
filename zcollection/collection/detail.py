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

import dask.distributed
import dask.utils

from .. import dask_utils, dataset, merging, meta, partitioning, store, sync
from ..fs_utils import join_path
from ..storage import (
    AbsentPartition,
    execute_transaction,
    open_zarr_group,
    read_partition,
    update_zarr_array,
    warn_absent_partition,
    write_zarr_group,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    import types

    import fsspec

    from .callable_objects import UpdateCallable, WrappedPartitionCallable

#: Partition's type.
PartitionSlice = tuple[tuple[str, ...], dict[str, slice]]

#: Maximum number of tries to remove a directory.
MAX_TRIES: int = 10


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


def _try_infer_callable(
    func: Callable,
    zds: dataset.Dataset,
    dim: str,
    *args,
    **kwargs,
) -> Any:
    """Try to call a function with the given arguments.

    Args:
        func: Function to call.
        zds: Dataset to pass to the function.
        dim: Name of the partitioning dimension.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function.
    """
    partition_info: tuple[str, slice]
    partition_info = dim, slice(0, zds.dimensions[dim])
    try:
        if dask.utils.has_keyword(func, 'partition_info'):
            return func(zds, *args, partition_info=partition_info, **kwargs)
        return func(zds, *args, **kwargs)
    except Exception as exc:
        exc_type: type[BaseException] | None
        exc_traceback: types.TracebackType | None
        exc_value: BaseException | None
        exc_type, exc_value, exc_traceback = sys.exc_info()
        execution_tb: str = ''.join(traceback.format_tb(exc_traceback))
        raise RuntimeError(
            f'An error occurred while applying the function {func} with '
            f'the arguments {args} and {kwargs} to the partition. The '
            f'error is: {exc_type!r}: {exc_value}. The traceback '
            f'is: {execution_tb}') from exc


def _update_partition(
    func: UpdateCallable,
    zds: dataset.Dataset,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
    *,
    indices: slice | None,
    dim: str,
    fs: fsspec.AbstractFileSystem,
    path: str,
    trim: bool,
    metadata: meta.DatasetSchema,
) -> None:
    """Call the update function and write its results to the target partition.

    With overlap, indices selects the current partition within the dataset.
    Without overlap, indices is None.
    """
    if indices is not None and dask.utils.has_keyword(func, 'partition_info'):
        dictionary = func(zds,
                          *func_args,
                          partition_info=(dim, indices),
                          **func_kwargs)
    else:
        dictionary = func(zds, *func_args, **func_kwargs)
    for name, array in dictionary.items():
        if trim and indices is not None:
            slices = _get_slices(metadata.variables[name], dim, indices)
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
        return tuple(
            _try_infer_callable(func, zds, dim, *func_args, **func_kwargs))
    return ()


def execute_update(
    func: WrappedPartitionCallable,
    partitions: Sequence[str],
    *,
    client: dask.distributed.Client | None,
    synchronizer: sync.Sync,
    name: str,
    npartitions: int | None,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
) -> None:
    """Execute partition updates locally or in batches on Dask workers."""
    if client is None:
        func(partitions, func_args, func_kwargs)
        return

    batches = tuple(
        dask_utils.split_sequence(
            partitions, npartitions
            or dask_utils.dask_workers(client, cores_only=True)))
    futures = client.map(func,
                         batches,
                         key=name,
                         func_args=func_args,
                         func_kwargs=func_kwargs)
    execute_transaction(client, synchronizer, futures)


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
            size: int = ds.dimensions[dim]
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
    dim: str,
    fs: fsspec.AbstractFileSystem,
    metadata: meta.DatasetSchema,
    trim: bool,
) -> WrappedPartitionCallable:
    """Load, update and write one partition or overlap window per iteration."""

    def wrap_function(partitions: Sequence[str], func_args: tuple[Any, ...],
                      func_kwargs: dict[str, Any]) -> None:
        for partition in partitions:
            try:
                if load_overlap is None:
                    zds = load(partition)
                    indices = None
                else:
                    zds, indices = load_overlap(partition)
            except AbsentPartition:
                warn_absent_partition(partition)
                continue
            _update_partition(func,
                              zds,
                              func_args,
                              func_kwargs,
                              indices=indices,
                              dim=dim,
                              fs=fs,
                              path=partition,
                              trim=trim,
                              metadata=metadata)
            del zds

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
    *,
    args: PartitionSlice,
    axis: str,
    zds: dataset.Dataset,
    fs: fsspec.AbstractFileSystem,
    merge_callable: merging.MergeCallable | None,
    partitioning_properties: PartitioningProperties,
    distributed: bool = True,
    **kwargs,
) -> None:
    """Insert or update a partition in the collection.

    Args:
        args: Tuple containing the partition's name and its slice.
        axis: The axis to merge on.
        zds: The dataset to process.
        fs: The file system that the partition is stored on.
        merge_callable: The merge callable.
        partitioning_properties: The partitioning properties.
        distributed: Whether to use dask or not. Default To True.
        **kwargs: Additional keyword arguments to pass to the merge callable.
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
                        delayed=zds.delayed if distributed else False,
                        distributed=distributed,
                        merge_callable=merge_callable,
                        **kwargs)
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
