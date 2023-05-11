# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Implementation details.
=======================
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence, Tuple
import dataclasses
import sys
import time
import traceback
import types

import dask.utils
import fsspec
import zarr.storage

from .. import dataset, merging, partitioning, sync
from ..fs_utils import join_path
from ..storage import open_zarr_group, update_zarr_array, write_zarr_group
from ..type_hints import ArrayLike
from .callable_objects import UpdateCallable, WrappedPartitionCallable

#: Partition's type.
PartitionSlice = Tuple[Tuple[str, ...], Dict[str, slice]]


@dataclasses.dataclass(frozen=True)
class PartitioningProperties:
    """Properties of a partition."""
    #: The base directory of the partition.
    dir: str
    #: The name of the partitioning dimension.
    dim: str


def _get_slices(variable: dataset.Variable, dim: str,
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


def _update_with_overlap(
    *args,
    func: UpdateCallable,
    zds: dataset.Dataset,
    indices: slice,
    dim: str,
    fs: fsspec.AbstractFileSystem,
    path: str,
    trim: bool,
    **kwargs,
) -> None:
    """Update a partition with overlap.

    Args:
        func: Function to apply to update each partition.
        zds: Dataset to update.
        indices: Indices of the partition to update.
        dim: Name of the partitioning dimension.
        fs: File system on which the Zarr dataset is stored.
        path: Path to the Zarr group.
        trim: Whether to trim the overlap.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The updated variables.
    """
    dictionary: dict[str, ArrayLike] = (func(
        zds, *args, partition_info=(dim, indices), **
        kwargs) if dask.utils.has_keyword(func, 'partition_info') else func(
            zds, *args, **kwargs))

    if trim:
        for varname, array in dictionary.items():
            slices: tuple[slice, ...] = _get_slices(zds[varname], dim, indices)
            update_zarr_array(
                dirname=join_path(path, varname),
                array=array[slices],  # type: ignore[index]
                fs=fs,
            )
    else:
        tuple(
            map(
                lambda items: update_zarr_array(
                    dirname=join_path(path, items[0]),
                    array=items[1],
                    fs=fs,
                ), dictionary.items()))


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
    zds: dataset.Dataset = open_zarr_group(
        partition, fs, delayed=delayed, selected_variables=selected_variables)
    if immutable:
        zds.merge(
            open_zarr_group(immutable,
                            fs,
                            delayed=delayed,
                            selected_variables=selected_variables))
    return zds


def _load_dataset_with_overlap(
    *,
    delayed: bool,
    depth: int,
    dim: str,
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    partition: str,
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

    def calculate_slice(
        groups: list[dataset.Dataset],
        selected_partitions: list[str],
    ) -> slice:
        """Compute the slice of the selected dataset (without overlap)."""
        start = 0
        indices = slice(0, 0, None)
        for idx, zds in enumerate(groups):
            size: int = zds.dimensions[dim]
            indices = slice(start, start + size, None)
            if partition == selected_partitions[idx]:
                break
            start += size
        return indices

    where: int = partitions.index(partition)

    # Search for the overlapping partitions
    selected_partitions: list[str] = [
        partitions[ix] for ix in range(where - depth, where + depth + 1)
        if 0 <= ix < len(partitions)
    ]

    # Load the datasets for each selected partition.
    groups: list[dataset.Dataset] = [
        open_zarr_group(partition,
                        fs,
                        delayed=delayed,
                        selected_variables=selected_variables)
        for partition in selected_partitions
    ]

    # Compute the slice of the given partition.
    indices: slice = calculate_slice(groups, selected_partitions)

    # Build the dataset for the selected partitions.
    zds: dataset.Dataset = groups.pop(0)
    if groups:
        zds = zds.concat(groups, dim)

    if immutable:
        zds.merge(
            open_zarr_group(partition,
                            fs,
                            delayed=delayed,
                            selected_variables=selected_variables))
    return zds, indices


def _wrap_update_func(
    *args,
    delayed: bool,
    func: UpdateCallable,
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    selected_variables: Iterable[str] | None,
    **kwargs,
) -> WrappedPartitionCallable:
    """Wrap an update function taking a partition's dataset as input and
    returning variable's values as a numpy array.

    Args:
        delayed: Whether to load the dataset lazily.
        func: Function to apply to update each partition.
        fs: File system on which the Zarr dataset is stored.
        immutable: Name of the immutable directory.
        selected_variables: Name of the variables to load from the dataset.
            If None, all variables are loaded.
        trim: Whether to trim the overlap.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The wrapped function that takes a set of dataset partitions and the
        variable name as input and returns the variable's values as a numpy
        array.
    """

    def wrap_function(partitions: Iterable[str]) -> None:
        # Applying function for each partition's data
        for partition in partitions:
            zds: dataset.Dataset = _load_dataset(delayed, fs, immutable,
                                                 partition, selected_variables)
            dictionary: dict[str, ArrayLike] = func(zds, *args, **kwargs)
            tuple(
                update_zarr_array(  # type: ignore[func-returns-value]
                    dirname=join_path(partition, varname),
                    array=array,
                    fs=fs,
                ) for varname, array in dictionary.items())

    return wrap_function


def _wrap_update_func_with_overlap(
    *args,
    delayed: bool,
    depth: int,
    dim: str,
    func: UpdateCallable,
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    selected_partitions: Sequence[str],
    selected_variables: Iterable[str] | None,
    trim: bool,
    **kwargs,
) -> WrappedPartitionCallable:
    """Wrap an update function taking a partition's dataset as input and
    returning variable's values as a numpy array.

    Args:
        delayed: Whether to load the dataset lazily.
        depth: Depth of the overlap.
        dim: Name of the partitioning dimension.
        func: Function to apply to update each partition.
        fs: File system on which the Zarr dataset is stored.
        immutable: Name of the immutable directory.
        selected_partitions: List of all partitions selected for the update.
        selected_variables: Name of the variables to load from the dataset.
            If None, all variables are loaded.
        trim: Whether to trim the overlap.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The wrapped function that takes a set of dataset partitions and the
        variable name as input and returns the variable's values as a numpy
        array.
    """
    if depth < 0:
        raise ValueError('Depth must be non-negative.')

    def wrap_function(partitions: Sequence[str]) -> None:
        # Applying function for each partition's data
        for partition in partitions:

            # pylint: disable=duplicate-code
            # False positive with the method Collection.map_overlap
            zds: dataset.Dataset
            indices: slice
            zds, indices = _load_dataset_with_overlap(
                delayed=delayed,
                depth=depth,
                dim=dim,
                fs=fs,
                immutable=immutable,
                partition=partition,
                partitions=selected_partitions,
                selected_variables=selected_variables)
            # pylint: enable=duplicate-code

            _update_with_overlap(*args,
                                 func=func,
                                 zds=zds,
                                 indices=indices,
                                 dim=dim,
                                 fs=fs,
                                 path=partition,
                                 trim=trim,
                                 **kwargs)

    return wrap_function


def _rm(fs: fsspec.AbstractFileSystem, dirname: str) -> None:
    """Remove a directory and its content.

    Args:
        fs: The file system on which the directory is stored.
        dirname: The name of the directory to remove.
    """
    tries = 0
    while tries < 10:
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
) -> None:
    """Insert or update a partition in the collection.

    Args:
        args: Tuple containing the partition's name and its slice.
        axis: The axis to merge on.
        zds: The dataset to process.
        fs: The file system that the partition is stored on.
        merge_callable: The merge callable.
        partitioning_properties: The partitioning properties.
    """
    partition: tuple[str, ...]
    indexer: dict[str, slice]

    partition, indexer = args
    dirname: str = join_path(*((partitioning_properties.dir, ) + partition))

    # If the consolidated zarr metadata does not exist, we consider the
    # partition as empty.
    if fs.exists(join_path(dirname, '.zmetadata')):
        # The current partition already exists, so we need to merge
        # the dataset.
        merging.perform(zds.isel(indexer),
                        dirname,
                        axis,
                        fs,
                        partitioning_properties.dim,
                        delayed=zds.delayed,
                        merge_callable=merge_callable)
        return

    # The current partition does not exist, so we need to create
    # it and insert the dataset.
    try:
        zarr.storage.init_group(store=fs.get_mapper(dirname))

        # The synchronization is done by the caller.
        write_zarr_group(zds.isel(indexer), dirname, fs, sync.NoSync())
    except:  # noqa: E722
        # If the construction of the new dataset fails, the created
        # partition is deleted, to guarantee the integrity of the
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
        selected_variable: The selected variables to load.

    Returns:
        The list of loaded datasets.
    """
    items: list[slice]
    partition_scheme: tuple[tuple[str, int], ...]

    partition_scheme, items = args
    partition: str = join_path(
        partition_properties.dir,
        partition_handler.join(partition_scheme, fs.sep))
    zds: dataset.Dataset = open_zarr_group(
        partition, fs, delayed=delayed, selected_variables=selected_variables)
    return list(
        zds.isel({partition_properties.dim: indexer}) for indexer in items)
