"""
Implementation details.
=======================
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence
import dataclasses
import sys
import time
import traceback

import dask.utils
import fsspec
import zarr.storage

from .. import dataset, merging, partitioning, sync
from ..fs_utils import join_path
from ..storage import open_zarr_group, update_zarr_array, write_zarr_group
from .callable_objects import UpdateCallable, WrappedPartitionCallable


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
    slices = [slice(None)] * len(variable.dimensions)
    slices[variable.dimensions.index(dim)] = indices
    return tuple(slices)


def try_infer_callable(
    func: Callable,
    ds: dataset.Dataset,
    dim: str,
    *args,
    **kwargs,
) -> Any:
    """Try to call a function with the given arguments.

    Args:
        func: Function to call.
        ds: Dataset to pass to the function.
        dim: Name of the partitioning dimension.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function.
    """
    partition_info = dim, slice(0, ds.dimensions[dim])
    try:
        if dask.utils.has_keyword(func, 'partition_info'):
            return func(ds, *args, partition_info=partition_info, **kwargs)
        return func(ds, *args, **kwargs)
    except Exception as exc:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        execution_tb = ''.join(traceback.format_tb(exc_traceback))
        raise RuntimeError(
            f'An error occurred while applying the function {func} with '
            f'the arguments {args} and {kwargs} to the partition. The '
            f'error is: {exc_type!r}: {exc_value}. The traceback '
            f'is: {execution_tb}') from exc


def update_with_overlap(func: UpdateCallable, ds: dataset.Dataset,
                        indices: slice, dim: str,
                        fs: fsspec.AbstractFileSystem, path: str, *args,
                        **kwargs) -> None:
    """Update a partition with overlap.

    Args:
        func: Function to apply to update each partition.
        ds: Dataset to update.
        indices: Indices of the partition to update.
        dim: Name of the partitioning dimension.
        fs: File system on which the Zarr dataset is stored.
        path: Path to the Zarr group.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The updated variables.
    """
    dictionary = (func(ds, *args, partition_info=(dim, indices), **kwargs)
                  if dask.utils.has_keyword(func, 'partition_info') else func(
                      ds, *args, **kwargs))

    for varname, array in dictionary.items():
        slices = _get_slices(ds[varname], dim, indices)
        update_zarr_array(
            dirname=join_path(path, varname),
            array=array[slices],  # type: ignore[index]
            fs=fs,
        )


def _load_dataset(
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    partition: str,
    selected_variables: Iterable[str] | None,
):
    """Load a dataset from a partition.

    Args:
        fs: File system on which the Zarr dataset is stored.
        immutable: Name of the immutable directory.
        partition: Name of the partition.
        selected_variables: Name of the variables to load from the dataset.

    Returns:
        The loaded dataset.
    """
    ds = open_zarr_group(partition, fs, selected_variables)
    if immutable:
        ds.merge(open_zarr_group(immutable, fs, selected_variables))
    return ds


def _load_dataset_with_overlap(
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
        for ix, ds in enumerate(groups):
            size = ds.dimensions[dim]
            indices = slice(start, start + size, None)
            if partition == selected_partitions[ix]:
                break
            start += size
        return indices

    where = partitions.index(partition)

    # Search for the overlapping partitions
    selected_partitions = [
        partitions[ix] for ix in range(where - depth, where + depth + 1)
        if 0 <= ix < len(partitions)
    ]

    # Load the datasets for each selected partition.
    groups = [
        open_zarr_group(partition, fs, selected_variables)
        for partition in selected_partitions
    ]

    # Compute the slice of the given partition.
    indices = calculate_slice(groups, selected_partitions)

    # Build the dataset for the selected partitions.
    ds = groups.pop(0)
    if groups:
        ds = ds.concat(groups, dim)

    if immutable:
        ds.merge(open_zarr_group(partition, fs, selected_variables))
    return ds, indices


def _wrap_update_func(
    func: UpdateCallable,
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    selected_variables: Iterable[str] | None,
    *args,
    **kwargs,
) -> WrappedPartitionCallable:
    """Wrap an update function taking a partition's dataset as input and
    returning variable's values as a numpy array.

    Args:
        func: Function to apply to update each partition.
        fs: File system on which the Zarr dataset is stored.
        immutable: Name of the immutable directory.
        selected_variables: Name of the variables to load from the dataset.
            If None, all variables are loaded.
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
            ds = _load_dataset(fs, immutable, partition, selected_variables)
            dictionary = func(ds, *args, **kwargs)
            tuple(
                update_zarr_array(  # type: ignore[func-returns-value]
                    dirname=join_path(partition, varname),
                    array=array,
                    fs=fs,
                ) for varname, array in dictionary.items())

    return wrap_function


def _wrap_update_func_with_overlap(
    depth: int,
    dim: str,
    func: UpdateCallable,
    fs: fsspec.AbstractFileSystem,
    immutable: str | None,
    selected_variables: Iterable[str] | None,
    *args,
    **kwargs,
) -> WrappedPartitionCallable:
    """Wrap an update function taking a partition's dataset as input and
    returning variable's values as a numpy array.

    Args:
        func: Function to apply to update each partition.
        fs: File system on which the Zarr dataset is stored.
        selected_variables: Name of the variables to load from the dataset.
            If None, all variables are loaded.
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
            ds, indices = _load_dataset_with_overlap(depth, dim, fs, immutable,
                                                     partition, partitions,
                                                     selected_variables)
            update_with_overlap(func, ds, indices, dim, fs, partition, *args,
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
    args: tuple[tuple[str, ...], dict[str, slice]],
    axis: str,
    ds: dataset.Dataset,
    fs: fsspec.AbstractFileSystem,
    merge_callable: merging.MergeCallable | None,
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
    dirname = join_path(*((partitioning_properties.dir, ) + partition))

    # If the consolidated zarr metadata does not exist, we consider the
    # partition as empty.
    if fs.exists(join_path(dirname, '.zmetadata')):
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
        write_zarr_group(ds.isel(indexer), dirname, fs, sync.NoSync())
    except:  # noqa: E722
        # If the construction of the new dataset fails, the created
        # partition is deleted, to guarantee the integrity of the
        # collection.
        _rm(fs, dirname)
        raise


def _load_and_apply_indexer(
    args: tuple[tuple[tuple[str, int], ...], list[slice]],
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
    partition_scheme, items = args
    partition = join_path(partition_properties.dir,
                          partition_handler.join(partition_scheme, fs.sep))
    ds = open_zarr_group(partition, fs, selected_variables)
    return [ds.isel({partition_properties.dim: indexer}) for indexer in items]
