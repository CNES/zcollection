"""
Implementation details.
=======================
"""
from __future__ import annotations

from typing import Iterable, Sequence
import dataclasses

import fsspec
import zarr.storage

from .. import dataset, merging, partitioning, storage, sync
from .callable_objects import UpdateCallable, WrappedPartitionCallable


@dataclasses.dataclass(frozen=True)
class PartitioningProperties:
    """Properties of a partition."""
    #: The base directory of the partition.
    dir: str
    #: The name of the partitioning dimension.
    dim: str


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
    ds = storage.open_zarr_group(partition, fs, selected_variables)
    if immutable:
        ds.merge(storage.open_zarr_group(immutable, fs, selected_variables))
    return ds


def _load_dataset_with_overlap(
    axis: str,
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
        axis: The axis of the collection.
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
            size = ds[axis].size
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
        storage.open_zarr_group(partition, fs, selected_variables)
        for partition in selected_partitions
    ]

    # Compute the slice of the given partition.
    indices = calculate_slice(groups, selected_partitions)

    # Build the dataset for the selected partitions.
    ds = groups.pop(0)
    ds = ds.concat(groups, dim)

    if immutable:
        ds.merge(storage.open_zarr_group(partition, fs, selected_variables))
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
                storage.update_zarr_array(  # type: ignore[func-returns-value]
                    dirname=fs.sep.join((partition, varname)),
                    array=array,
                    fs=fs,
                ) for varname, array in dictionary.items())

    return wrap_function


def _wrap_update_func_with_overlap(
    axis: str,
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

    def wrap_function(partitions: Sequence[str]) -> None:
        # Applying function for each partition's data
        for partition in partitions:

            ds, indices = _load_dataset_with_overlap(axis, depth, dim, fs,
                                                     immutable, partition,
                                                     partitions,
                                                     selected_variables)
            dictionary = func(ds, *args, **kwargs)

            for varname, array in dictionary.items():
                slices = tuple(indices if dimname == dim else slice(None)
                               for dimname, _ in ds[varname].dimension_index())
                storage.update_zarr_array(
                    dirname=fs.sep.join((partition, varname)),
                    array=array[slices],  # type: ignore[index]
                    fs=fs,
                )

    return wrap_function


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
    dirname = fs.sep.join((partitioning_properties.dir, ) + partition)

    # If the consolidated zarr metadata does not exist, we consider the
    # partition as empty.
    if fs.exists(fs.sep.join((dirname, '.zmetadata'))):
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
    partition = fs.sep.join((partition_properties.dir,
                             partition_handler.join(partition_scheme, fs.sep)))

    ds = storage.open_zarr_group(partition, fs, selected_variables)
    arrays = []
    _ = {
        arrays.append(  # type: ignore[func-returns-value]
            ds.isel({partition_properties.dim: indexer}))
        for indexer in items
    }
    return arrays
