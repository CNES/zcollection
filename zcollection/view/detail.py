"""
Implementation details.
=======================
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Sequence
import dataclasses

import dask.array.core
import dask.distributed
import fsspec
import zarr

from .. import collection, dataset, meta, storage, utilities
from ..collection.detail import update_with_overlap


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


def _create_zarr_array(args: tuple[str, zarr.Group], base_dir: str,
                       fs: fsspec.AbstractFileSystem, template: str,
                       variable: meta.Variable) -> None:
    """Create a Zarr array, with fill_value being used as the default value for
    uninitialized portions of the array.

    Args:
        args: tuple of (path, zarr.Group).
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
    args: tuple[tuple[tuple[str, int], ...], list[slice]],
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    selected_variables: Iterable[str] | None,
    view_ref: collection.Collection,
    variables: Sequence[str],
) -> tuple[dataset.Dataset, str] | None:
    """Load a dataset from a partition stored in the reference collection and
    merge it with the variables defined in this view.

    Args:
        args: tuple containing the partition's keys and its indexer.
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
        ds_list: list[dataset.Dataset] = []
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
    selected_variables: Iterable[str] | None = None,
) -> Iterator[tuple[dataset.Dataset, str]]:
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
    arguments: tuple[tuple[tuple[tuple[str, int], ...], list], ...] = tuple(
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


def _select_overlap(
    arguments: tuple[dataset.Dataset, str],
    datasets_list: tuple[tuple[dataset.Dataset, str], ...],
    depth: int,
    view_ref: collection.Collection,
) -> tuple[dataset.Dataset, slice]:
    """Select the neighboring partitions of a given partition.

    Args:
        arguments: The partition to select the neighbors of.
        datasets_list: The list of partitions.
        depth: The depth of the overlap.
        view_ref: The view reference.

    Returns:
        The neighboring partitions.
    """

    def calculate_slice(
            selected_datasets: list[tuple[dataset.Dataset, str]]) -> slice:
        """Compute the slice of the selected dataset (without overlap)."""
        start = 0
        indices = slice(0, 0, None)
        for ds, selected_partition in selected_datasets:
            size = ds[view_ref.axis].shape[0]
            indices = slice(start, start + size, None)
            if partition == selected_partition:
                break
            start += size
        return indices

    # The local function is not taken into account for counting
    # locals.
    _, partition = arguments
    where = next(ix for ix, item in enumerate(datasets_list)
                 if item[1] == partition)

    # Search for the overlapping partitions
    selected_datasets = [
        datasets_list[ix] for ix in range(where - depth, where + depth + 1)
        if 0 <= ix < len(datasets_list)
    ]

    # Build the dataset for the selected partitions.
    groups = [ds for ds, _ in selected_datasets]
    ds = groups.pop(0)
    ds = ds.concat(groups, view_ref.partition_properties.dim)

    return ds, calculate_slice(selected_datasets)


def _wrap_update_func(
    func: collection.UpdateCallable,
    fs: fsspec.AbstractFileSystem,
    *args,
    **kwargs,
) -> Callable[[Iterable[tuple[dataset.Dataset, str]], str], None]:
    """Wrap an update function taking a list of partition's dataset and
    partition's path as input and returning None.

    Args:
        func: The update function.
        fs: The file system used to access the variables in the view.
        *args: The arguments of the update function.
        **kwargs: The keyword arguments of the update function.

    Returns:
        The wrapped function.
    """

    def wrap_function(parameters: Iterable[tuple[dataset.Dataset, str]],
                      base_dir: str) -> None:
        """Wrap the function to be applied to the dataset."""
        for ds, partition in parameters:
            # Applying function on partition's data
            dictionary = func(ds, *args, **kwargs)
            tuple(
                storage.update_zarr_array(  # type: ignore[func-returns-value]
                    dirname=fs.sep.join((base_dir, partition, varname)),
                    array=array,
                    fs=fs,
                ) for varname, array in dictionary.items())

    return wrap_function


def _wrap_update_func_overlap(
    datasets_list: tuple[tuple[dataset.Dataset, str], ...],
    depth: int,
    func: collection.UpdateCallable,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    *args,
    **kwargs,
) -> Callable[[Iterable[tuple[dataset.Dataset, str]], str], None]:
    """Wrap an update function taking a list of partition's dataset and
    partition's path as input and returning None.

    Args:
        datasets_list: The list of datasets and their paths.
        depth: The depth of the overlap.
        func: The update function.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
        *args: The arguments of the update function.
        **kwargs: The keyword arguments of the update function.

    Returns:
        The wrapped function.
    """
    dim = view_ref.partition_properties.dim

    def wrap_function(parameters: Iterable[tuple[dataset.Dataset, str]],
                      base_dir: str) -> None:
        """Wrap the function to be applied to the dataset."""
        for ds, partition in parameters:
            ds, indices = _select_overlap((ds, partition), datasets_list,
                                          depth, view_ref)
            update_with_overlap(func, ds, indices, dim, fs,
                                fs.sep.join((base_dir, partition)), *args,
                                **kwargs)

    return wrap_function
