"""
Implementation details.
=======================
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Sequence
import base64
import dataclasses
import hashlib
import io
import pathlib
import pickle
import warnings

import dask.array.core
import dask.bag.core
import dask.distributed
import fsspec
import numpy
import zarr

from .. import collection, dataset, meta
from ..collection.detail import update_with_overlap
from ..fs_utils import get_fs, join_path
from ..storage import (
    DIMENSIONS,
    open_zarr_array,
    open_zarr_group,
    update_zarr_array,
    write_zattrs,
)

#: Name of the file that contains the checksum of the view.
CHECKSUM_FILE = '.checksum'

#: Name of the attribute that contains the checksum of the view.
CHECKSUM_ATTR = 'checksum'


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
    filesystem: fsspec.AbstractFileSystem = get_fs('file')


@dataclasses.dataclass(frozen=True)
class AxisReference:
    """Properties of the axis used as reference by a view.

    Args:
        array: Axis values.
        dimension: Name of the dimension.
        checksum: Checksum of the axis values.
    """
    #: Axis values.
    array: numpy.ndarray
    #: Name of the dimension.
    dimension: str
    #: Checksum of the axis values.
    checksum: str


def _create_zarr_array(args: tuple[str, zarr.Group],
                       base_dir: str,
                       fs: fsspec.AbstractFileSystem,
                       template: str,
                       variable: meta.Variable,
                       invalidate_cache: bool = True) -> None:
    """Create a Zarr array, with fill_value being used as the default value for
    uninitialized portions of the array.

    Args:
        args: tuple of (path, zarr.Group).
        base_dir: Base directory for the Zarr array.
        fs: The filesystem used to create the Zarr array.
        template: The variable's name is used as a template for the Zarr array
            to determine the shape of the new variable.
        variable: The properties of the variable to create.
        invalidate_cache: If True, invalidate the cache of the directory
            containing the Zarr array.
    """
    partition, group = args
    data: dask.array.core.Array = dask.array.core.from_zarr(group[template])

    dirname = join_path(base_dir, partition)
    mapper = fs.get_mapper(join_path(dirname, variable.name))
    zarr.full(data.shape,
              chunks=data.chunksize,
              dtype=variable.dtype,
              compressor=variable.compressor,
              fill_value=variable.fill_value,
              store=mapper,
              overwrite=True,
              filters=variable.filters)
    write_zattrs(dirname, variable, fs)
    if invalidate_cache:
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
        fs.rm(join_path(partition, variable), recursive=True)
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
    ds = open_zarr_group(
        join_path(view_ref.partition_properties.dir, partition), view_ref.fs,
        selected_variables)
    if ds is None:
        return None

    # If the user has not selected any variables in the reference view. In this
    # case, the dataset is built from all the variables selected in the view.
    if len(ds.dimensions) == 0:
        return dataset.Dataset(
            [
                open_zarr_array(
                    zarr.open(  # type: ignore[arg-type]
                        fs.get_mapper(join_path(base_dir, partition,
                                                variable)),
                        mode='r',
                    ),
                    variable) for variable in variables
            ],
            ds.attrs), partition

    _ = {
        ds.add_variable(item.metadata(), item.array)  # type: ignore[arg-type]
        for item in (
            open_zarr_array(
                zarr.open(  # type: ignore[arg-type]
                    fs.get_mapper(join_path(base_dir, partition, variable)),
                    mode='r',
                ),
                variable) for variable in variables)
    }

    # Apply indexing if needed.
    if len(slices):
        dim = view_ref.partition_properties.dim
        ds_list = [ds.isel({dim: indexer}) for indexer in slices]
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
            size = ds.dimensions[view_ref.partition_properties.dim]
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
                update_zarr_array(  # type: ignore[func-returns-value]
                    dirname=join_path(base_dir, partition, varname),
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

    if depth < 0:
        raise ValueError('The depth must be positive')

    def wrap_function(parameters: Iterable[tuple[dataset.Dataset, str]],
                      base_dir: str) -> None:
        """Wrap the function to be applied to the dataset."""
        for ds, partition in parameters:
            ds, indices = _select_overlap((ds, partition), datasets_list,
                                          depth, view_ref)
            update_with_overlap(func, ds, indices, dim, fs,
                                join_path(base_dir, partition), *args,
                                **kwargs)

    return wrap_function


def _calculate_axis_reference(
        path: str,
        view_ref: collection.Collection,
        group: zarr.Group | None = None) -> AxisReference:
    """Compute the axis reference of a partition (checksum, array and dimension
    name).

    Args:
        path: The path of the partition.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
        group: The group to compute the checksum of (if None, the group is
            reload from the file system).

    Returns:
        The axis reference of the partition.
    """
    store = group or zarr.open_consolidated(  # type: ignore[arg-type]
        view_ref.fs.get_mapper(path), )
    array: zarr.Array = store[view_ref.axis]  # type: ignore[arg-type]
    axis = array[...]
    checksum = hashlib.sha256(axis.tobytes()).hexdigest()
    return AxisReference(axis, array.attrs[DIMENSIONS][0], checksum)


def _write_checksum_array(
    partition: str,
    fs: fsspec.AbstractFileSystem,
    axis_ref: AxisReference,
) -> None:
    """Write the checksum of a partition to a file.

    Args:
        partition: The path of the partition.
        fs: The file system used to access the variables in the view.
        axis_ref: The axis reference of the partition.
    """
    checksum_path = join_path(partition, CHECKSUM_FILE)
    mapper = fs.get_mapper(checksum_path)
    if fs.exists(checksum_path):
        array = zarr.open(mapper)
    else:
        array = zarr.create(shape=axis_ref.array.shape,
                            dtype=axis_ref.array.dtype,
                            store=mapper)
        array.attrs[DIMENSIONS] = axis_ref.dimension
    array[...] = axis_ref.array
    array.attrs[CHECKSUM_ATTR] = axis_ref.checksum
    fs.invalidate_cache(checksum_path)


def _load_checksum_array(
    partition: str,
    fs: fsspec.AbstractFileSystem,
) -> zarr.Array:
    """Load the checksum of a partition from a file.

    Args:
        partition: The path of the partition.
        fs: The file system used to access the variables in the view.

    Returns:
        The checksum of the partition and the axis of the reference partition.
    """
    checksum_path = join_path(partition, CHECKSUM_FILE)
    mapper = fs.get_mapper(checksum_path)
    return zarr.open_array(mapper)


def _write_checksum(
    partition: str,
    base_dir: str,
    view_ref: collection.Collection,
    fs: fsspec.AbstractFileSystem,
    group: zarr.Group | None = None,
) -> None:
    """Write the checksum of a partition to a file.

    Args:
        partition: The path of the partition.
        base_dir: The base directory of the view.
        view_ref: The view reference.
        fs: The file system used to access the variables in the view.
        group: The group to compute the checksum of (if None, the group is
            reload from the file system).
    """
    partition_ref = join_path(
        view_ref.partition_properties.dir,
        str(pathlib.Path(partition).relative_to(base_dir).as_posix()))
    _write_checksum_array(
        partition, fs,
        _calculate_axis_reference(partition_ref, view_ref, group=group))


def _sync_partition(
    metadata: meta.Dataset,
    partition: str,
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
) -> None:
    """Sync a partition: create the partition and the underlying variables.

    Args:
        metadata: The dataset to sync.
        partition: The partition to sync.
        base_dir: The base directory of the view.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
    """
    mapper = view_ref.fs.get_mapper(
        join_path(view_ref.partition_properties.dir, partition))
    group: zarr.Group = zarr.open_consolidated(  # type: ignore
        mapper, mode='r')
    path = join_path(base_dir, partition)
    try:
        for variable in metadata.variables.values():
            template = view_ref.metadata.search_same_dimensions_as(variable)
            _create_zarr_array((partition, group),
                               base_dir,
                               fs,
                               template.name,
                               variable,
                               invalidate_cache=False)
        _write_checksum(path, base_dir, view_ref, fs, group)
        fs.invalidate_cache(path)

    except Exception as exc:
        fs.rm(path, recursive=True)
        fs.invalidate_cache(path)
        raise exc from None


def _extend_partition(
    partition: str,
    fs: fsspec.AbstractFileSystem,
    axis_ref: AxisReference,
) -> None:
    """Sync a partition: create the partition and the underlying variables.

    Args:
        partition: The partition to sync.
        fs: The file system used to access the variables in the view.
        axis_ref: The axis reference of the partition.
    """
    axis_name = axis_ref.dimension
    new_size = axis_ref.array.shape[0]
    try:
        for variable in fs.listdir(partition):
            array = zarr.open_array(fs.get_mapper(variable['name']))
            dimensions = array.attrs[DIMENSIONS]
            if axis_name in dimensions:
                axis = dimensions.index(axis_name)
                shape = list(array.shape)
                shape[axis] = new_size
                array.resize(shape)
        _write_checksum_array(partition, fs, axis_ref)
        # fs.invalidate_cache(partition) is not done by
        # _write_checksum_array
    except Exception as exc:
        fs.rm(partition, recursive=True)
        fs.invalidate_cache(partition)
        raise exc from None


def _sync(
    partition: str,
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    metadata: meta.Dataset,
    dry_run: bool = False,
) -> str | None:
    """Sync the partitions of a view.

    Args:
        partition: The partition to sync.
        base_dir: The base directory of the view.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
        metadata: The dataset to sync.
        dry_run: If True, the partition is not synced.

    Returns:
        The partition synced or None if the partition is already synced.
    """
    partition_view = join_path(base_dir, partition)
    if not fs.exists(partition_view):
        # The partition does not exist, so we create it.
        if not dry_run:
            _sync_partition(metadata, partition, base_dir, fs, view_ref)
        return partition

    # The partition exists, so we check if it is synced.
    partition_ref = join_path(view_ref.partition_properties.dir, partition)
    array = _load_checksum_array(partition_view, fs)
    axis_ref = _calculate_axis_reference(partition_ref, view_ref)
    # If the checksums are different, the partition is not synced. So we
    # remove it (information between the partition and the reference are
    # not consistent)
    if axis_ref.checksum != array.attrs[CHECKSUM_ATTR]:
        # If the checksums are different, the partition is not synced. we load
        # the axis of partition's view to check if the partition is a subset of
        # the reference partition.
        axis_view = array[:]
        if axis_ref.array.size > axis_view.size and numpy.all(
                axis_view == axis_ref.array[:axis_view.size]):
            # The view partition is a subset of the reference partition.
            # So we extend the view partition to the reference partition.
            # fs_invalid_cache is done by _extend_partition
            if not dry_run:
                _extend_partition(partition_view, fs, axis_ref)
            return partition
        # The partition is not synced, so we remove it.
        if not dry_run:
            fs.rm(partition_view, recursive=True)
            fs.invalidate_cache(partition_view)
            _sync_partition(metadata, partition, base_dir, fs, view_ref)
        return partition
    return None


def _serialize_filters(filters: collection.PartitionFilter, ) -> str:
    """Serialize a partition filter.

    Args:
        filters: The partition filter to serialize.

    Returns:
        The serialized partition filter.
    """
    stream = io.BytesIO()
    try:
        pickle.dump(filters, stream)
    except (pickle.PicklingError, AttributeError):
        warnings.warn(
            'The partition filter cannot be serialized, it will be ignored.',
            UserWarning,
            stacklevel=2)
        pickle.dump(None, stream)
    return base64.b64encode(stream.getvalue()).decode('utf-8')


def _deserialize_filters(filters: str) -> collection.PartitionFilter:
    """Unserialize a partition filter.

    Args:
        filters: The partition filter to unserialize.

    Returns:
        The unserialized partition filter.
    """
    stream = io.BytesIO(base64.b64decode(filters.encode('utf-8')))
    try:
        return pickle.load(stream)
    except pickle.UnpicklingError:
        return None
