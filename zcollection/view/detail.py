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
import base64
from collections.abc import Callable, Iterable, Iterator, Sequence
import dataclasses
import hashlib
import io
import logging
import pathlib
import pickle
import warnings

import dask.array.core
import dask.bag.core
import numpy
import zarr

from .. import collection, dataset, meta
from ..collection.detail import _update_with_overlap
from ..fs_utils import get_fs, join_path
from ..storage import (
    DIMENSIONS,
    open_zarr_array,
    open_zarr_group,
    update_zarr_array,
    variable_shape,
    write_zattrs,
)

if TYPE_CHECKING:
    import dask.distributed
    import fsspec

    from ..type_hints import ArrayLike, NDArray

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)

#: Type of the function used to update a view.
ViewUpdateCallable = Callable[[
    Iterable[tuple[dataset.Dataset, str]], str, tuple[Any, ...], dict[str, Any]
], None]

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
    filesystem: fsspec.AbstractFileSystem = dataclasses.field(
        default_factory=lambda: get_fs('file'))


@dataclasses.dataclass(frozen=True)
class AxisReference:
    """Properties of the axis used as reference by a view.

    Args:
        array: Axis values.
        dimension: Name of the dimension.
        checksum: Checksum of the axis values.
    """
    #: Axis values.
    array: NDArray
    #: Name of the dimension.
    dimension: str
    #: Checksum of the axis values.
    checksum: str


def _create_zarr_array(args: tuple[str, str],
                       *,
                       base_dir: str,
                       variable: meta.Variable,
                       dimensions: meta.DimensionType,
                       chunks: dict[str, int] | None,
                       axis: str | None,
                       fs: fsspec.AbstractFileSystem,
                       invalidate_cache: bool = True) -> None:
    """Create a Zarr array, with fill_value being used as the default value for
    uninitialized portions of the array.

    Args:
        args: tuple of (relative path, absolute path).
        base_dir: Base directory for the Zarr array.
        fs: The filesystem used to create the Zarr array.
        dimensions: Known dimensions and their size.
        variable: The properties of the variable to create.
        axis: Axis containing the main dimension data and size.
        invalidate_cache: If True, invalidate the cache of the directory
            containing the Zarr array.
    """
    partition, partition_ref = args
    dirname = join_path(base_dir, partition)

    _LOGGER.debug('Adding variable %r to Zarr dataset %r', variable.name,
                  dirname)
    var_shape = variable_shape(
        variable=variable,
        dimensions=dimensions,
        axis=join_path(partition_ref, axis) if axis is not None else None,
        fs=fs)

    var_chunks: tuple[int, ...] = var_shape if chunks is None else tuple(
        chunks.get(dim, var_shape[ix])
        for ix, dim in enumerate(variable.dimensions))

    store: fsspec.FSMap = fs.get_mapper(join_path(dirname, variable.name))
    zarr.full(var_shape,
              chunks=var_chunks,
              dtype=variable.dtype,
              compressor=variable.compressor,
              fill_value=variable.fill_value,
              store=store,
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
        fs: The filesystem used to delete the Zarr array.
        variable: The name of the variable to drop.
        ignore_errors: If True, ignore errors when dropping the array.
    """
    try:
        fs.rm(join_path(partition, variable), recursive=True)
        fs.invalidate_cache(partition)

    # We don't want to fail on errors.
    except Exception:
        if not ignore_errors:
            raise


def _load_one_dataset(
    args: tuple[tuple[tuple[str, int], ...], list[slice]],
    *,
    base_dir: str,
    delayed: bool,
    fs: fsspec.AbstractFileSystem,
    selected_variables: Iterable[str] | None,
    view_ref: collection.Collection,
    variables: Iterable[str],
    with_immutable: bool = False,
) -> tuple[dataset.Dataset, str] | None:
    """Load a dataset from a partition stored in the reference collection and
    merge it with the variables defined in this view.

    Args:
        args: tuple containing the partition's keys and its indexer.
        base_dir: Base directory of the view.
        delayed: If True, load the dataset lazily.
        fs: The file system used to access the variables in the view.
        selected_variables: The list of variable to retain from the view
            reference.
        view_ref: The view reference.
        variables: The variables to retain from the view
        with_immutable: Whether to include immutable variables or not.

    Returns:
        The dataset and the partition's path.
    """
    partition_scheme: tuple[tuple[str, int], ...]
    slices: list[slice]

    partition_scheme, slices = args
    partition: str = view_ref.partitioning.join(partition_scheme, fs.sep)
    zds: dataset.Dataset = open_zarr_group(
        join_path(view_ref.partition_properties.dir, partition),
        view_ref.fs,
        delayed=delayed,
        selected_variables=selected_variables)

    if zds is None:
        # No data for this partition
        return None

    # Filling missing dimensions
    mds = view_ref.metadata
    missing_dimensions = set(
        mds.dimensions) - {*zds.dimensions, view_ref.dimension}
    for name in missing_dimensions:
        zds.dimensions[name] = mds.dimensions[name].value

    data = list(zds.variables.values()) + [
        open_zarr_array(
            zarr.open(  # type: ignore[arg-type]
                fs.get_mapper(join_path(base_dir, partition, variable)), 'r'),
            variable,
            delayed=delayed) for variable in variables
    ]

    zds = dataset.Dataset(data, attrs=zds.attrs, delayed=zds.delayed)

    # Adding immutable variables
    if with_immutable and view_ref.have_immutable:
        zds.merge(
            open_zarr_group(view_ref.immutable_path,
                            fs,
                            delayed=delayed,
                            selected_variables=selected_variables))

    # Apply indexing if needed.
    if len(slices):
        ds_list: list[dataset.Dataset] = [
            zds.isel({view_ref.dimension: indexer}) for indexer in slices
        ]
        zds = ds_list.pop(0)
        if ds_list:
            zds = zds.concat(ds_list, view_ref.dimension)

    return zds, partition


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
    *,
    client: dask.distributed.Client | None,
    base_dir: str,
    delayed: bool,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    metadata: meta.Dataset,
    partitions: Iterable[str],
    selected_variables: Iterable[str] | None = None,
    with_immutable: bool = False,
) -> Iterator[tuple[dataset.Dataset, str]]:
    """Load datasets from a list of partitions.

    Args:
        client: The client used to load the datasets (or None to
            avoid dask usage).
        base_dir: Base directory of the view.
        delayed: If True, load the dataset lazily.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
        metadata: The metadata of the dataset.
        partitions: The list of partitions to load.
        selected_variables: The list of variable to retain from the view
        with_immutable: Whether to include immutable variables or not.

    Returns:
        The datasets and their paths.
    """
    arguments: tuple[tuple[tuple[tuple[str, int], ...], list], ...] = tuple(
        (view_ref.partitioning.parse(item), []) for item in partitions)

    datasets: list[tuple[dataset.Dataset, str] | None]

    if client is not None:
        futures: list[dask.distributed.Future] = client.map(
            _load_one_dataset,
            arguments,
            base_dir=base_dir,
            delayed=delayed,
            fs=fs,
            selected_variables=view_ref.metadata.select_variables(
                selected_variables),
            view_ref=client.scatter(view_ref),
            variables=metadata.select_variables(selected_variables),
            with_immutable=with_immutable)
        datasets = client.gather(futures)
    else:
        datasets = [
            _load_one_dataset(
                arg,
                base_dir=base_dir,
                delayed=False,
                fs=fs,
                selected_variables=view_ref.metadata.select_variables(
                    selected_variables),
                view_ref=view_ref,
                variables=metadata.select_variables(selected_variables),
                with_immutable=with_immutable) for arg in arguments
        ]
    return filter(
        lambda item: item is not None,  # type: ignore[arg-type]
        datasets)


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
        for zds, selected_partition in selected_datasets:
            size = zds.dimensions[view_ref.dimension]
            indices = slice(start, start + size, None)
            if partition == selected_partition:
                break
            start += size
        return indices

    # The local function is not taken into account for counting
    # locals.
    partition: str = arguments[1]
    where: int = next(ix for ix, item in enumerate(datasets_list)
                      if item[1] == partition)

    # Search for the overlapping partitions
    selected_datasets: list[tuple[dataset.Dataset, str]] = [
        datasets_list[ix] for ix in range(where - depth, where + depth + 1)
        if 0 <= ix < len(datasets_list)
    ]

    # Build the dataset for the selected partitions.
    groups: list[dataset.Dataset] = [ds for ds, _ in selected_datasets]
    zds: dataset.Dataset = groups.pop(0)
    zds = zds.concat(groups, view_ref.dimension)

    return zds, calculate_slice(selected_datasets)


def _wrap_update_func(
    func: collection.UpdateCallable,
    fs: fsspec.AbstractFileSystem,
) -> ViewUpdateCallable:
    """Wrap an update function taking a list of partition's dataset and
    partition's path as input and returning None.

    Args:
        func: The update function.
        fs: The file system used to access the variables in the view.

    Returns:
        The wrapped function.
    """

    def wrap_function(parameters: Iterable[tuple[dataset.Dataset, str]],
                      base_dir: str, func_args: tuple[Any, ...],
                      func_kwargs: dict[str, Any]) -> None:
        """Wrap the function to be applied to the dataset."""
        for zds, partition in parameters:
            # Applying function on partition's data
            dictionary: dict[str, ArrayLike] = func(zds, *func_args,
                                                    **func_kwargs)
            tuple(
                update_zarr_array(  # type: ignore[func-returns-value]
                    join_path(base_dir, partition, varname), array, fs)
                for varname, array in dictionary.items())

    return wrap_function


def _wrap_update_func_overlap(
    datasets_list: tuple[tuple[dataset.Dataset, str], ...],
    depth: int,
    func: collection.UpdateCallable,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    trim: bool,
) -> ViewUpdateCallable:
    """Wrap an update function taking a list of partition's dataset and
    partition's path as input and returning None.

    Args:
        datasets_list: The list of datasets and their paths.
        depth: The depth of the overlap.
        func: The update function.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
        trim: If True, trim the dataset to the overlap.

    Returns:
        The wrapped function.
    """
    dim = view_ref.dimension

    if depth < 0:
        raise ValueError('The depth must be positive')

    def wrap_function(parameters: Iterable[tuple[dataset.Dataset, str]],
                      base_dir: str, func_args: tuple[Any, ...],
                      func_kwargs: dict[str, Any]) -> None:
        """Wrap the function to be applied to the dataset."""
        zds: dataset.Dataset
        indices: slice

        for zds, partition in parameters:
            selected_zds, indices = _select_overlap(
                (zds, partition), datasets_list, depth, view_ref)

            # False positive with the function _wrap_update_func_with_overlap
            # defined in the module zcollection.collection.detail
            _update_with_overlap(*func_args,
                                 func=func,
                                 zds=selected_zds,
                                 indices=indices,
                                 dim=dim,
                                 fs=fs,
                                 path=join_path(base_dir, partition),
                                 trim=trim,
                                 **func_kwargs)

    return wrap_function


def _calculate_axis_reference(
        path: str, view_ref: collection.Collection) -> AxisReference:
    """Compute the axis reference of a partition (checksum, array and dimension
    name).

    Args:
        path: The path of the partition.
        view_ref: The view reference.

    Returns:
        The axis reference of the partition.
    """
    store: zarr.Group = zarr.open_consolidated(view_ref.fs.get_mapper(path))
    array: zarr.Array = store[view_ref.axis]  # type: ignore[arg-type]
    axis: NDArray = array[...]
    checksum: str = hashlib.sha256(axis.tobytes()).hexdigest()
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
    array: zarr.Array
    checksum_path = join_path(partition, CHECKSUM_FILE)
    mapper: fsspec.FSMap = fs.get_mapper(checksum_path)
    if fs.exists(checksum_path):
        array = zarr.open(mapper)  # type: ignore[arg-type]
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
    mapper: fsspec.FSMap = fs.get_mapper(checksum_path)
    return zarr.open_array(mapper)


def _write_checksum(
    partition: str,
    base_dir: str,
    view_ref: collection.Collection,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Write the checksum of a partition to a file.

    Args:
        partition: The path of the partition.
        base_dir: The base directory of the view.
        view_ref: The view reference.
        fs: The file system used to access the variables in the view.
    """
    partition_ref = join_path(
        view_ref.partition_properties.dir,
        str(pathlib.Path(partition).relative_to(base_dir).as_posix()))
    _write_checksum_array(
        partition,
        fs,
        _calculate_axis_reference(partition_ref, view_ref),
    )


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
    path = join_path(base_dir, partition)

    dimensions, chunks = view_ref.dimensions_properties()

    try:
        for variable in metadata.variables.values():
            _create_zarr_array(
                (partition,
                 join_path(view_ref.partition_properties.dir, partition)),
                base_dir=base_dir,
                variable=variable,
                dimensions=dimensions,
                chunks=chunks,
                axis=view_ref.axis,
                fs=fs,
                invalidate_cache=False)
        _write_checksum(path, base_dir, view_ref, fs)
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
    axis_name: str = axis_ref.dimension
    new_size: int = axis_ref.array.shape[0]
    try:
        for variable in fs.listdir(partition):
            array: zarr.Array = zarr.open_array(fs.get_mapper(
                variable['name']))
            dimensions: Sequence[str] = array.attrs[DIMENSIONS]
            if axis_name in dimensions:
                axis: int = dimensions.index(axis_name)
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
    *,
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
    array: zarr.Array = _load_checksum_array(partition_view, fs)
    axis_ref: AxisReference = _calculate_axis_reference(path=partition_ref,
                                                        view_ref=view_ref)
    # If the checksums are different, the partition is not synced. So we
    # remove it (information between the partition and the reference are
    # not consistent)
    if axis_ref.checksum != array.attrs[CHECKSUM_ATTR]:
        # If the checksums are different, the partition is not synced. we load
        # the axis of partition's view to check if the partition is a subset of
        # the reference partition.
        axis_view: NDArray = array[:]
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
    """Deserialize a partition filter.

    Args:
        filters: The partition filter to deserialize.

    Returns:
        The deserialized partition filter.
    """
    stream = io.BytesIO(base64.b64decode(filters.encode('utf-8')))
    try:
        return pickle.load(stream)
    except pickle.UnpicklingError:
        return None
