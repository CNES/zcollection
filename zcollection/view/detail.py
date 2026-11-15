# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Implementation details.
=======================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import base64
import contextlib
import dataclasses
import hashlib
import io
import logging
import os
import pathlib
import pickle
import warnings

import numpy

from .. import collection, dataset, meta, store
from ..collection.callable_objects import WrappedPartitionCallable
from ..execution import SKIPPED, Skipped
from ..fs_utils import get_fs, join_path
from ..storage import (
    AbsentPartition,
    open_zarr_array,
    open_zarr_group,
    partition_dimension_size,
    read_partition,
    variable_attributes,
    variable_shape,
    warn_absent_partition,
)
from ..variable.abc import read_dimensions

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import fsspec
    import zarr

    from ..type_hints import NDArray

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)

#: Type of the function used to update a view.
ViewUpdateCallable = WrappedPartitionCallable

#: Name of the Zarr array storing the reference axis and its checksum.
CHECKSUM_FILE = '.checksum'

#: Name of the attribute storing the reference axis checksum.
CHECKSUM_ATTR = 'checksum'


@dataclasses.dataclass(frozen=True)
class ViewReference:
    """Access properties of the reference collection of a view.

    Args:
        path: Path to the reference collection.
        filesystem: The file system used to access the reference collection.
    """
    #: Path to the collection.
    path: str
    #: The file system used to access the reference collection.
    filesystem: fsspec.AbstractFileSystem = dataclasses.field(
        default_factory=lambda: get_fs('file'))


@dataclasses.dataclass(frozen=True)
class AxisReference:
    """Reference axis values and their checksum for a view.

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
                       variable: meta.VariableSchema,
                       dimensions: meta.DimensionType,
                       chunks: dict[str, int] | None,
                       axis: str | None,
                       fs: fsspec.AbstractFileSystem,
                       reference_fs: fsspec.AbstractFileSystem,
                       invalidate_cache: bool = True) -> None:
    """Create a Zarr array initialized with the variable's fill value.

    Args:
        args: The path relative to the view and the corresponding
            reference partition path.
        base_dir: Base directory of the view.
        fs: The file system on which to create the Zarr array.
        reference_fs: The file system storing the reference axis.
        dimensions: Known dimension sizes.
        variable: The properties of the variable to create.
        axis: Reference axis used to determine the partitioned size.
        chunks: Chunk sizes for the new array.
        invalidate_cache: Whether to invalidate the array directory cache.
    """
    partition, partition_ref = args
    dirname = join_path(base_dir, partition)

    _LOGGER.debug('Adding variable %r to Zarr dataset %r', variable.name,
                  dirname)
    var_shape = variable_shape(
        variable=variable,
        dimensions=dimensions,
        axis=join_path(partition_ref, axis) if axis is not None else None,
        fs=reference_fs)

    var_chunks: tuple[int, ...] = var_shape if chunks is None else tuple(
        chunks.get(dim, var_shape[ix])
        for ix, dim in enumerate(variable.dimensions))

    store.create_array(fs,
                       join_path(dirname, variable.name),
                       shape=var_shape,
                       chunks=var_chunks,
                       dtype=variable.dtype,
                       compressor=variable.compressor,
                       fill_value=variable.fill_value,
                       overwrite=True,
                       filters=variable.filters,
                       dimensions=variable.dimensions,
                       attributes=variable_attributes(variable))
    if invalidate_cache:
        fs.invalidate_cache(dirname)


def _add_array(args: tuple[str, str], *, base_dir: str,
               fs: fsspec.AbstractFileSystem, **kwargs: Any) -> Skipped | None:
    """Create the array of a variable in a partition of a view, skipping an
    absent partition.

    Args:
        args: The path of the partition relative to the view, and the path
            of the partition of the reference collection.
        base_dir: The base directory of the view.
        fs: The file system of the view.
        **kwargs: The other arguments of :func:`_create_zarr_array`.
    """
    if not fs.exists(join_path(base_dir, args[0])):
        return SKIPPED
    _create_zarr_array(args, base_dir=base_dir, fs=fs, **kwargs)
    return None


def _drop_array(partition: str, *, fs: fsspec.AbstractFileSystem,
                variable: str) -> Skipped | None:
    """Remove the array of a variable from a partition of a view, skipping
    an absent partition. Removing an absent array succeeds."""
    if not fs.exists(partition):
        return SKIPPED
    with contextlib.suppress(FileNotFoundError):
        fs.rm(join_path(partition, variable), recursive=True)
    fs.invalidate_cache(partition)
    return None


def _held_one_dataset(
    args: tuple[tuple[tuple[str, int], ...], list[slice]],
    **kwargs,
) -> tuple[dataset.Dataset, str] | None:
    """Load a view partition, returning ``None`` if it is absent.

    This function identifies partitions by their keys;
    :func:`zcollection.storage.skip_absent_partition` expects paths,
    so its wrapper cannot be used here.

    Args:
        args: The partition keys and indexer.
        **kwargs: The arguments of :func:`_load_one_dataset`.

    Returns:
        The dataset and partition path, or ``None`` if a required
        partition is absent. Its absence is logged as a warning.
    """
    try:
        return _load_one_dataset(args, **kwargs)
    except AbsentPartition as err:
        warn_absent_partition(str(err))
        return None


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
) -> tuple[dataset.Dataset, str]:
    """Load a reference partition and merge in the variables of the view.

    Args:
        args: The partition keys and indexer.
        base_dir: Base directory of the view.
        delayed: If True, load the dataset lazily.
        fs: The file system used to access the variables in the view.
        selected_variables: Variables to retain from the reference collection.
        view_ref: Access information for the reference collection.
        variables: Variables to retain from the view.
        with_immutable: Whether to include immutable variables or not.

    Returns:
        The dataset and the partition's path.
    """
    partition_scheme: tuple[tuple[str, int], ...]
    slices: list[slice]

    partition_scheme, slices = args
    partition: str = view_ref.partitioning.join(partition_scheme, fs.sep)
    held = join_path(base_dir, partition)
    variables = tuple(variables)
    if not variables and not fs.exists(held):
        raise AbsentPartition(held)
    zds: dataset.Dataset = read_partition(
        join_path(view_ref.partition_properties.dir, partition),
        view_ref.fs,
        delayed=delayed,
        selected_variables=selected_variables)

    # Add missing dimensions.
    mds = view_ref.metadata
    missing_dimensions = set(
        mds.dimensions) - {*zds.dimensions, view_ref.dimension}
    for name in missing_dimensions:
        zds.dimensions[name] = mds.dimensions[name].value

    try:
        data = list(zds.variables.values()) + [
            open_zarr_array(store.open_array(
                fs, join_path(held, variable), mode='r'),
                            variable,
                            delayed=delayed) for variable in variables
        ]
    except FileNotFoundError as err:
        # A missing variable array in an existing view partition indicates
        # an incomplete partition. The caller skips only absent partitions.
        if fs.exists(held):
            raise
        raise AbsentPartition(held) from err

    zds = dataset.Dataset(data, attrs=zds.attrs, delayed=zds.delayed)

    # Add immutable variables.
    if with_immutable and view_ref.have_immutable:
        zds.merge(
            open_zarr_group(view_ref.immutable_path,
                            view_ref.fs,
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


def _assert_variable_handled(reference: meta.DatasetSchema,
                             view: meta.DatasetSchema, variable: str) -> None:
    """Check that a variable is defined by the view, not its reference.

    Args:
        reference: The reference dataset.
        view: The view dataset.
        variable: The variable to check.

    Raises:
        ValueError: If the variable belongs to the reference collection
            or is not defined by the view.
    """
    if variable in reference.variables:
        raise ValueError(f'Variable {variable} is read-only')
    if variable not in view.variables:
        raise ValueError(f'Variable {variable} does not exist')


def _load_dataset(
    partition: str,
    *,
    base_dir: str,
    delayed: bool,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    metadata: meta.DatasetSchema,
    selected_variables: Iterable[str] | None,
) -> dataset.Dataset:
    """Load the selected variables of one view partition."""
    return _load_one_dataset(
        (view_ref.partitioning.parse(partition), []),
        base_dir=base_dir,
        delayed=delayed,
        fs=fs,
        selected_variables=view_ref.metadata.select_variables(
            selected_variables),
        view_ref=view_ref,
        variables=metadata.select_variables(selected_variables),
        with_immutable=True)[0]


def _assert_have_variables(metadata: meta.DatasetSchema) -> None:
    """Check that the view defines at least one variable.

    Args:
        metadata: The view schema.

    Raises:
        ValueError: If the view defines no variables.
    """
    if not metadata.variables:
        raise ValueError('The view has no variables')


def _load_dataset_with_overlap(
    partition: str,
    *,
    base_dir: str,
    delayed: bool,
    depth: int,
    fs: fsspec.AbstractFileSystem,
    metadata: meta.DatasetSchema,
    partitions: tuple[str, ...],
    selected_variables: Iterable[str] | None,
    view_ref: collection.Collection,
) -> tuple[dataset.Dataset, slice]:
    """Load a view partition and the neighbors selected by ``depth``.

    Args:
        base_dir: Base directory of the view.
        delayed: If True, load the datasets lazily.
        depth: The number of neighbors to load on either side.
        fs: The file system used to access the variables in the view.
        metadata: The metadata of the dataset.
        partition: The partition to load.
        partitions: The partitions of the view.
        selected_variables: The list of variables to retain from the view.
        view_ref: The view reference.

    Returns:
        The concatenated dataset and the slice locating the selected
        partition within it.
    """
    where: int = partitions.index(partition)
    selected: list[str] = [
        partitions[ix] for ix in range(where - depth, where + depth + 1)
        if 0 <= ix < len(partitions)
    ]
    # A neighbour the reference does not hold is left out. The partition
    # itself is the one being mapped over, so its absence raises
    # AbsentPartition.
    loaded: list[tuple[dataset.Dataset, str]] = []
    for item in selected:
        try:
            zds = _load_dataset(item,
                                base_dir=base_dir,
                                delayed=delayed,
                                fs=fs,
                                view_ref=view_ref,
                                metadata=metadata,
                                selected_variables=selected_variables)
        except AbsentPartition as err:
            if item == partition:
                raise
            warn_absent_partition(str(err))
            continue
        loaded.append((zds, item))

    start = 0
    indices = slice(0, 0, None)
    for zds, selected_partition in loaded:
        scheme = view_ref.partitioning.parse(selected_partition)
        axis = join_path(view_ref.partition_properties.dir,
                         view_ref.partitioning.join(scheme, view_ref.fs.sep),
                         view_ref.axis)
        size = partition_dimension_size(dimensions=zds.dimensions,
                                        dim=view_ref.dimension,
                                        axis=axis,
                                        fs=view_ref.fs)
        indices = slice(start, start + size, None)
        if selected_partition == partition:
            break
        start += size

    groups: list[dataset.Dataset] = [item[0] for item in loaded]
    result: dataset.Dataset = groups.pop(0)
    if groups:
        result = result.concat(groups, view_ref.dimension)
    return result, indices


def _calculate_axis_reference(
        path: str, view_ref: collection.Collection) -> AxisReference:
    """Read the reference axis of a partition and calculate its checksum.

    Args:
        path: The path of the partition.
        view_ref: The view reference.

    Returns:
        The axis values, their dimension name and SHA-256 checksum.
    """
    group: zarr.Group = store.open_group(view_ref.fs, path)
    array = cast('zarr.Array', group[view_ref.axis])
    axis = numpy.asarray(array[...])
    checksum: str = hashlib.sha256(axis.tobytes()).hexdigest()
    return AxisReference(axis, read_dimensions(array)[0], checksum)


def _write_checksum_array(
    partition: str,
    fs: fsspec.AbstractFileSystem,
    axis_ref: AxisReference,
) -> None:
    """Store the reference axis and its checksum in the view partition.

    Args:
        partition: The path of the partition.
        fs: The file system used to access the variables in the view.
        axis_ref: The reference axis values and their checksum.
    """
    array: zarr.Array
    checksum_path = join_path(partition, CHECKSUM_FILE)
    if fs.exists(checksum_path):
        array = store.open_array(fs, checksum_path, mode='r+')
    else:
        array = store.create_array(fs,
                                   checksum_path,
                                   shape=axis_ref.array.shape,
                                   dtype=axis_ref.array.dtype,
                                   dimensions=(axis_ref.dimension, ))
    array[...] = axis_ref.array
    array.attrs[CHECKSUM_ATTR] = axis_ref.checksum
    fs.invalidate_cache(checksum_path)


def _load_checksum_array(
    partition: str,
    fs: fsspec.AbstractFileSystem,
) -> zarr.Array:
    """Open the checksum array of a view partition.

    Args:
        partition: The path of the partition.
        fs: The file system used to access the variables in the view.

    Returns:
        The Zarr array containing the reference axis values and their
        checksum attribute.
    """
    return store.open_array(fs, join_path(partition, CHECKSUM_FILE))


def _write_checksum(
    partition: str,
    base_dir: str,
    view_ref: collection.Collection,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Write the reference axis checksum for a view partition.

    Args:
        partition: The path of the partition.
        base_dir: The base directory of the view.
        view_ref: The view reference.
        fs: The file system used to access the variables in the view.
    """
    relative = str(pathlib.Path(partition).relative_to(base_dir).as_posix())
    partition_ref = join_path(view_ref.partition_properties.dir, relative)
    _write_checksum_array(partition, fs,
                          _calculate_axis_reference(partition_ref, view_ref))


def _sync_partition(
    metadata: meta.DatasetSchema,
    partition: str,
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
) -> None:
    """Create a view partition, its variable arrays and its checksum.

    Args:
        metadata: The schema of the view variables.
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
                reference_fs=view_ref.fs,
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
    """Extend the arrays along the reference axis of a view partition.

    The checksum array is also rewritten.

    Args:
        partition: The partition to extend.
        fs: The file system used to access the variables in the view.
        axis_ref: The axis reference of the partition.
    """
    axis_name: str = axis_ref.dimension
    new_size: int = axis_ref.array.shape[0]
    try:
        for variable in fs.listdir(partition):
            array: zarr.Array = store.open_array(fs,
                                                 variable['name'],
                                                 mode='r+')
            dimensions: Sequence[str] = read_dimensions(array)
            if axis_name in dimensions:
                axis: int = dimensions.index(axis_name)
                shape = list(array.shape)
                shape[axis] = new_size
                array.resize(shape)
        _write_checksum_array(partition, fs, axis_ref)
        # _write_checksum_array invalidates the checksum array's cache,
        # not the cache of the partition directory.
    except Exception as exc:
        fs.rm(partition, recursive=True)
        fs.invalidate_cache(partition)
        raise exc from None


def _check_partition(
    partition: str,
    *,
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    metadata: meta.DatasetSchema,
) -> Skipped | None:
    """Check a view partition, returning SKIPPED if valid or None otherwise.

    A partition is valid when the reference collection still has it, it
    contains an array for each declared variable and the checksum array,
    and all these arrays can be read.

    Args:
        partition: The partition to check, relative to the view.
        base_dir: The base directory of the view.
        fs: The file system used to access the variables in the view.
        view_ref: The view reference.
        metadata: The schema of the variables owned by the view.
    """
    partition_ref = join_path(view_ref.partition_properties.dir, partition)
    if not view_ref.fs.exists(partition_ref):
        return None

    path = join_path(base_dir, partition)
    expected = {*metadata.variables, CHECKSUM_FILE}
    try:
        # An S3 listing may be stale after a variable is removed or repaired.
        fs.invalidate_cache(path)
        held = {os.path.basename(item['name']) for item in fs.listdir(path)}
    except FileNotFoundError:
        return None
    if not expected.issubset(held):
        return None

    try:
        for name in expected:
            array = store.open_array(fs, join_path(path, name), mode='r')
            data = array[...]
            del data
    except (FileNotFoundError, ValueError, TypeError, RuntimeError):
        return None
    return SKIPPED


def _repair_partition(record: tuple[str, bool], *, base_dir: str,
                      fs: fsspec.AbstractFileSystem,
                      view_ref: collection.Collection,
                      metadata: meta.DatasetSchema) -> None:
    """Delete a partition of a view, then write it again if the reference
    collection still has it.

    Args:
        record: The partition, relative to the view, and whether the
            reference collection still has it.
        base_dir: The base directory of the view.
        fs: The file system of the view.
        view_ref: The reference collection.
        metadata: The schema of the variables owned by the view.
    """
    partition, rewrite = record
    path = join_path(base_dir, partition)
    with contextlib.suppress(FileNotFoundError):
        fs.rm(path, recursive=True)
    fs.invalidate_cache(path)
    if rewrite:
        _sync_partition(metadata, partition, base_dir, fs, view_ref)


def _sync(
    partition: str,
    *,
    base_dir: str,
    fs: fsspec.AbstractFileSystem,
    view_ref: collection.Collection,
    metadata: meta.DatasetSchema,
    dry_run: bool = False,
) -> Skipped | None:
    """Synchronize a view partition with its reference collection.

    Partitions already synchronized are skipped.

    Args:
        partition: The partition, relative to the view.
        base_dir: The base directory of the view.
        fs: The file system of the view.
        view_ref: The reference collection.
        metadata: The schema of the variables owned by the view.
        dry_run: If True, the partition is checked but not written.

    Returns:
        :data:`~zcollection.execution.SKIPPED` if the partition is already
        synchronized, otherwise None.
    """
    partition_view = join_path(base_dir, partition)
    if not fs.exists(partition_view):
        if not dry_run:
            _sync_partition(metadata, partition, base_dir, fs, view_ref)
        return None

    partition_ref = join_path(view_ref.partition_properties.dir, partition)
    array: zarr.Array = _load_checksum_array(partition_view, fs)
    axis_ref: AxisReference = _calculate_axis_reference(path=partition_ref,
                                                        view_ref=view_ref)
    if axis_ref.checksum != array.attrs[CHECKSUM_ATTR]:
        # The axis of the view is a prefix of the reference one when the
        # partition holds the same measurements up to its end.
        axis_view = numpy.asarray(array[:])
        if axis_ref.array.size > axis_view.size and numpy.all(
                axis_view == axis_ref.array[:axis_view.size]):
            if not dry_run:
                _extend_partition(partition_view, fs, axis_ref)
            return None
        if not dry_run:
            fs.rm(partition_view, recursive=True)
            fs.invalidate_cache(partition_view)
            _sync_partition(metadata, partition, base_dir, fs, view_ref)
        return None
    return SKIPPED


def _serialize_filters(filters: collection.PartitionFilter, ) -> str:
    """Serialize a partition filter.

    Args:
        filters: The partition filter to serialize.

    Warns:
        UserWarning: If the filter cannot be serialized.

    Returns:
        The Base64-encoded pickle of the filter, or of ``None`` if
        serialization fails.
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
        The deserialized filter, or ``None`` if unpickling raises
        ``pickle.UnpicklingError``.
    """
    stream = io.BytesIO(base64.b64decode(filters.encode('utf-8')))
    try:
        return pickle.load(stream)
    except pickle.UnpicklingError:
        return None
