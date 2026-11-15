"""
Migration of a collection
=========================

Migrate collections created by earlier releases to the current storage
layout. Each partition is read and written again, preserving its variable
compressors, filters, fill values, dimensions, and attributes.

Migration writes to a separate target without modifying source data.
Incomplete target partitions may be removed before they are rewritten.
The pruning functions, :func:`prune <zcollection.migration.prune>` and
:func:`prune_view <zcollection.migration.prune_view>`, remove partitions
from their respective sources. Pruning preserves the source collection's
configuration and immutable variables.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import json
import logging
import math
import os
import pathlib
import traceback
import warnings

import dask.bag.core
import fsspec
import fsspec.implementations.local
import numpy

from .. import (
    collection,
    dask_utils,
    fs_utils,
    meta,
    partitioning,
    storage,
    store,
    sync,
    view,
)
from ..collection import IMMUTABLE
from ..convenience import open_collection
from ..fs_utils import join_path
from ..variable.abc import read_dimensions
from ..view.detail import CHECKSUM_FILE, _write_checksum
from .abc import Outcome, Status

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import zarr

    from .. import dataset
    from ..collection import PartitionFilter
    from ..type_hints import NDArray

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)

#: Metadata filenames used by Zarr 2 and Zarr 3 nodes. Other files
#: in an array store are chunks.
_METADATA_FILES = frozenset(
    {'.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json'})


def _require_distinct_paths(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> None:
    """Refuse pruning when both paths identify the same location."""
    if source_fs != target_fs:
        return

    source = fs_utils.normalize_path(source_fs, source)
    target = fs_utils.normalize_path(target_fs, target)
    if isinstance(source_fs, fsspec.implementations.local.LocalFileSystem):
        source, target = os.path.normpath(source), os.path.normpath(target)

    if source.rstrip('/') == target.rstrip('/'):
        raise ValueError('The source and target identify the same location.')


def _merge(outcomes: Iterable[Outcome]) -> Outcome:
    """Combine the outcomes of several partitions.

    Args:
        outcomes: The outcomes to gather.

    Returns:
        A combined outcome with the partitions in each field sorted.
    """
    items: tuple[Outcome, ...] = tuple(outcomes)
    return Outcome(
        tuple(sorted(name for item in items for name in item.done)),
        tuple(sorted(name for item in items for name in item.skipped)),
        tuple(sorted(name for item in items for name in item.failed)))


def _failure(partition: str, err: Exception) -> Outcome:
    """Report a failed partition and log its exception.

    Called from an exception handler to include the traceback in the log.

    Args:
        partition: The path of the partition, relative to its collection.
        err: The error raised.

    Returns:
        An outcome containing the partition and its error message.
    """
    _LOGGER.exception('The run stopped on the partition %r', partition)
    text: str = ''.join(traceback.format_exception_only(err)).strip()
    return Outcome((), (), ((partition, text), ))


def _relative_path(path: str, base_dir: str) -> str:
    """Return a partition path relative to its collection.

    Args:
        path: The path of the partition.
        base_dir: The collection directory.

    Returns:
        The relative path.
    """
    return pathlib.Path(path).relative_to(base_dir).as_posix()


def is_complete(fs: fsspec.AbstractFileSystem, dirname: str) -> bool:
    """Check whether a partition has consolidated metadata.

    Args:
        fs: The file system storing the partition.
        dirname: The path of the partition.

    Returns:
        Whether the partition can be opened using consolidated metadata,
        which is written last when a partition is created.
    """
    try:
        store.open_group(fs, dirname)
    except (OSError, ValueError):
        return False
    return True


def dimension_chunks(
        arrays: Iterable[tuple[str, zarr.Array]]) -> meta.DimensionType:
    """Read the chunk size of each dimension in a Zarr group.

    Args:
        arrays: The name and the array of each variable of the group.

    Returns:
        The chunk size for each dimension.

    Raises:
        ValueError: If arrays disagree on the chunk size of a
            dimension. Each dimension has one chunk size per collection.
    """
    chunks: meta.DimensionType = {}
    for name, array in arrays:
        for dimension, size in zip(read_dimensions(array),
                                   array.chunks,
                                   strict=True):
            held: int = chunks.setdefault(dimension, size)
            if held != size:
                raise ValueError(
                    f'The array {name!r} is chunked by {size} along the '
                    f'dimension {dimension!r}, another by {held}.')
    return chunks


def copy_group(
    source_dirname: str,
    target_dirname: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> None:
    """Copy a Zarr group while preserving its chunk sizes.

    Args:
        source_dirname: The path of the group to read.
        target_dirname: The destination group path. Any existing group
            is replaced.
        source_fs: The file system storing the source group.
        target_fs: The file system for the destination group.

    Raises:
        ValueError: If the arrays of the source disagree on the chunk size
            of a dimension.
    """
    zds = storage.open_zarr_group(source_dirname, source_fs, delayed=False)

    # A dataset read from storage has no chunking metadata; the
    # source arrays provide it.
    zds.dim_chunks = dimension_chunks(
        store.open_group(source_fs, source_dirname).arrays())

    _remove(target_fs, target_dirname)
    store.create_group(target_fs, target_dirname)
    try:
        storage.write_zarr_group(zds,
                                 target_dirname,
                                 target_fs,
                                 sync.NoSync(),
                                 distributed=False)
    except Exception:
        _remove(target_fs, target_dirname)
        raise


def copy_arrays(
    source_dirname: str,
    target_dirname: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    names: Sequence[str],
) -> None:
    """Copy the Zarr arrays of one view partition.

    Args:
        source_dirname: The path of the partition to read.
        target_dirname: The destination partition path. Any existing
            partition is replaced.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: Names of the variables defined by the view.

    Raises:
        ValueError: If the arrays of the source disagree on the chunk size
            of a dimension.
    """
    arrays: dict[str, zarr.Array] = {
        name:
        store.open_array(source_fs,
                         fs_utils.join_path(source_dirname, name),
                         mode='r')
        for name in names
    }
    chunks: meta.DimensionType = dimension_chunks(arrays.items())

    _remove(target_fs, target_dirname)
    for name, array in arrays.items():
        variable = storage.open_zarr_array(array, name, delayed=False)
        path = fs_utils.join_path(target_dirname, name)
        store.create_array(target_fs,
                           path,
                           shape=variable.shape,
                           chunks=tuple(chunks[dimension]
                                        for dimension in variable.dimensions),
                           dtype=variable.dtype,
                           compressor=variable.compressor,
                           fill_value=variable.fill_value,
                           filters=variable.filters,
                           dimensions=variable.dimensions,
                           attributes=storage.variable_attributes(variable),
                           overwrite=True,
                           write_empty_chunks=False)
        storage.update_zarr_array(path, variable.values, target_fs)
    target_fs.invalidate_cache(target_dirname)


def _remove(fs: fsspec.AbstractFileSystem, dirname: str) -> None:
    """Remove a directory and invalidate the file system cache.

    Args:
        fs: The file system storing the directory.
        dirname: The path of the directory.
    """
    if fs.exists(dirname):
        fs.rm(dirname, recursive=True)
        fs.invalidate_cache(dirname)


def _differ(first: NDArray, second: NDArray) -> bool:
    """Compare two arrays, including their masks and data types.

    Args:
        first: The values read from the source.
        second: The values read from the target.

    Returns:
        Whether their shapes, data types, masks or unmasked values differ.
        Two NaNs or two NaTs are considered equal.
    """
    mask: NDArray = numpy.ma.getmaskarray(first)
    if first.shape != second.shape or not numpy.array_equal(
            mask, numpy.ma.getmaskarray(second)):
        return True

    values: NDArray = numpy.ma.getdata(first)[~mask]
    other: NDArray = numpy.ma.getdata(second)[~mask]
    if values.dtype != other.dtype:
        return True
    if values.dtype.kind in 'fcMm':
        return not numpy.array_equal(values, other, equal_nan=True)
    return not numpy.array_equal(values, other)


def incomplete_arrays(
    dirname: str,
    fs: fsspec.AbstractFileSystem,
) -> tuple[str, ...]:
    """Find arrays with fewer stored chunk files than their chunk grids require.

    Zarr reads missing chunks as the fill value. An array can therefore be
    opened and read even when some of its chunk files are missing.

    The check is reliable for collections written by earlier releases,
    which stored every chunk, including chunks containing only the fill
    value. Current collections omit fill-only chunks, so this function
    may also report arrays that are complete.

    Args:
        dirname: The path of the group.
        fs: The file system storing the group.

    Returns:
        The sorted names of arrays with fewer chunk files than expected.
    """
    start = len(fs_utils.normalize_path(fs, dirname)) + 1
    held: dict[str, int] = {}
    for key in (item[start:] for item in fs.find(dirname)):
        parts = key.split(fs_utils.SEPARATOR)
        if len(parts) > 1 and parts[-1] not in _METADATA_FILES:
            held[parts[0]] = held.get(parts[0], 0) + 1

    short: list[str] = []
    for name, array in store.open_group(fs, dirname).arrays():
        expected = math.prod(
            -(-size // chunk)
            for size, chunk in zip(array.shape, array.chunks, strict=True))
        if held.get(name, 0) < expected:
            short.append(name)
    return tuple(sorted(short))


def _nodes(fs: fsspec.AbstractFileSystem, dirname: str) -> set[str]:
    """Return the names of Zarr nodes with metadata in a directory.

    Args:
        fs: The file system the directory is held on.
        dirname: The path of the directory.

    Returns:
        The node names.
    """
    start = len(fs_utils.normalize_path(fs, dirname)) + 1
    return {
        parts[0]
        for parts in (key.split(fs_utils.SEPARATOR)
                      for key in (item[start:] for item in fs.find(dirname)))
        if len(parts) > 1 and parts[-1] in _METADATA_FILES
    }


def is_view_partition_complete(
    fs: fsspec.AbstractFileSystem,
    dirname: str,
    names: Iterable[str],
) -> bool:
    """Check whether a view partition was written completely.

    View variables are stored as individual Zarr arrays, without a group
    containing consolidated metadata. The checksum is written after the
    arrays, so its presence together with all declared arrays indicates
    that the partition write finished.

    Args:
        fs: The file system storing the view.
        dirname: The path of the partition.
        names: The variable names declared by the view.

    Returns:
        ``True`` if the checksum and all declared arrays are present;
        otherwise ``False``.
    """
    held = _nodes(fs, dirname)
    return CHECKSUM_FILE in held and set(names) <= held


def warn_absent(dirname: str, fs: fsspec.AbstractFileSystem) -> None:
    """Warn about arrays with missing chunks in storage.

    Args:
        dirname: The path of the group.
        fs: The file system storing the group.
    """
    short = incomplete_arrays(dirname, fs)
    if short:
        _LOGGER.warning('%r is short of a chunk for %s', dirname,
                        ', '.join(short))


def differing_arrays(
    source_dirname: str,
    target_dirname: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    names: Iterable[str],
) -> tuple[str, ...]:
    """Compare the array values in two view partitions.

    Args:
        source_dirname: The path of the partition to read.
        target_dirname: The path of the partition to compare it with.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.

    Returns:
        Sorted names of variables with different values or missing from
        the target.
    """
    held = _nodes(target_fs, target_dirname)
    differing = {name for name in names if name not in held}
    for name in set(names) - differing:
        if _differ(
                store.open_ndarray(source_fs,
                                   fs_utils.join_path(source_dirname, name)),
                store.open_ndarray(target_fs,
                                   fs_utils.join_path(target_dirname, name))):
            differing.add(name)
    return tuple(sorted(differing))


def _attributes(
        attrs: Iterable[meta.Attribute]) -> tuple[tuple[str, Any], ...]:
    """Return attributes sorted by name for comparison.

    Args:
        attrs: The attributes of a variable or of a dataset.

    Returns:
        The name and value of each attribute, sorted by name.
    """
    return tuple(sorted((item.name, item.value) for item in attrs))


def _chunks(
    dirname: str,
    fs: fsspec.AbstractFileSystem,
) -> dict[str, tuple[int, ...]]:
    """Return the chunk shape of every array in a Zarr group.

    Args:
        dirname: The path of the group.
        fs: The file system the group is held on.

    Returns:
        A mapping from array names to chunk shapes.
    """
    return {
        name: array.chunks
        for name, array in store.open_group(fs, dirname).arrays()
    }


def _variable_differs(first: dataset.Variable,
                      second: dataset.Variable) -> bool:
    """Check whether two variables differ in their migrated properties.

    Args:
        first: The variable read from the source.
        second: The variable read from the target.

    Returns:
        Whether their dimensions, attributes, fill values or data differ.
    """
    if (first.dimensions != second.dimensions
            or _attributes(first.attrs) != _attributes(second.attrs)):
        return True
    if (first.fill_value is None) != (second.fill_value is None):
        return True
    if first.fill_value is not None and _differ(
            numpy.asarray(first.fill_value), numpy.asarray(second.fill_value)):
        return True
    return _differ(first.values, second.values)


def differing_variables(
    source_dirname: str,
    target_dirname: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> tuple[str, ...]:
    """Compare the variables and attributes of two Zarr groups.

    The comparison covers values and masks, attributes, fill values,
    dimension names and chunk shapes. Differences in group attributes
    are reported under ``attributes``.

    Args:
        source_dirname: The path of the group to read.
        target_dirname: The path of the group to compare it with.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        Sorted names of variables that differ or exist in only one group.
    """
    first = storage.open_zarr_group(source_dirname, source_fs, delayed=False)
    second = storage.open_zarr_group(target_dirname, target_fs, delayed=False)
    source_chunks = _chunks(source_dirname, source_fs)
    target_chunks = _chunks(target_dirname, target_fs)

    differing: set[str] = set(first.variables) ^ set(second.variables)
    if _attributes(first.attrs) != _attributes(second.attrs):
        differing.add('attributes')
    for name in set(first.variables) & set(second.variables):
        if (source_chunks.get(name) != target_chunks.get(name)
                or _variable_differs(first[name], second[name])):
            differing.add(name)
    return tuple(sorted(differing))


def _migrate_one(
    partition: str,
    *,
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> Outcome:
    """Copy a source partition to the target.

    Args:
        partition: The path of the partition, relative to its collection.
        source: The directory of the source.
        target: The directory the target is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        An outcome with the partition under ``done`` if copied, under
        ``skipped`` if already complete in the target, or under
        ``failed`` with an error message if copying failed.
    """
    target_dirname = fs_utils.join_path(target, partition)
    if is_complete(target_fs, target_dirname):
        return Outcome((), (partition, ), ())

    try:
        copy_group(fs_utils.join_path(source, partition), target_dirname,
                   source_fs, target_fs)
    except Exception as err:
        return _failure(partition, err)
    return Outcome((partition, ), (), ())


def _check_one(
    partition: str,
    *,
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> Outcome:
    """Compare a source partition with its migrated copy.

    Args:
        partition: The path of the partition, relative to its collection.
        source: The directory of the source.
        target: The directory the target is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        An outcome with the partition under ``done`` if values match,
        under ``skipped`` if the target copy is incomplete or absent,
        or under ``failed`` with the names of differing variables.
    """
    target_dirname = fs_utils.join_path(target, partition)
    if not is_complete(target_fs, target_dirname):
        return Outcome((), (partition, ), ())

    source_dirname = fs_utils.join_path(source, partition)
    try:
        differing = differing_variables(source_dirname, target_dirname,
                                        source_fs, target_fs)
        warn_absent(source_dirname, source_fs)
    except Exception as err:
        return _failure(partition, err)

    if differing:
        return Outcome((), (), ((partition, ', '.join(differing)), ))
    return Outcome((partition, ), (), ())


def _prune_one(
    partition: str,
    *,
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> Outcome:
    """Remove one partition from the source.

    Args:
        partition: The path of the partition, relative to its collection.
        source: The directory of the source.
        target: The directory the target is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        An outcome with the partition under ``done`` if removed, under
        ``skipped`` if already absent from the source, or under
        ``failed`` with the reason if no complete target copy is available
        or removal fails.
    """
    source_dirname = fs_utils.join_path(source, partition)
    if not source_fs.exists(source_dirname):
        return Outcome((), (partition, ), ())

    if not is_complete(target_fs, fs_utils.join_path(target, partition)):
        return Outcome((), (),
                       ((partition, 'the target does not hold this partition '
                         'complete'), ))

    try:
        source_fs.rm(source_dirname, recursive=True)
        source_fs.invalidate_cache(source_dirname)
    except Exception as err:
        return _failure(partition, err)
    return Outcome((partition, ), (), ())


def _prune_view_one(
    partition: str,
    *,
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    names: Sequence[str],
) -> Outcome:
    """Remove one partition from a view.

    Args:
        partition: The path of the partition, relative to the view.
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.

    Returns:
        An outcome with the partition under ``done`` if removed, under
        ``skipped`` if already absent from the source, or under
        ``failed`` with the reason if no complete target copy is available
        or removal fails.
    """
    source_dirname = fs_utils.join_path(source, partition)
    if not source_fs.exists(source_dirname):
        return Outcome((), (partition, ), ())

    if not is_view_partition_complete(
            target_fs, fs_utils.join_path(target, partition), names):
        return Outcome((), (),
                       ((partition, 'the target does not hold this partition '
                         'in full'), ))

    try:
        source_fs.rm(source_dirname, recursive=True)
        source_fs.invalidate_cache(source_dirname)
    except Exception as err:
        return _failure(partition, err)
    return Outcome((partition, ), (), ())


def _migrate_view_one(
    partition: str,
    *,
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    names: Sequence[str],
    view_ref: collection.Collection,
) -> Outcome:
    """Copy one view partition to the target.

    The checksum of the reference axis is recomputed from the collection
    the target view references.

    Args:
        partition: The path of the partition, relative to the view.
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.
        view_ref: The collection the target view references.

    Returns:
        An outcome with the partition under ``done`` if copied, under
        ``skipped`` if already complete in the target, or under
        ``failed`` with an error message if copying failed.
    """
    target_dirname = fs_utils.join_path(target, partition)
    if is_view_partition_complete(target_fs, target_dirname, names):
        return Outcome((), (partition, ), ())

    try:
        copy_arrays(fs_utils.join_path(source, partition), target_dirname,
                    source_fs, target_fs, names)
        _write_checksum(target_dirname, target, view_ref, target_fs)
    except Exception as err:
        return _failure(partition, err)
    return Outcome((partition, ), (), ())


def _check_view_one(
    partition: str,
    *,
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    names: Sequence[str],
) -> Outcome:
    """Compare the variables of a view partition with their copies.

    Args:
        partition: The path of the partition, relative to the view.
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.

    Returns:
        An outcome with the partition under ``done`` if values match,
        under ``skipped`` if the target copy is incomplete or absent,
        or under ``failed`` with the names of differing variables.
    """
    target_dirname = fs_utils.join_path(target, partition)
    if not is_view_partition_complete(target_fs, target_dirname, names):
        return Outcome((), (partition, ), ())

    try:
        differing = differing_arrays(fs_utils.join_path(source, partition),
                                     target_dirname, source_fs, target_fs,
                                     names)
    except Exception as err:
        return _failure(partition, err)

    if differing:
        return Outcome((), (), ((partition, ', '.join(differing)), ))
    return Outcome((partition, ), (), ())


def _run(
    func: Callable[..., Outcome],
    partitions: Sequence[str],
    *,
    distributed: bool,
    **kwargs,
) -> Outcome:
    """Run an operation on each selected partition.

    Args:
        func: The function to apply. It returns an outcome for the
            partition it is given, and raises nothing.
        partitions: The paths of the partitions, relative to their
            collection.
        distributed: Whether to spread the partitions over a Dask cluster.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        The combined outcome of all selected partitions.
    """
    if not partitions:
        return Outcome((), (), ())

    if distributed:
        client = dask_utils.get_client()
        bag: dask.bag.core.Bag = dask.bag.core.from_sequence(
            partitions, npartitions=len(partitions))
        return _merge(bag.map(func, **kwargs).compute(scheduler=client))

    return _merge(func(item, **kwargs) for item in partitions)


def _selection(
    src: collection.ReadOnlyCollection,
    filters: PartitionFilter,
    selected_partitions: Iterable[str] | None,
) -> tuple[str, ...]:
    """Select and sort the source partitions for an operation.

    Args:
        src: The source collection.
        filters: The predicate selecting the partitions.
        selected_partitions: The partitions to act on, named by their path
            relative to the collection.

    Returns:
        Partition paths relative to the source, sorted lexicographically.
    """
    if selected_partitions is not None:
        selected_partitions = tuple(selected_partitions)

    partitions: set[str] = set(
        src.partitions(filters=filters,
                       selected_partitions=selected_partitions,
                       relative=True))
    return tuple(sorted(partitions))


def migrate_partitions(
    src: collection.ReadOnlyCollection,
    target: str,
    target_fs: fsspec.AbstractFileSystem,
    *,
    filters: PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> Outcome:
    """Copy the selected partitions of a collection into the target.

    Args:
        src: The source collection.
        target: The directory the target is stored in.
        target_fs: The file system the target is held on.
        filters: The predicate selecting the partitions to copy.
        selected_partitions: The partitions to copy, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.

    Returns:
        An outcome with partitions copied under ``done``, already
        complete in the target under ``skipped``, and unsuccessful
        copies under ``failed``.
    """
    return _run(_migrate_one,
                _selection(src, filters, selected_partitions),
                distributed=distributed,
                source=src.partition_properties.dir,
                target=target,
                source_fs=src.fs,
                target_fs=target_fs)


def migrate_collection(
    src: collection.ReadOnlyCollection,
    target: str,
    target_fs: fsspec.AbstractFileSystem,
    *,
    filters: PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> Outcome:
    """Copy the immutable variables and the selected partitions.

    Args:
        src: The source collection.
        target: The directory the target is stored in.
        target_fs: The file system the target is held on.
        filters: The predicate selecting the partitions to copy.
        selected_partitions: The partitions to copy, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.

    Returns:
        A combined outcome for immutable variables and partitions.
    """
    return _merge((
        migrate_immutable(src, target, target_fs),
        migrate_partitions(src,
                           target,
                           target_fs,
                           filters=filters,
                           selected_partitions=selected_partitions,
                           distributed=distributed),
    ))


def migrate_immutable(
    src: collection.ReadOnlyCollection,
    target: str,
    target_fs: fsspec.AbstractFileSystem,
) -> Outcome:
    """Copy the immutable variables of a collection into the target.

    Args:
        src: The source collection.
        target: The directory the target is stored in.
        target_fs: The file system the target is held on.

    Returns:
        ``.immutable`` under ``done`` when it was written, under
        ``skipped`` when the target already held every variable of the source,
        and under ``failed`` with the error as text when the copy stopped.
        Every field is empty when the source holds no immutable variable.
    """
    source_dirname = fs_utils.join_path(src.partition_properties.dir,
                                        IMMUTABLE)
    if not is_complete(src.fs, source_dirname):
        return Outcome((), (), ())

    held = set(store.open_group(src.fs, source_dirname))
    if not held:
        return Outcome((), (), ())

    target_dirname = fs_utils.join_path(target, IMMUTABLE)
    if is_complete(target_fs, target_dirname) and held <= set(
            store.open_group(target_fs, target_dirname)):
        return Outcome((), (IMMUTABLE, ), ())

    try:
        copy_group(source_dirname, target_dirname, src.fs, target_fs)
    except Exception as err:
        return _failure(IMMUTABLE, err)
    return Outcome((IMMUTABLE, ), (), ())


def check_immutable(
    src: collection.ReadOnlyCollection,
    target: str,
    target_fs: fsspec.AbstractFileSystem,
) -> Outcome:
    """Compare the immutable variables of a collection with their copies.

    Args:
        src: The source collection.
        target: The directory the target is stored in.
        target_fs: The file system the target is held on.

    Returns:
        ``.immutable`` under ``done`` when the two hold the same values,
        under ``skipped`` when the target does not hold it complete, and
        under ``failed`` with the names of the variables that differ. Every
        field is empty when the source holds no immutable variable.
    """
    source_dirname = fs_utils.join_path(src.partition_properties.dir,
                                        IMMUTABLE)
    if not is_complete(src.fs, source_dirname) or not set(
            store.open_group(src.fs, source_dirname)):
        return Outcome((), (), ())

    target_dirname = fs_utils.join_path(target, IMMUTABLE)
    if not is_complete(target_fs, target_dirname):
        return Outcome((), (IMMUTABLE, ), ())

    try:
        differing = differing_variables(source_dirname, target_dirname, src.fs,
                                        target_fs)
        warn_absent(source_dirname, src.fs)
    except Exception as err:
        return _failure(IMMUTABLE, err)

    if differing:
        return Outcome((), (), ((IMMUTABLE, ', '.join(differing)), ))
    return Outcome((IMMUTABLE, ), (), ())


def check_partitions(
    src: collection.ReadOnlyCollection,
    target: str,
    target_fs: fsspec.AbstractFileSystem,
    *,
    filters: PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> Outcome:
    """Compare the values of the selected partitions with their copies.

    Args:
        src: The source collection.
        target: The directory the target is stored in.
        target_fs: The file system the target is held on.
        filters: The predicate selecting the partitions to compare.
        selected_partitions: The partitions to compare, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.

    Returns:
        The partitions whose values are equal under ``done``, those the
        target does not hold complete under ``skipped``, and those whose
        variables differ under ``failed``. The immutable variables are
        reported under the name ``.immutable``.

        A warning names the source arrays with fewer chunk files than their
        chunk grid requires. Zarr reads missing chunks as fill values, so a
        value comparison alone cannot detect their absence.
    """
    return _merge((
        check_immutable(src, target, target_fs),
        _run(_check_one,
             _selection(src, filters, selected_partitions),
             distributed=distributed,
             source=src.partition_properties.dir,
             target=target,
             source_fs=src.fs,
             target_fs=target_fs),
    ))


def prune_partitions(
    src: collection.ReadOnlyCollection,
    target: str,
    target_fs: fsspec.AbstractFileSystem,
    *,
    filters: PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> Outcome:
    """Remove the selected partitions from the source.

    Args:
        src: The source collection.
        target: The directory the target is stored in.
        target_fs: The file system the target is held on.
        filters: The predicate selecting the partitions to remove.
        selected_partitions: The partitions to remove, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.

    Returns:
        The partitions removed under ``done``, those the source no longer
        held under ``skipped``, and those the target does not hold complete
        under ``failed``.
    """
    return _run(_prune_one,
                _selection(src, filters, selected_partitions),
                distributed=distributed,
                source=src.partition_properties.dir,
                target=target,
                source_fs=src.fs,
                target_fs=target_fs)


def migrate_view_partitions(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    *,
    partitions: Sequence[str],
    names: Sequence[str],
    view_ref: collection.Collection,
    distributed: bool = True,
) -> Outcome:
    """Copy selected view partitions into the target.

    Args:
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        partitions: The partitions to copy, named by their path relative
            to the view.
        names: The variables the view declares.
        view_ref: The collection the target view references.
        distributed: Whether to spread the partitions over a Dask cluster.

    Returns:
        An outcome with copied partitions under ``done``, partitions
        already complete in the target under ``skipped``, and failed
        copies under ``failed``.
    """
    return _run(_migrate_view_one,
                partitions,
                distributed=distributed,
                source=source,
                target=target,
                source_fs=source_fs,
                target_fs=target_fs,
                names=names,
                view_ref=view_ref)


def check_view_partitions(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    *,
    partitions: Sequence[str],
    names: Sequence[str],
    distributed: bool = True,
) -> Outcome:
    """Compare view variables with their migrated copies.

    Args:
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        partitions: The partitions to compare, named by their path
            relative to the view.
        names: The variables the view declares.
        distributed: Whether to spread the partitions over a Dask cluster.

    Returns:
        The partitions whose values are equal under ``done``, those the
        target does not hold in full under ``skipped``, and those whose
        variables differ under ``failed``.
    """
    return _run(_check_view_one,
                partitions,
                distributed=distributed,
                source=source,
                target=target,
                source_fs=source_fs,
                target_fs=target_fs,
                names=names)


def prune_view_partitions(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    *,
    partitions: Sequence[str],
    names: Sequence[str],
    distributed: bool = True,
) -> Outcome:
    """Remove source view partitions that are complete in the target.

    Args:
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        partitions: The partitions to remove, named by their path relative
            to the view.
        names: The variables the view declares.
        distributed: Whether to spread the partitions over a Dask cluster.

    Returns:
        The partitions removed under ``done``, those the source no longer
        held under ``skipped``, and those the target does not hold in full
        under ``failed``.
    """
    return _run(_prune_view_one,
                partitions,
                distributed=distributed,
                source=source,
                target=target,
                source_fs=source_fs,
                target_fs=target_fs,
                names=names)


def view_status(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
    *,
    names: Sequence[str],
    view_ref: collection.Collection,
) -> Status:
    """Compare the partitions of a view with those of its copy.

    Args:
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.
        view_ref: The collection the source view references.

    Returns:
        The status of each partition, classified as described in
        :class:`~zcollection.migration.Status`.
    """
    source_partitions = {
        _relative_path(item, source)
        for item in view_ref.partitioning.list_partitions(source_fs, source)
    }
    target_partitions: set[str] = set()
    if target_fs.exists(target):
        target_partitions = {
            _relative_path(item, target)
            for item in view_ref.partitioning.list_partitions(
                target_fs, target)
        }

    complete = {
        item
        for item in target_partitions if is_view_partition_complete(
            target_fs, fs_utils.join_path(target, item), names)
    }
    return Status(
        tuple(sorted(source_partitions & complete)),
        tuple(sorted(source_partitions - complete)),
        tuple(sorted(complete - source_partitions)),
        tuple(sorted(target_partitions - complete - source_partitions)))


def collection_status(
    src: collection.ReadOnlyCollection,
    target: str,
    target_fs: fsspec.AbstractFileSystem,
) -> Status:
    """Compare the partitions of a collection with those of its copy.

    Args:
        src: The source collection.
        target: The directory the target is stored in.
        target_fs: The file system the target is held on.

    Returns:
        The status of each partition, classified as described in
        :class:`~zcollection.migration.Status`.
    """
    source_partitions: set[str] = set(src.partitions(relative=True))
    target_partitions: set[str] = set()
    if target_fs.exists(target):
        target_partitions = {
            _relative_path(item, target)
            for item in src.partitioning.list_partitions(target_fs, target)
        }

    complete = {
        item
        for item in target_partitions
        if is_complete(target_fs, fs_utils.join_path(target, item))
    }
    return Status(
        tuple(sorted(source_partitions & complete)),
        tuple(sorted(source_partitions - complete)),
        tuple(sorted(complete - source_partitions)),
        tuple(sorted(target_partitions - complete - source_partitions)))


#: Start of the warning emitted when a collection written by an earlier
#: release is opened. The functions here open such a collection
#: deliberately.
_DEPRECATED_WARNING = 'This collection was written by an earlier release'


def _open_source(
    path: str,
    fs: fsspec.AbstractFileSystem,
) -> collection.Collection:
    """Open the source collection in read-only mode.

    Args:
        path: The path to the collection.
        fs: The file system storing the collection.

    Returns:
        The collection.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message=_DEPRECATED_WARNING,
                                category=UserWarning)
        return open_collection(path, mode='r', filesystem=fs)


def read_collection_config(path: str,
                           fs: fsspec.AbstractFileSystem) -> dict[str, Any]:
    """Read the configuration of a collection.

    Args:
        path: The path to the collection.
        fs: The file system storing the collection.

    Returns:
        The configuration.

    Raises:
        ValueError: If the directory contains no collection configuration.
    """
    config: str = collection.Collection._config(path)

    if not fs.exists(config):
        raise ValueError(f'zarr collection not found at path {path!r}')

    with fs.open(config) as stream:
        return json.load(stream)


def _configuration(
    path: str,
    fs: fsspec.AbstractFileSystem,
    data: dict[str, Any],
) -> tuple[str, meta.DatasetSchema, partitioning.Partitioning]:
    """Read the axis, schema and partitioning from a collection configuration.

    Args:
        path: The path to the collection.
        fs: The file system the collection is held on.
        data: The configuration of the collection.

    Returns:
        The axis, metadata and partitioning. For older configurations,
        dimension sizes are recovered from the first partition or from
        immutable variables.

    Raises:
        ValueError: If an older collection has no partition from which
            to recover dimension sizes.
    """
    zcol_partitioning: partitioning.Partitioning = partitioning.get_codecs(
        data['partitioning'])
    axis: str = data['axis']

    if meta.format_version(data) != meta.DEPRECATED_FORMAT_VERSION:
        return axis, meta.DatasetSchema.from_config(
            data['dataset']), zcol_partitioning

    ds = meta.DatasetSchema.from_deprecated_config(data['dataset'])
    main_dimension = ds.variables[axis].dimensions[0]

    dimensions_source = _dimensions_source(ds, main_dimension)

    partitions = list(zcol_partitioning.list_partitions(fs=fs, path=path))

    if not partitions:
        raise ValueError(
            'The collection requires at least one partition to be updatable.')

    # Read dimension sizes from the first partition.
    partition = partitions[0]

    for dimension, variable in dimensions_source.items():
        dim_index = variable.dimensions.index(dimension)
        if main_dimension in variable.dimensions:
            _LOGGER.warning('Extracting %r size from %r variable.', dimension,
                            variable.name)
            var_path = join_path(partition, variable.name)
        else:
            # This is an immutable variable
            _LOGGER.warning('Extracting %r size from %r constant.', dimension,
                            variable.name)
            var_path = join_path(path, collection.IMMUTABLE, variable.name)

        var_shape = store.open_array(fs, var_path, mode='r').shape

        dim_size = var_shape[dim_index]
        _LOGGER.warning('Assigning size %r to dimension %r.', dim_size,
                        dimension)

        ds.dimensions[dimension].value = dim_size

    missing_dimensions = set(dimensions_source) - set(ds.dimensions)
    if missing_dimensions:
        raise ValueError('Something is wrong, some dimensions are missing: '
                         f'{missing_dimensions}')

    return axis, ds, zcol_partitioning


def _dimensions_source(ds: meta.DatasetSchema,
                       main_dimension: str) -> dict[str, meta.VariableSchema]:
    """Map each non-partitioned dimension to a variable that uses it.

    Args:
        ds: The dataset containing the variables.
        main_dimension: The partitioned dimension to exclude.

    Returns:
        A mapping from each non-partitioned dimension to a variable
        that uses it.
    """
    dimensions_source: dict[str, meta.VariableSchema] = {}
    dimensions = list(ds.dimensions)

    for dimension in dimensions:
        if dimension == main_dimension:
            continue

        for variable in ds.variables.values():
            if dimension in variable.dimensions:
                dimensions_source[dimension] = variable
                break

    return dimensions_source


def _matching_target(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> collection.Collection:
    """Open the copy of a collection, read only.

    The source and target must agree on the axis, partitioning and
    variable names. Dimension sizes are not checked: they do not
    identify a collection, and checking them would require reading
    source arrays.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The copy.

    Raises:
        ValueError: If the target is not the copy of the source.
    """
    data = read_collection_config(source, source_fs)
    zcol = open_collection(target, mode='r', filesystem=target_fs)
    expected = (data['axis'],
                partitioning.get_codecs(data['partitioning']).get_config(),
                {item['name']
                 for item in data['dataset']['variables']})
    if (zcol.axis, zcol.partitioning.get_config(),
            set(zcol.metadata.variables)) != expected:
        raise ValueError(f'The collection {target!r} is not the copy of '
                         f'{source!r}.')
    return zcol


def _open_target(
    source: str,
    target: str,
    source_fs: fsspec.AbstractFileSystem,
    target_fs: fsspec.AbstractFileSystem,
) -> collection.Collection:
    """Open or create the target collection for migration.

    Args:
        source: The path to the collection to copy.
        target: The path to the collection to write.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The target collection. If no configuration exists, one is
        created from the source configuration.

    Raises:
        ValueError: If the target holds a configuration that disagrees with
            the one the source describes.
    """
    axis, ds, zcol_partitioning = _configuration(
        source, source_fs, read_collection_config(source, source_fs))

    if not target_fs.exists(collection.Collection._config(target)):
        return collection.Collection(axis,
                                     ds,
                                     zcol_partitioning,
                                     target,
                                     mode='w',
                                     filesystem=target_fs)

    zcol = collection.Collection.from_config(target,
                                             mode='w',
                                             filesystem=target_fs)
    derived = collection.ReadOnlyCollection(axis,
                                            ds,
                                            zcol_partitioning,
                                            target,
                                            mode='r',
                                            filesystem=target_fs)
    if (zcol.axis, zcol.metadata.get_config(),
            zcol.partitioning.get_config()) != (
                derived.axis, derived.metadata.get_config(),
                derived.partitioning.get_config()):
        raise ValueError(
            f'The collection {target!r} is not configured as the one derived '
            f'from {source!r}.')
    return zcol


def _reference(
    path: str,
    fs: fsspec.AbstractFileSystem,
) -> tuple[meta.DatasetSchema, view.View]:
    """Open a view and its reference collection.

    Args:
        path: The path to the view.
        fs: The file system the view is held on.

    Returns:
        The variables the view declares, and the view itself.
    """
    data = read_view_config(path, fs)
    ds = _view_metadata(data)
    view_ref: dict[str, Any] = data['view_ref']
    zview = view.View(fs_utils.normalize_path(fs, path),
                      view.ViewReference(
                          view_ref['path'],
                          fsspec.AbstractFileSystem.from_json(
                              json.dumps(view_ref['fs']))),
                      ds=ds,
                      filesystem=fs,
                      filters=view._deserialize_filters(data['filters']))
    return ds, zview


def _view_selection(
    zview: view.View,
    filters: collection.PartitionFilter,
    selected_partitions: Iterable[str] | None,
) -> tuple[str, ...]:
    """Select the view partitions for an operation.

    Args:
        zview: The view to list.
        filters: The predicate selecting the partitions.
        selected_partitions: The partitions to act on, named by their path
            relative to the view.

    Returns:
        Partition paths relative to the view, sorted alphabetically.
    """
    start = len(zview.base_dir) + 1
    return tuple(
        sorted(item[start:] for item in zview.partitions(
            filters=filters, selected_partitions=selected_partitions)))


def read_view_config(path: str,
                     fs: fsspec.AbstractFileSystem) -> dict[str, Any]:
    """Read the configuration of a view.

    Args:
        path: The path to the view.
        fs: The file system storing the view.

    Returns:
        The configuration.

    Raises:
        ValueError: If the directory contains no view configuration.
    """
    config: str = view.View._config(path)

    if not fs.exists(config):
        raise ValueError(f'View not found at path {path!r}')

    with fs.open(config) as stream:
        return json.load(stream)


def _view_metadata(data: dict[str, Any]) -> meta.DatasetSchema:
    """Read the metadata of variables defined by a view.

    Args:
        data: The configuration of the view.

    Returns:
        The variable metadata without dimensions, which are defined
        by the reference collection.
    """
    if meta.format_version(data) != meta.DEPRECATED_FORMAT_VERSION:
        return meta.DatasetSchema.from_config(data['metadata'])

    ds: meta.DatasetSchema = meta.DatasetSchema.from_deprecated_config(
        data['metadata'])
    ds.dimensions = {}
    return ds
