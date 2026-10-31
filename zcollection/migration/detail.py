"""
Migration of a collection
=========================

A collection written by an earlier release is migrated by copying it into a
new one, partition by partition. A partition is read with the compressor, the
filters, the fill value, the dimensions and the attributes of each variable,
and written with them in the layout this release writes.

Each function here reads a collection and its copy. Only
:func:`prune <zcollection.migration.prune>` removes anything, and only from the source:
it removes the partitions the copy holds, and the configuration and the
immutable variables of the source stay.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
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

#: Names of the files a node holds its metadata in, in either layout. Every
#: other file an array holds is a chunk.
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
    """Gather the outcomes of several partitions into one.

    Args:
        outcomes: The outcomes to gather.

    Returns:
        The outcome holding every partition of every field, sorted.
    """
    items: tuple[Outcome, ...] = tuple(outcomes)
    return Outcome(
        tuple(sorted(name for item in items for name in item.done)),
        tuple(sorted(name for item in items for name in item.skipped)),
        tuple(sorted(name for item in items for name in item.failed)))


def _failure(partition: str, err: Exception) -> Outcome:
    """Return the outcome of a partition the run stopped on.

    Called from the handler that caught the error, so that the log carries
    the traceback.

    Args:
        partition: The path of the partition, relative to its collection.
        err: The error raised.

    Returns:
        The outcome holding the partition and the error as text.
    """
    _LOGGER.exception('The run stopped on the partition %r', partition)
    text: str = ''.join(traceback.format_exception_only(err)).strip()
    return Outcome((), (), ((partition, text), ))


def _relative_path(path: str, base_dir: str) -> str:
    """Return the path of a partition relative to its collection.

    Args:
        path: The path of the partition.
        base_dir: The directory of the collection.

    Returns:
        The relative path.
    """
    return pathlib.Path(path).relative_to(base_dir).as_posix()


def is_complete(fs: fsspec.AbstractFileSystem, dirname: str) -> bool:
    """Say whether a partition is complete.

    Args:
        fs: The file system the partition is held on.
        dirname: The path of the partition.

    Returns:
        Whether the metadata of the partition is consolidated. Consolidating
        it is the last write a partition receives.
    """
    try:
        store.open_group(fs, dirname)
    except (OSError, ValueError):
        return False
    return True


def dimension_chunks(
        arrays: Iterable[tuple[str, zarr.Array]]) -> meta.DimensionType:
    """Read the size of a chunk along each dimension a group spans.

    Args:
        arrays: The name and the array of each variable of the group.

    Returns:
        The size of a chunk along each dimension.

    Raises:
        ValueError: If two arrays hold a different chunk size along one
            dimension. A collection carries one chunk size per dimension.
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
    """Copy a Zarr group, with the chunking of the source.

    Args:
        source_dirname: The path of the group to read.
        target_dirname: The path of the group to write. A group already
            there is replaced.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Raises:
        ValueError: If the arrays of the source disagree on the chunk size
            of a dimension.
    """
    zds = storage.open_zarr_group(source_dirname, source_fs, delayed=False)

    # A dataset read from a store carries no chunking: the arrays hold it.
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
    """Copy the Zarr arrays a view holds in one of its partitions.

    Args:
        source_dirname: The path of the partition to read.
        target_dirname: The path of the partition to write. A partition
            already there is replaced.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.

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
    """Remove a directory and the cached listing of its parent.

    Args:
        fs: The file system the directory is held on.
        dirname: The path of the directory.
    """
    if fs.exists(dirname):
        fs.rm(dirname, recursive=True)
        fs.invalidate_cache(dirname)


def _differ(first: NDArray, second: NDArray) -> bool:
    """Say whether two sets of values differ.

    Args:
        first: The values read from the source.
        second: The values read from the target.

    Returns:
        Whether the two arrays mask different elements, or hold different
        values where neither masks one. Two NaNs or two NaTs count as equal.
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
    """Return the names of the arrays whose metadata calls for more chunks
    than the group holds.

    Zarr reads a chunk it cannot find as the fill value. An array that lost
    one is still declared in the consolidated metadata of its group, so the
    group is read without error and the missing chunk is read as that
    value.

    The count is sound on a collection written by an earlier release, which
    writes every chunk, including one holding nothing but the fill value.
    The collections this release writes leave such a chunk out, so a group
    of theirs holds fewer files than its grid calls for and this reports
    names it should not.

    Args:
        dirname: The path of the group.
        fs: The file system the group is held on.

    Returns:
        The names of the arrays short of a chunk, sorted.
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
    """Return the name of each node a directory holds the metadata of.

    Args:
        fs: The file system the directory is held on.
        dirname: The path of the directory.

    Returns:
        The names.
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
    """Say whether a partition of a view is written in full.

    A view holds its variables as arrays standing on their own, with no
    group over them, so the metadata a collection is gathered under has no
    equivalent here. The checksum is written after the variables, which
    makes it the mark of a partition written to the end.

    Args:
        fs: The file system the view is held on.
        dirname: The path of the partition.
        names: The variables the view declares.

    Returns:
        Whether the partition holds its checksum and every variable.
    """
    held = _nodes(fs, dirname)
    return CHECKSUM_FILE in held and set(names) <= held


def warn_absent(dirname: str, fs: fsspec.AbstractFileSystem) -> None:
    """Log the arrays whose metadata calls for more chunks than the group
    holds.

    Args:
        dirname: The path of the group.
        fs: The file system the group is held on.
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
    """Compare the values two partitions of a view hold.

    Args:
        source_dirname: The path of the partition to read.
        target_dirname: The path of the partition to compare it with.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.

    Returns:
        The names of the variables whose values differ, and of those the
        target does not hold, sorted.
    """
    held = _nodes(target_fs, target_dirname)
    differing = {name for name in names if name not in held}
    for name in set(names) - differing:
        first = store.open_array(source_fs,
                                 fs_utils.join_path(source_dirname, name),
                                 mode='r')
        second = store.open_array(target_fs,
                                  fs_utils.join_path(target_dirname, name),
                                  mode='r')
        if _differ(cast('NDArray', first[...]), cast('NDArray', second[...])):
            differing.add(name)
    return tuple(sorted(differing))


def _attributes(
        attrs: Iterable[meta.Attribute]) -> tuple[tuple[str, Any], ...]:
    """Return the attributes in an order two groups can be compared in.

    Args:
        attrs: The attributes of a variable or of a dataset.

    Returns:
        The name and the value of each, by name.
    """
    return tuple(sorted((item.name, item.value) for item in attrs))


def _chunks(
    dirname: str,
    fs: fsspec.AbstractFileSystem,
) -> dict[str, tuple[int, ...]]:
    """Return the shape of a chunk of each array a group holds.

    Args:
        dirname: The path of the group.
        fs: The file system the group is held on.

    Returns:
        The shape, by array name.
    """
    return {
        name: array.chunks
        for name, array in store.open_group(fs, dirname).arrays()
    }


def _variable_differs(first: dataset.Variable,
                      second: dataset.Variable) -> bool:
    """Say whether two variables differ in anything the copy carries over.

    Args:
        first: The variable read from the source.
        second: The variable read from the target.

    Returns:
        Whether their dimensions, their attributes, their fill value or
        their values differ.
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
    """Compare what two groups hold.

    The values of each variable are compared, masks included, and so are
    its attributes, its fill value, the name of each of its dimensions and
    the shape of its chunks. The attributes of the group itself are
    reported under the name ``attributes``.

    Args:
        source_dirname: The path of the group to read.
        target_dirname: The path of the group to compare it with.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The names of the variables that differ, and of those one group
        holds alone, sorted.
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
    """Copy one partition into the target.

    Args:
        partition: The path of the partition, relative to its collection.
        source: The directory of the source.
        target: The directory the target is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The partition under ``done`` when it was written, under ``skipped``
        when the target already held it complete, and under ``failed`` with
        the error as text when the copy stopped.
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
    """Compare the values of one partition with the ones of its copy.

    Args:
        partition: The path of the partition, relative to its collection.
        source: The directory of the source.
        target: The directory the target is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The partition under ``done`` when the two hold the same values,
        under ``skipped`` when the target does not hold it complete, and
        under ``failed`` with the names of the variables that differ.
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
        The partition under ``done`` when it was removed, under ``skipped``
        when the source no longer held it, and under ``failed`` with the
        reason when the target does not hold it complete.
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
        The partition under ``done`` when it was removed, under ``skipped``
        when the source no longer held it, and under ``failed`` with the
        reason when the target does not hold it in full.
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
    """Copy the variables a view holds in one partition, and its checksum.

    Args:
        partition: The path of the partition, relative to the view.
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.
        view_ref: The collection the target view references.

    Returns:
        The partition under ``done`` when it was written, under ``skipped``
        when the target already held it in full, and under ``failed`` with
        the error as text when the copy stopped.
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
    """Compare the variables a view holds in one partition with their copies.

    Args:
        partition: The path of the partition, relative to the view.
        source: The directory the source view is stored in.
        target: The directory the target view is stored in.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.
        names: The variables the view declares.

    Returns:
        The partition under ``done`` when the two hold the same values,
        under ``skipped`` when the target does not hold it in full, and
        under ``failed`` with the names of the variables that differ.
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
    """Apply a function to every selected partition.

    Args:
        func: The function to apply. It returns an outcome for the
            partition it is given, and raises nothing.
        partitions: The paths of the partitions, relative to their
            collection.
        distributed: Whether to spread the partitions over a Dask cluster.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        The outcomes of every partition, gathered.
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
    """List the partitions of the source a run acts on.

    Args:
        src: The source collection.
        filters: The predicate selecting the partitions.
        selected_partitions: The partitions to act on, named by their path
            relative to the collection.

    Returns:
        The paths of the partitions, relative to the source, sorted.
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
        The partitions written under ``done``, those the target already held
        complete under ``skipped``, and those the copy stopped on under
        ``failed``.
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
        The outcome of the immutable variables and that of the partitions,
        gathered.
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

        The source is read for the arrays its metadata declares and holds
        no chunk of; this module logs their names as a warning. Zarr reads
        such an array as the fill value on both sides, so comparing the
        values cannot see it.
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
    """Copy the partitions of a view into the target.

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
        The partitions written under ``done``, those the target already
        held in full under ``skipped``, and those the copy stopped on under
        ``failed``.
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
    """Compare the variables a view holds with their copies.

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
    """Remove from a view the partitions its copy holds.

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
        The partitions the target holds in full, those it does not, those
        it holds alone, and those neither view can give back. A view is
        copied whole, so a partition under ``pruned`` is one the source
        view no longer holds and the copy was never told to drop.
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
        The partitions the target holds complete, those it does not, those
        it holds alone, and those neither collection can give back.
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
    """Open the source collection, read only.

    Args:
        path: The path to the collection.
        fs: The file system the collection is held on.

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
        fs: The file system the collection is held on.

    Returns:
        The configuration.

    Raises:
        ValueError: If the directory holds no collection configuration.
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
    """Read the properties a collection is built from.

    Args:
        path: The path to the collection.
        fs: The file system the collection is held on.
        data: The configuration of the collection.

    Returns:
        The axis, the metadata and the partitioning. A configuration
        written by an earlier release names its dimensions without their
        size: these sizes are read from the first partition.

    Raises:
        ValueError: If the collection holds no partition, or if the
            metadata misses a dimension its variables span.
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

    # Using the first partition to extract dimensions information
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
    """Associate dimension to a variable containing it.

    Args:
        ds: Dataset from which to get variables information.
        main_dimension: Name of the main dimension.

    Returns:
        Dictionary of dimension associated to a variable containing it.
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

    The axis, the partitioning and the variables of the two are compared,
    which is what tells one collection from another. The size of a
    dimension is not: reading it costs an array of the source, and two
    collections are not told apart by it.

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
    """Open the collection a migration writes into.

    Args:
        source: The path to the collection to copy.
        target: The path to the collection to write.
        source_fs: The file system the source is held on.
        target_fs: The file system the target is held on.

    Returns:
        The target collection. A directory holding no configuration is
        created with the one the source describes.

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
    """Read a view and open the collection it references.

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
    """List the partitions of a view a run acts on.

    Args:
        zview: The view to list.
        filters: The predicate selecting the partitions.
        selected_partitions: The partitions to act on, named by their path
            relative to the view.

    Returns:
        The paths, relative to the view, in alphabetical order.
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
        fs: The file system the view is held on.

    Returns:
        The configuration.

    Raises:
        ValueError: If the directory holds no view configuration.
    """
    config: str = view.View._config(path)

    if not fs.exists(config):
        raise ValueError(f'View not found at path {path!r}')

    with fs.open(config) as stream:
        return json.load(stream)


def _view_metadata(data: dict[str, Any]) -> meta.DatasetSchema:
    """Read the metadata of the variables a view holds.

    Args:
        data: The configuration of the view.

    Returns:
        The metadata, without dimensions: a view takes them from the
        collection it references.
    """
    if meta.format_version(data) != meta.DEPRECATED_FORMAT_VERSION:
        return meta.DatasetSchema.from_config(data['metadata'])

    ds: meta.DatasetSchema = meta.DatasetSchema.from_deprecated_config(
        data['metadata'])
    ds.dimensions = {}
    return ds
