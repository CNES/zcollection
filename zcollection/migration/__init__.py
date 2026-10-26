# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Migration of a collection to Zarr 3
===================================

A collection written by an earlier release is read and nothing else. The
functions here copy one into a collection this release writes, and the copy
is what makes its data writable again.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import json

import fsspec

from . import detail
from .. import collection, fs_utils, view
from ..partitions import PartitionIndex
from .abc import Outcome, Status

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable

__all__ = (
    'Outcome',
    'Status',
    'check',
    'check_view',
    'migrate',
    'migrate_view',
    'prune',
    'prune_view',
    'status',
    'view_status',
)


def migrate(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Copy a collection into a new one, in the layout this release writes.

    The source is read: its partitions, its configuration and its immutable
    variables stay as they are. The first call creates the target, with the
    configuration the source describes.

    Args:
        source: The path to the collection to copy.
        target: The path to the collection to write.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system to hold the target on.
        filters: The predicate selecting the partitions to copy, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to copy, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions written under ``done``, those the target already
        held complete under ``skipped``, and those the copy stopped on
        under ``failed``, each with the error as text. The immutable
        variables are reported under the name ``.immutable``.

    Raises:
        ValueError: If the target holds a configuration that disagrees with
            the one the source describes.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.migrate("/data/old", "/data/new")
        >>> outcome.failed
        ()
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = detail._open_source(source, source_fs)
    zcol = detail._open_target(source, target, source_fs, target_fs)

    with zcol._index.updating():
        outcome = detail.migrate_collection(
            src,
            zcol.partition_properties.dir,
            target_fs,
            filters=filters,
            selected_partitions=selected_partitions,
            distributed=distributed)
        zcol._index.add(item for item in (*outcome.done, *outcome.skipped)
                        if item != collection.IMMUTABLE)
    return outcome


def status(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
) -> detail.Status:
    """Compare the partitions of a collection with those of the copy it was
    migrated into.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.

    Returns:
        The partitions both hold under ``migrated``, those of the source
        the target does not hold complete under ``missing``, and those the
        target holds alone under ``pruned``. Printing it gives the counts
        of one group per value of the first partitioning key.

    Example:
        >>> import zcollection
        >>> print(zcollection.migration.status("/data/old", "/data/new"))
        year=2014 : 365 migrated, 0 missing
        year=2015 : 212 migrated, 153 missing
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = detail._open_source(source, source_fs)

    return detail.collection_status(src,
                                    fs_utils.normalize_path(target_fs, target),
                                    target_fs)


def check(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Compare the values of a collection with those of the copy it was
    migrated into.

    Every value of every variable of the selected partitions is read and
    compared, masks included.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.
        filters: The predicate selecting the partitions to compare, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to compare, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions whose values are equal under ``done``, those the
        target does not hold complete under ``skipped``, and those whose
        variables differ under ``failed``, each with the names of those
        variables.

    Raises:
        ValueError: If one of the two directories holds no collection.
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = detail._open_source(source, source_fs)
    zcol = detail._matching_target(source, target, source_fs, target_fs)

    return detail.check_partitions(src,
                                   zcol.partition_properties.dir,
                                   target_fs,
                                   filters=filters,
                                   selected_partitions=selected_partitions,
                                   distributed=distributed)


def prune(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Remove from a collection the partitions its copy holds.

    A partition directory is removed; the files of the collection itself
    stay. Comparing the values of the two collections is the job of
    :py:func:`check <zcollection.migration.check>`.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.
        filters: The predicate selecting the partitions to remove, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to remove, named by their path
            relative to the collection.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions removed under ``done``, those the source no longer
        held under ``skipped``, and those the target does not hold complete
        under ``failed``, each with the reason.

    Raises:
        ValueError: If one of the two directories holds no collection, or
            source and target identify the same location.
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)
    detail._require_distinct_paths(source, target, source_fs, target_fs)

    src = detail._open_source(source, source_fs)
    zcol = detail._matching_target(source, target, source_fs, target_fs)

    index = PartitionIndex(source_fs,
                           src.partition_properties.dir,
                           src.partitioning,
                           writable=True)
    with index.updating():
        outcome = detail.prune_partitions(
            src,
            zcol.partition_properties.dir,
            target_fs,
            filters=filters,
            selected_partitions=selected_partitions,
            distributed=distributed)
        index.remove((*outcome.done, *outcome.skipped))
    return outcome


def migrate_view(
    source: str,
    target: str,
    *,
    reference: str | None = None,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Copy a view into a new one, in the layout this release writes.

    The source is read: the variables it holds in each partition, its
    configuration and its checksums stay as they are. In the target, the
    checksum of each copied partition is computed again from the collection
    it references.

    Args:
        source: The path to the view to copy.
        target: The path to the view to write.
        reference: The path to the collection the target references, on
            the file system the target is held on. Defaults to ``None``, which
            keeps the collection and the file system the source
            references.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.
        filters: The predicate selecting the partitions to copy. Defaults
            to ``None``, which copies every partition of the source.
        selected_partitions: The partitions to copy, named by their path
            relative to the view.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions written under ``done``, and those the copy stopped
        on under ``failed``, each with the error as text.

    Raises:
        ValueError: If the source holds no view configuration, or the target
            holds an incompatible schema or reference collection.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.migrate_view(
        ...     "/data/myview", "/data/myview.new",
        ...     reference="/data/mycollection.new")
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    data = detail.read_view_config(source, source_fs)
    ds = detail._view_metadata(data)

    view_ref: dict[str, Any] = data['view_ref']
    reference_fs: fsspec.AbstractFileSystem = (
        target_fs if reference is not None else
        fsspec.AbstractFileSystem.from_json(json.dumps(view_ref['fs'])))

    source_dir: str = fs_utils.normalize_path(source_fs, source)
    base_dir: str = fs_utils.normalize_path(target_fs, target)
    reference_path = fs_utils.normalize_path(reference_fs, reference
                                             or view_ref['path'])
    if target_fs.exists(view.View._config(base_dir)):
        target_data = detail.read_view_config(base_dir, target_fs)
        target_ref = target_data['view_ref']
        target_reference_fs = fsspec.AbstractFileSystem.from_json(
            json.dumps(target_ref['fs']))
        if (detail._view_metadata(target_data) != ds
                or target_reference_fs != reference_fs or
                fs_utils.normalize_path(target_reference_fs,
                                        target_ref['path']) != reference_path):
            raise ValueError(
                f'The view {target!r} is not configured as the one derived '
                f'from {source!r}.')
    target_fs.makedirs(base_dir, exist_ok=True)

    zview = view.View(base_dir,
                      view.ViewReference(reference_path, reference_fs),
                      ds=ds,
                      filesystem=target_fs,
                      filters=view._deserialize_filters(data['filters']),
                      _initialize=False)
    zview._write_config()

    _, source_view = detail._reference(source, source_fs)
    with zview._index.updating():
        outcome = detail.migrate_view_partitions(
            source_dir,
            base_dir,
            source_fs,
            target_fs,
            partitions=detail._view_selection(source_view, filters,
                                              selected_partitions),
            names=tuple(ds.variables),
            view_ref=zview.view_ref,
            distributed=distributed)
        zview._index.add((*outcome.done, *outcome.skipped))
    return outcome


def view_status(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
) -> detail.Status:
    """Say how far the copy of a view has got.

    Args:
        source: The path to the view that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.

    Returns:
        The partitions the copy holds in full, those it does not, those it
        holds alone, and those neither view can give back.

    Raises:
        ValueError: If the source holds no view configuration.

    Example:
        >>> import zcollection
        >>> print(zcollection.migration.view_status("/data/myview", "/data/myview.new"))
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)
    ds, source_view = detail._reference(source, source_fs)

    return detail.view_status(fs_utils.normalize_path(source_fs, source),
                              fs_utils.normalize_path(target_fs, target),
                              source_fs,
                              target_fs,
                              names=tuple(ds.variables),
                              view_ref=source_view.view_ref)


def prune_view(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Remove from a view the partitions its copy holds.

    A partition directory is removed; the configuration of the view stays.
    Comparing the values of the two views is the job of
    :py:func:`check_view <zcollection.migration.check_view>`.

    Args:
        source: The path to the view that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.
        filters: The predicate selecting the partitions to remove. Defaults
            to ``None``, which removes every partition the copy holds.
        selected_partitions: The partitions to remove, named by their path
            relative to the view.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions removed under ``done``, those the source no longer
        held under ``skipped``, and those the copy does not hold in full
        under ``failed``, each with the reason.

    Raises:
        ValueError: If the source holds no view configuration, or source
            and target identify the same location.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.prune_view(
        ...     "/data/myview", "/data/myview.new")
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)
    detail._require_distinct_paths(source, target, source_fs, target_fs)
    ds, source_view = detail._reference(source, source_fs)

    with source_view._index.updating():
        outcome = detail.prune_view_partitions(
            fs_utils.normalize_path(source_fs, source),
            fs_utils.normalize_path(target_fs, target),
            source_fs,
            target_fs,
            partitions=detail._view_selection(source_view, filters,
                                              selected_partitions),
            names=tuple(ds.variables),
            distributed=distributed)
        source_view._index.remove((*outcome.done, *outcome.skipped))
    return outcome


def check_view(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Compare the variables of a view with those of the copy it was
    migrated into.

    Every value of every variable the view declares is read on both sides
    and compared, masks included. The partitions of the collection the view
    references are not read: comparing those is the job of
    :py:func:`check <zcollection.migration.check>`.

    Args:
        source: The path to the view that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.
        filters: The predicate selecting the partitions to compare.
            Defaults to ``None``, which compares every partition of the
            source.
        selected_partitions: The partitions to compare, named by their path
            relative to the view.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        The partitions whose values are equal under ``done``, those the
        target does not hold in full under ``skipped``, and those whose
        variables differ under ``failed``, each with the names of the
        variables.

    Raises:
        ValueError: If the source holds no view configuration.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.check_view(
        ...     "/data/myview", "/data/myview.new")
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)
    ds, source_view = detail._reference(source, source_fs)

    return detail.check_view_partitions(
        fs_utils.normalize_path(source_fs, source),
        fs_utils.normalize_path(target_fs, target),
        source_fs,
        target_fs,
        partitions=detail._view_selection(source_view, filters,
                                          selected_partitions),
        names=tuple(ds.variables),
        distributed=distributed)
