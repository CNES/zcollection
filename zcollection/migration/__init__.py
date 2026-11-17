# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Migration of a collection to Zarr 3
===================================

Collections written by earlier releases can only be opened read-only.
These functions copy them into the current layout, making the copied
data writable again.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import json

import fsspec

from . import detail
from .. import collection, execution, fs_utils, view
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
    npartitions: int | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Copy a collection into the current storage layout.

    The source remains unchanged, including its partitions, configuration
    and immutable variables. The first call creates the target using the
    source configuration.

    Args:
        source: The path to the collection to copy.
        target: The path to the collection to write.
        source_filesystem: The file system storing the source collection.
        target_filesystem: The file system for the target collection.
        filters: The predicate selecting the partitions to copy, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to copy, named by their path
            relative to the collection.
        npartitions: Maximum number of partition tasks in progress.
            Defaults to the number of Dask workers. With
            ``distributed=False``, partitions run sequentially.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        An outcome with copied partitions under ``done``, partitions
        already complete in the target under ``skipped``, and failed
        copies under ``failed`` with their error messages. Immutable
        variables are reported as ``.immutable``.

    Raises:
        ValueError: If ``npartitions`` is below one, or if the target
            configuration differs from the configuration derived from
            the source.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.migrate("/data/old", "/data/new")
        >>> outcome.failed
        ()
    """
    execution.check_npartitions(npartitions)

    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = detail._open_source(source, source_fs)
    detail.warn_unstable_dtypes(src.metadata, source)
    zcol = detail._open_target(source, target, source_fs, target_fs)

    with zcol._index.updating():
        outcome = detail.migrate_collection(
            src,
            zcol.partition_properties.dir,
            target_fs,
            filters=filters,
            selected_partitions=selected_partitions,
            npartitions=npartitions,
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
    """Compare a collection with its migrated copy.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.

    Returns:
        A status grouping partitions as follows: ``migrated`` for those
        present in the source and complete in the target; ``missing`` for
        those present in the source but absent or incomplete in the target;
        ``pruned`` for those absent from the source but complete in the
        target; and ``lost`` for those absent from the source and incomplete
        in the target. Its string representation groups counts by the
        first partitioning key.

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
    npartitions: int | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Compare a collection with its migrated copy.

    All values in the selected partitions are compared, including masks.

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
        npartitions: Maximum number of partition tasks in progress.
            Defaults to the number of Dask workers. With
            ``distributed=False``, partitions run sequentially.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        An outcome with matching partitions under ``done``, partitions
        missing or incomplete in the target under ``skipped``, and
        differing partitions under ``failed`` with the names of the
        variables that differ.

    Raises:
        ValueError: If ``npartitions`` is below one, either directory contains
            no collection, or the target is not a copy of the source.
    """
    execution.check_npartitions(npartitions)

    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    src = detail._open_source(source, source_fs)
    zcol = detail._matching_target(source, target, source_fs, target_fs)

    return detail.check_partitions(src,
                                   zcol.partition_properties.dir,
                                   target_fs,
                                   filters=filters,
                                   selected_partitions=selected_partitions,
                                   npartitions=npartitions,
                                   distributed=distributed)


def prune(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    npartitions: int | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Remove source partitions already copied to the target.

    Only partition directories are removed; the collection files remain.
    To compare values before pruning, use
    :py:func:`check <zcollection.migration.check>`.

    Args:
        source: The path to the collection that was copied.
        target: The path to the copy.
        source_filesystem: The file system storing the source collection.
        target_filesystem: The file system storing the target collection.
        filters: The predicate selecting the partitions to remove, as
            :py:meth:`Collection.partitions
            <zcollection.collection.abc.ReadOnlyCollection.partitions>`
            reads it.
        selected_partitions: The partitions to remove, named by their path
            relative to the collection.
        npartitions: Maximum number of partition tasks in progress.
            Defaults to the number of Dask workers. With
            ``distributed=False``, partitions run sequentially.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        An outcome with removed partitions under ``done``, partitions
        already absent from the source under ``skipped``, and partitions
        without a complete target copy under ``failed``, with the reason.

    Raises:
        ValueError: If ``npartitions`` is below one, one of the two directories
            holds no collection, or source and target identify the same
            location.
    """
    execution.check_npartitions(npartitions)

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
            npartitions=npartitions,
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
    npartitions: int | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Copy a view into the current storage layout.

    The source remains unchanged, including its variables, configuration
    and checksums. The target recalculates each copied partition's
    checksum from its reference collection.

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
        npartitions: Maximum number of partition tasks in progress.
            Defaults to the number of Dask workers. With
            ``distributed=False``, partitions run sequentially.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        An outcome with copied partitions under ``done``, partitions
        already complete in the target under ``skipped``, and failed
        copies under ``failed`` with their error messages.

    Raises:
        ValueError: If ``npartitions`` is below one, the source holds no view
            configuration, or the target holds an incompatible schema or
            reference collection.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.migrate_view(
        ...     "/data/myview", "/data/myview.new",
        ...     reference="/data/mycollection.new")
    """
    execution.check_npartitions(npartitions)

    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    data = detail.read_view_config(source, source_fs)
    ds = detail._view_metadata(data)
    detail.warn_unstable_dtypes(ds, source)

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
            npartitions=npartitions,
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
    """Report the migration status of a view.

    Args:
        source: The path to the view that was copied.
        target: The path to the copy.
        source_filesystem: The file system the source is held on.
        target_filesystem: The file system the target is held on.

    Returns:
        A status grouping partitions as follows: ``migrated`` for those
        present in the source and complete in the target; ``missing`` for
        those present in the source but absent or incomplete in the target;
        ``pruned`` for those absent from the source but complete in the
        target; and ``lost`` for those absent from the source and incomplete
        in the target.

    Raises:
        ValueError: If the source holds no view configuration.

    Example:
        >>> import zcollection
        >>> print(zcollection.migration.view_status(
        ...     "/data/myview", "/data/myview.new"))
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
    npartitions: int | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Remove source view partitions already copied to the target.

    Only partition directories are removed; the view configuration
    remains. To compare values before pruning, use
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
        npartitions: Maximum number of partition tasks in progress.
            Defaults to the number of Dask workers. With
            ``distributed=False``, partitions run sequentially.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        An outcome with removed partitions under ``done``, partitions
        already absent from the source under ``skipped``, and partitions
        without a complete target copy under ``failed``, with the reason.

    Raises:
        ValueError: If ``npartitions`` is below one, the source holds no view
            configuration, or source and target identify the same location.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.prune_view(
        ...     "/data/myview", "/data/myview.new")
    """
    execution.check_npartitions(npartitions)

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
            npartitions=npartitions,
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
    npartitions: int | None = None,
    distributed: bool = True,
) -> detail.Outcome:
    """Compare the variables of a view with their migrated copies.

    Every variable defined by the view is compared, including values and
    masks. Variables from the reference collection are not compared; use
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
        npartitions: Maximum number of partition tasks in progress.
            Defaults to the number of Dask workers. With
            ``distributed=False``, partitions run sequentially.
        distributed: Whether to spread the partitions over a Dask cluster.
            Defaults to ``True``.

    Returns:
        An outcome with matching partitions under ``done``, partitions
        missing or incomplete in the target under ``skipped``, and
        differing partitions under ``failed`` with the names of the
        variables that differ.

    Raises:
        ValueError: If ``npartitions`` is below one, or the source holds no
            view configuration.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migration.check_view(
        ...     "/data/myview", "/data/myview.new")
    """
    execution.check_npartitions(npartitions)

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
        npartitions=npartitions,
        distributed=distributed)
