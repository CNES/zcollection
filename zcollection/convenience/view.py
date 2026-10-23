# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import json
import logging

import fsspec

from .. import collection, fs_utils, meta, migration, sync, view

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)


def create_view(path: str,
                view_ref: view.ViewReference,
                *,
                filesystem: fsspec.AbstractFileSystem | str | None = None,
                filters: collection.PartitionFilter = None,
                synchronizer: sync.Sync | None = None,
                distributed: bool = True) -> view.View:
    """Create a new view.

    Args:
        path: View storage directory.
        view_ref: Access properties for the reference view.
        filesystem: The file system used to access the view.
        filters: The filters used to select the partitions of the reference
            view. If not provided, all partitions are selected.
        synchronizer: The synchronizer used to synchronize the view.
        distributed: Whether to use dask or not. Default To True.

    Example:
        >>> view_ref = ViewReference(
        ...     partition_base_dir="/data/mycollection")
        >>> view = create_view("/home/user/myview", view_ref)

    Returns:
        The created view.

    Raises:
        ValueError: If the path already exists.
    """
    fs = fs_utils.get_fs(filesystem)
    if fs.exists(path):
        raise ValueError(f'path {path!r} already exists.')
    return view.View(path,
                     view_ref,
                     ds=None,
                     filesystem=fs,
                     filters=filters,
                     synchronizer=synchronizer,
                     distributed=distributed)


def open_view(
    path: str,
    *,
    filesystem: fsspec.AbstractFileSystem | str | None = None,
    synchronizer: sync.Sync | None = None,
) -> view.View:
    """Open an existing view.

    Args:
        path: View storage directory.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.

    Returns:
        The opened view.

    Example:
        >>> view = open_view("/home/user/myview")
    """
    return view.View.from_config(path,
                                 filesystem=filesystem,
                                 synchronizer=synchronizer)


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
) -> migration.Outcome:
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
        ValueError: If the source holds no view configuration.

    Example:
        >>> import zcollection
        >>> outcome = zcollection.migrate_view(
        ...     "/data/myview", "/data/myview.new",
        ...     reference="/data/mycollection.new")
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)

    data = _read_config(source, source_fs)
    ds = _view_metadata(data)

    view_ref: dict[str, Any] = data['view_ref']
    reference_fs: fsspec.AbstractFileSystem = (
        target_fs if reference is not None else
        fsspec.AbstractFileSystem.from_json(json.dumps(view_ref['fs'])))

    source_dir: str = fs_utils.normalize_path(source_fs, source)
    base_dir: str = fs_utils.normalize_path(target_fs, target)
    target_fs.makedirs(base_dir, exist_ok=True)

    zview = view.View(base_dir,
                      view.ViewReference(reference or view_ref['path'],
                                         reference_fs),
                      ds=ds,
                      filesystem=target_fs,
                      filters=view._deserialize_filters(data['filters']))
    zview._write_config()

    _, source_view = _reference(source, source_fs)
    return migration.migrate_view_partitions(source_dir,
                                             base_dir,
                                             source_fs,
                                             target_fs,
                                             partitions=_selection(
                                                 source_view, filters,
                                                 selected_partitions),
                                             names=tuple(ds.variables),
                                             view_ref=zview.view_ref,
                                             distributed=distributed)


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
    data = _read_config(path, fs)
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


def _selection(
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


def view_status(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
) -> migration.Status:
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
        >>> print(zcollection.view_status("/data/myview", "/data/myview.new"))
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)
    ds, source_view = _reference(source, source_fs)

    return migration.view_status(fs_utils.normalize_path(source_fs, source),
                                 fs_utils.normalize_path(target_fs, target),
                                 source_fs,
                                 target_fs,
                                 names=tuple(ds.variables),
                                 view_ref=source_view.view_ref)


def check_view(
    source: str,
    target: str,
    *,
    source_filesystem: fsspec.AbstractFileSystem | str | None = None,
    target_filesystem: fsspec.AbstractFileSystem | str | None = None,
    filters: collection.PartitionFilter = None,
    selected_partitions: Iterable[str] | None = None,
    distributed: bool = True,
) -> migration.Outcome:
    """Compare the variables of a view with those of the copy it was
    migrated into.

    Every value of every variable the view declares is read on both sides
    and compared, masks included. The partitions of the collection the view
    references are not read: comparing those is the job of
    :py:func:`check <zcollection.check>`.

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
        >>> outcome = zcollection.check_view(
        ...     "/data/myview", "/data/myview.new")
    """
    source_fs = fs_utils.get_fs(source_filesystem)
    target_fs = fs_utils.get_fs(target_filesystem)
    ds, source_view = _reference(source, source_fs)

    return migration.check_view_partitions(
        fs_utils.normalize_path(source_fs, source),
        fs_utils.normalize_path(target_fs, target),
        source_fs,
        target_fs,
        partitions=_selection(source_view, filters, selected_partitions),
        names=tuple(ds.variables),
        distributed=distributed)


def _read_config(path: str, fs: fsspec.AbstractFileSystem) -> dict[str, Any]:
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
