# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import logging

from .. import collection, fs_utils, sync, view

if TYPE_CHECKING:
    import fsspec

#: Module logger.
_LOGGER: logging.Logger = logging.getLogger(__name__)


def create_view(path: str,
                view_ref: view.ViewReference,
                *,
                filesystem: fsspec.AbstractFileSystem | str | None = None,
                filters: collection.PartitionFilter = None,
                synchronizer: sync.Sync | None = None,
                distributed: bool = True,
                npartitions: int | None = None,
                stop_on_error: bool = True) -> view.View:
    """Create a view of a reference collection.

    Args:
        path: The view directory.
        view_ref: Access properties for the reference collection.
        filesystem: The file system for the view.
        filters: Select partitions from the reference collection. If ``None``,
            select all partitions.
        synchronizer: The synchronizer for the view.
        distributed: Whether to use Dask. Defaults to ``True``.
        npartitions: Maximum number of partition tasks in progress.
            Defaults to the number of Dask workers.
        stop_on_error: Whether to stop submitting partitions after a failure.
            Tasks already submitted to Dask are allowed to finish.
            Defaults to ``True``.

    Example:
        >>> view_ref = ViewReference(path="/data/mycollection")
        >>> view = create_view("/home/user/myview", view_ref)

    Returns:
        The new view.

    Raises:
        ValueError: If the path already exists, or if ``npartitions`` is
            below one when there are partitions to write.
        PartitionErrors: If a partition failed.
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
                     distributed=distributed,
                     npartitions=npartitions,
                     stop_on_error=stop_on_error)


def open_view(
    path: str,
    *,
    filesystem: fsspec.AbstractFileSystem | str | None = None,
    synchronizer: sync.Sync | None = None,
) -> view.View:
    """Open an existing view.

    Args:
        path: The view directory.
        filesystem: The file system for the view.
        synchronizer: The synchronizer for the view.

    Returns:
        The opened view.

    Example:
        >>> view = open_view("/home/user/myview")
    """
    return view.View.from_config(path,
                                 filesystem=filesystem,
                                 synchronizer=synchronizer)
