# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from __future__ import annotations

from typing import Any
import logging

import fsspec

from .. import collection, fs_utils, meta, sync, view

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
    filesystem = fs_utils.get_fs(filesystem)
    if filesystem.exists(path):
        raise ValueError(f'path {path!r} already exists.')
    return view.View(path,
                     view_ref,
                     ds=None,
                     filesystem=filesystem,
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


def update_deprecated_view(
        path: str,
        filesystem: fsspec.AbstractFileSystem | str | None = None) -> None:
    """Update deprecated view's configuration. A backup of the existing view
    configuration will be kept.

    Args:
        path: The path to the view.
        filesystem: The filesystem to use for the view. This is an
            instance of a subclass of :py:class:`fsspec.AbstractFileSystem`.

        Raises:
            ValueError:
                If the provided directory does not contain a valid view
                configuration file.
    """

    import json

    _LOGGER.warning('Updating view: %r', path)
    fs = fs_utils.get_fs(filesystem)

    config = view.View._config(path)

    if not fs.exists(config):
        raise ValueError(f'View not found at path {path!r}')

    with fs.open(config) as stream:
        data: dict[str, Any] = json.load(stream)

    if data.get('version', '0') != '0':
        _LOGGER.error('View already updated.')
        return

    ds = meta.Dataset.from_deprecated_config(data['metadata'])
    # Views do not contain dimensions
    ds.dimensions = {}

    view_ref: dict[str, Any] = data['view_ref']

    zview = view.View(data['base_dir'],
                      view.ViewReference(
                          view_ref['path'],
                          fsspec.AbstractFileSystem.from_json(
                              json.dumps(view_ref['fs']))),
                      ds=ds,
                      filesystem=filesystem,
                      filters=view._deserialize_filters(data['filters']))

    config_back = f'{config}.bak'
    _LOGGER.warning('Copying old configuration to: %r', config_back)
    fs.copy(config, config_back)

    _LOGGER.warning('Writing new configuration: %r', path)

    zview._write_config()
