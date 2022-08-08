# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from typing import Optional, Union

import fsspec

from .. import sync, utilities, view


def create_view(
    path: str,
    view_ref: view.ViewReference,
    *,
    filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
    synchronizer: Optional[sync.Sync] = None,
) -> view.View:
    """Create a new view.

    Args:
        path: View storage directory.
        view_ref: Access properties for the reference view.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.

    Example:
        >>> view_ref = ViewReference(
        ...     partition_base_dir="/data/mycollection")
        >>> view = create_view("/home/user/myview", view_ref)

    Returns:
        The created view.

    Raises:
        ValueError: If the path already exists.
    """
    filesystem = utilities.get_fs(filesystem)
    if filesystem.exists(path):
        raise ValueError(f'path {path!r} already exists.')
    return view.View(path,
                     view_ref,
                     ds=None,
                     filesystem=filesystem,
                     synchronizer=synchronizer)


def open_view(
    path: str,
    *,
    filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
    synchronizer: Optional[sync.Sync] = None,
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
