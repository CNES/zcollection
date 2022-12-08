# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
File system tools
=================
"""
from __future__ import annotations

from typing import Iterator
import os

import fsspec

#: Path separator
SEPARATOR = '/'


def join_path(*args: str) -> str:
    """Join path elements."""
    return SEPARATOR.join(args)


def normalize_path(fs: fsspec.AbstractFileSystem, path: str) -> str:
    """Normalize the path.

    Args:
        fs: file system object
        path: path to test

    Returns:
        Normalized path.
    """
    # pylint: disable=protected-access
    # There is no public method to perform this operation.
    path = fs._strip_protocol(path)  # type: ignore[return-value]
    # pylint: enable=protected-access
    if path == '':
        path = fs.sep
    if fs.protocol in ('file', 'memory'):
        return os.path.normpath(path)
    return path


def get_fs(
    filesystem: fsspec.AbstractFileSystem | str | None = None
) -> fsspec.AbstractFileSystem:
    """Return the file system object from the input.

    Args:
        filesystem: file system object or file system name

    Returns:
        File system object.

    Example:
        >>> from fsspec.implementations.local import LocalFileSystem
        >>> get_fs("hdfs")
        >>> get_fs(LocalFileSystem("/tmp/swot"))
    """
    filesystem = filesystem or 'file'
    return (fsspec.filesystem(filesystem)
            if isinstance(filesystem, str) else filesystem)


def fs_walk(
    fs: fsspec.AbstractFileSystem,
    path: str,
    sort: bool = False,
) -> Iterator[tuple[str, list[str], list[str]]]:
    """Return the list of files and directories in a directory.

    Args:
        fs: file system object
        path: path to the directory
        sort: if True, the list of files and directories is sorted
            alphabetically

    Returns:
        Iterator of (path, directories, files).
    """
    dirs, files = [], []
    try:
        listing = fs.ls(path, detail=True)
    except (FileNotFoundError, OSError):
        yield '', [], []
        return

    for info in listing:
        # each info name must be at least [path]/part , but here
        # we check also for names like [path]/part/
        pathname = info['name'].rstrip('/')
        name = pathname.rsplit('/', 1)[-1]
        if info['type'] == 'directory' and pathname != path:
            # do not include "self" path
            dirs.append(pathname)
        else:
            files.append(name)

    def sort_sequence(sequence):
        """Sort the sequence if the user wishes."""
        return sorted(sequence) if sort else sequence

    dirs = sort_sequence(dirs)
    yield path, dirs, sort_sequence(files)

    for item in sort_sequence(dirs):
        yield from fs_walk(fs, item, sort=sort)
