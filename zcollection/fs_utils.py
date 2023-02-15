# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
File system tools
=================
"""
from __future__ import annotations

from typing import Iterator, Sequence
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
        pathname = info['name'].rstrip(SEPARATOR)
        name = pathname.rsplit(SEPARATOR, 1)[-1]
        if info['type'] == 'directory' and pathname != path:
            # do not include "self" path
            dirs.append(pathname)
        else:
            files.append(name)

    def sort_sequence(sequence):
        """Sort the sequence if the user wishes."""
        return sorted(sequence) if sort else sequence

    dirs = sort_sequence(dirs)
    yield path.rstrip(SEPARATOR), dirs, sort_sequence(files)

    for item in sort_sequence(dirs):
        yield from fs_walk(fs, item, sort=sort)


def copy_file(
    source: str,
    target: str,
    fs_source: fsspec.AbstractFileSystem,
    fs_target: fsspec.AbstractFileSystem,
) -> None:
    """Copy a file from one location to another.

    Args:
        source: The name of the source file.
        target: The name of the target file.
        fs_source: The file system that the source file is stored on.
        fs_target: The file system that the target file is stored on.
    """
    with fs_source.open(source, 'rb') as source_stream:
        with fs_target.open(target, 'wb') as target_stream:
            target_stream.write(source_stream.read())


def copy_files(
    source: Sequence[str],
    target: str,
    fs_source: fsspec.AbstractFileSystem,
    fs_target: fsspec.AbstractFileSystem,
) -> None:
    """Copy a list of files from one location to another.

    Args:
        source: The names of the source files.
        target: The name of the target directory.
        fs_source: The file system that the source files are stored on.
        fs_target: The file system that the target directory is stored on.
    """
    tuple(
        map(
            lambda path: copy_file(path,
                                   join_path(target, os.path.basename(path)),
                                   fs_source, fs_target), source))


def copy_tree(
    source: str,
    target: str,
    fs_source: fsspec.AbstractFileSystem,
    fs_target: fsspec.AbstractFileSystem,
) -> None:
    """Copy a directory tree from one location to another.

    Args:
        source: The name of the source directory.
        target: The name of the target directory.
        fs_source: The file system that the source directory is stored on.
        fs_target: The file system that the target directory is stored on.

    Raises:
        ValueError: If the target already exists.
    """
    if fs_target.exists(target):
        raise ValueError(f'Target {target} already exists')
    fs_target.mkdir(target)
    for root, dirs, files in tuple(fs_walk(fs_source, source)):
        for name in files:
            source_path = join_path(root, name)
            copy_file(source_path,
                      join_path(target, os.path.relpath(source_path, source)),
                      fs_source, fs_target)
        for source_path in dirs:
            fs_target.mkdir(
                join_path(target, os.path.relpath(source_path, source)))
