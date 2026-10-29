# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
File system tools
=================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import os
import uuid

import fsspec
import fsspec.implementations.dirfs
import fsspec.implementations.local

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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

    # There is no public method to perform this operation.
    path = fs._strip_protocol(path)  # type: ignore[return-value]

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
    *,
    strict: bool = False,
) -> Iterator[tuple[str, list[str], list[str]]]:
    """Return the list of files and directories in a directory.

    Args:
        fs: file system object
        path: path to the directory
        sort: if True, the list of files and directories is sorted
            alphabetically
        strict: Whether file system errors are raised.

    Returns:
        Iterator of (path, directories, files).
    """
    dirs: list[str]
    files: list[str]

    dirs, files = [], []
    try:
        listing: list[dict[str, Any]] = fs.ls(path, detail=True)
    except OSError:
        if strict:
            raise
        yield '', [], []
        return

    for is_dir, name in ((info['type'] == 'directory', info['name'])
                         for info in listing):
        # each info name must be at least [path]/part , but here
        # we check also for names like [path]/part/
        dirs.append(name) if is_dir else files.append(
            name.rsplit(SEPARATOR, 1)[-1])

    def sort_sequence(sequence: list[str]) -> list[str]:
        """Sort the sequence if the user wishes."""
        return sorted(sequence) if sort else sequence

    dirs = sort_sequence(dirs)
    yield path.rstrip(SEPARATOR), dirs, sort_sequence(files)

    for item in dirs:
        yield from fs_walk(fs, item, sort=sort, strict=strict)


def remove_tree(fs: fsspec.AbstractFileSystem, path: str) -> None:
    """Remove a directory and what it holds.

    If the directory is absent, the function returns without raising an
    exception.

    Args:
        fs: The file system holding the directory.
        path: The path of the directory.
    """
    try:
        fs.rm(path, recursive=True)
    except FileNotFoundError:
        if fs.exists(path):
            raise


def replace_file(
    fs: fsspec.AbstractFileSystem,
    path: str,
    data: str | bytes,
) -> None:
    """Replace the contents of a file.

    Local files, accessed directly or through ``DirFileSystem``, are written
    to a neighboring temporary file and replaced atomically. A failed write
    or replacement preserves the previous contents. Other filesystems receive
    a direct ``pipe_file`` call; replacement guarantees depend on the backend.

    Args:
        fs: The file system holding the file.
        path: The path of the file.
        data: What the file is to hold.
    """
    content: bytes = data.encode() if isinstance(data, str) else data
    backend_fs = fs
    backend_path = path
    while isinstance(backend_fs, fsspec.implementations.dirfs.DirFileSystem):
        if backend_fs.path:
            backend_path = join_path(backend_fs.path,
                                     normalize_path(backend_fs, backend_path))
        backend_fs = backend_fs.fs
    if not isinstance(backend_fs,
                      fsspec.implementations.local.LocalFileSystem):
        fs.pipe_file(path, content)
        return

    local_path = normalize_path(backend_fs, backend_path)
    suffix = f'.{uuid.uuid4().hex}'
    beside: str = path + suffix
    try:
        fs.pipe_file(beside, content)
        os.replace(local_path + suffix, local_path)
    except Exception:
        if fs.exists(beside):
            fs.rm(beside)
        raise


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
            target_stream.write(source_stream.read())  # type: ignore[arg-type]


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
        copy_file(  # type: ignore[func-returns-value]
            path,
            join_path(target, os.path.basename(path)),
            fs_source,
            fs_target,
        ) for path in source)


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
    listing = tuple(fs_walk(fs_source, source, strict=True))
    fs_target.mkdir(target)
    try:
        for root, dirs, files in listing:
            for name in files:
                source_path = join_path(root, name)
                copy_file(
                    source_path,
                    join_path(target, os.path.relpath(source_path, source)),
                    fs_source, fs_target)
            for source_path in dirs:
                fs_target.mkdir(
                    join_path(target, os.path.relpath(source_path, source)))
    except Exception:
        remove_tree(fs_target, target)
        raise
