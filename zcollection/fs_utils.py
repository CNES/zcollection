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
import posixpath
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
    """Normalize a file system path.

    Args:
        fs: The file system.
        path: The path to normalize.

    Returns:
        The normalized path.
    """

    # There is no public method to perform this operation.
    path = fs._strip_protocol(path)  # type: ignore[return-value]

    if path == '':
        path = fs.sep
    if fs.protocol == 'memory':
        return posixpath.normpath(path)
    if fs.protocol == 'file':
        return os.path.normpath(path)
    return path


def get_fs(
    filesystem: fsspec.AbstractFileSystem | str | None = None
) -> fsspec.AbstractFileSystem:
    """Return a file system from an instance or protocol name.

    Args:
        filesystem: A file system instance or protocol name.

    Returns:
        The file system instance.

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
    """Walk a directory tree, yielding directories and files at each level.

    Args:
        fs: The file system.
        path: The directory to traverse.
        sort: Whether to sort entries alphabetically.
        strict: Whether to raise file system errors.

    Yields:
        A tuple of ``(path, directories, files)`` for each directory.
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
        # Classify each listed entry as a directory or a file.
        dirs.append(name) if is_dir else files.append(
            name.rsplit(SEPARATOR, 1)[-1])

    def sort_sequence(sequence: list[str]) -> list[str]:
        """Sort the sequence if requested."""
        return sorted(sequence) if sort else sequence

    dirs = sort_sequence(dirs)
    yield path.rstrip(SEPARATOR), dirs, sort_sequence(files)

    for item in dirs:
        yield from fs_walk(fs, item, sort=sort, strict=strict)


def remove_tree(fs: fsspec.AbstractFileSystem, path: str) -> None:
    """Remove a directory and its contents.

    If the directory is absent, return without raising an exception.

    Args:
        fs: The file system containing the directory.
        path: The directory to remove.
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

    For local files, including those accessed through ``DirFileSystem``,
    the new contents are written to a temporary file beside the target,
    then atomically moved into place. If writing or replacement fails,
    the original contents are preserved. Other file systems use
    ``pipe_file``; their guarantees depend on the backend.

    Args:
        fs: The file system containing the file.
        path: The path of the file.
        data: The new contents.
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
    """Copy a file between locations.

    Args:
        source: The source file path.
        target: The target file path.
        fs_source: The source file system.
        fs_target: The target file system.
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
    """Copy files to another directory.

    Args:
        source: The paths of the source files.
        target: The target directory.
        fs_source: The source file system.
        fs_target: The target file system.
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
    """Copy a directory tree to another location.

    If copying fails, remove the target before raising the error again.

    Args:
        source: The source directory.
        target: The target directory.
        fs_source: The source file system.
        fs_target: The target file system.

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
