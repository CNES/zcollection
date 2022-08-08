# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Internal utilities
==================
"""
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import asyncio
import functools
import itertools
import operator
import os

import dask.distributed
import fsspec


def dask_workers(client: dask.distributed.Client,
                 cores_only: bool = False) -> int:
    """Return the number of dask workers available.

    Args:
        client: dask client
        cores_only: if True, only the number of cores is returned,
            otherwise the total number of threads is returned.

    Returns:
        number of dask workers

    Raises:
        ValueError: If no dask workers are available.
    """
    result = len(
        client.ncores()) if cores_only else sum(  # type: ignore[arg-type]
            item
            for item in client.nthreads().values())  # type: ignore[arg-type]
    if result == 0:
        raise RuntimeError('No dask workers available')
    return result


def get_client() -> dask.distributed.Client:
    """Return the default dask client.

    Returns:
        default dask client
    """
    try:
        return dask.distributed.get_client()
    except ValueError:
        return dask.distributed.Client()


def get_fs(
    filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None
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
    return fsspec.filesystem(filesystem) if isinstance(filesystem,
                                                       str) else filesystem


def fs_walk(
    fs: fsspec.AbstractFileSystem,
    path: str,
    sort: bool = False,
) -> Iterator[Tuple[str, List[str], List[str]]]:
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
    path = fs._strip_protocol(path)
    # pylint: enable=protected-access
    if path == '':
        path = fs.sep
    if fs.protocol in ('file', 'memory'):
        return os.path.normpath(path)
    return path


async def _available_workers(client: dask.distributed.Client) -> Set[str]:
    """Get the list of available workers.
    Args:
        client: Client connected to the Dask cluster.

    Returns:
        The list of available workers.
    """
    while True:
        info = client.scheduler_info()
        assert client.scheduler is not None
        tasks = await client.scheduler.processing(workers=None)
        result = set(info['workers']) - {k
                                         for k, v in tasks.items()
                                         if v}  # type: ignore[arg-type]
        if result:
            return result
        await asyncio.sleep(0.1)


def calculation_stream(func: Callable,
                       seq: Iterator,
                       *args,
                       max_workers: Optional[int] = None,
                       **kwargs) -> List[Any]:
    """Streams the calculation to almost n workers.

    Args:
        func: function to be applied to each element of the sequence.
        seq: sequence of elements handled by the function.
        args: positional arguments to be passed to the function.
        max_workers: The maximum number of workers allowed. If None, all
            available workers are used.
        kwargs: keyword arguments to be passed to the function.

    Returns:
        The list of results (the order is not guaranteed to be the same
        as the order of the input sequence).
    """
    client = get_client()
    completed = dask.distributed.as_completed()

    def n_workers(max_workers: Optional[int]) -> int:
        """Limit the number of workers.

        Args:
            n_workers: The maximum number of workers allowed. If None, all
                available workers are used.
        Returns:
            int: The maximum number of workers allowed.
        """
        return max_workers or len(client.scheduler_info()['workers'])

    workers: Set[str] = set()

    result = []
    iterate = True

    while iterate:
        # As long as there are workers available, we can submit tasks.
        while completed.count() < n_workers(max_workers):
            try:
                if not workers:
                    workers = client.sync(  # type: ignore[arg-type]
                        _available_workers, client)
                completed.add(
                    client.submit(func,
                                  next(seq),
                                  *args,
                                  workers=workers.pop(),
                                  allow_other_workers=False,
                                  **kwargs))
            except StopIteration:
                iterate = False
                break

        # The computation queue is full, we consume the finished jobs to be
        # able to continue.
        if iterate:
            try:
                result += client.gather(
                    completed.next_batch())  # type: ignore[arg-type]
            except StopIteration:
                pass

    result += [item.result() for item in completed]  # type: ignore[arg-type]
    return result


def split_sequence(sequence: Sequence[Any],
                   sections: Optional[int] = None) -> Iterator[Sequence[Any]]:
    """Split a sequence into sections.

    Args:
        sequence: The sequence to split.
        sections: The number of sections to split the sequence into. Default
            divides the sequence into n sections of one element.

    Returns:
        Iterator of sequences.
    """
    sections = len(sequence) if sections is None else sections
    if sections <= 0:
        raise ValueError('The number of sections must be greater than zero.')
    length = len(sequence)
    sections = min(sections, length)
    size, extras = divmod(length, sections)
    div = tuple(
        itertools.accumulate([0] + extras * [size + 1] +
                             (sections - extras) * [size]))
    yield from (sequence[item:div[ix + 1]] for ix, item in enumerate(div[:-1]))


def prod(iterable: Iterable) -> int:
    """Get the product of an iterable.

    Args:
        iterable: An iterable.

    Returns:
        The product of the iterable.
    """
    return functools.reduce(operator.mul, iterable, 1)
