# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Internal utilities
==================
"""
from typing import Any, Callable, Iterator, List, Optional, Set, Tuple, Union
import time

import dask.distributed
import fsspec


def dask_workers(client: dask.distributed.Client,
                 cores_only: bool = False) -> int:
    """Return the number of dask workers available

    Args:
        client: dask client
        cores_only: if True, only the number of cores is returned,
            otherwise the total number of threads is returned.

    Returns:
        number of dask workers
    """
    return len(client.ncores()) if cores_only else sum(  # type: ignore
        item for item in client.nthreads().values())  # type: ignore


def get_client() -> dask.distributed.Client:
    """Return the default dask client

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
    """Return the file system object from the input

    Args:
        filesystem: file system object or file system name

    Returns:
        File system object.

    Example:
        >>> from fsspec.implementations.local import LocalFileSystem
        >>> get_fs("hdfs")
        >>> get_fs(LocalFileSystem("/tmp/swot"))
    """
    filesystem = filesystem or "file"
    return fsspec.filesystem(filesystem) if isinstance(filesystem,
                                                       str) else filesystem


def fs_walk(
    fs: fsspec.AbstractFileSystem,
    path: str,
    sort: bool = False,
) -> Iterator[Tuple[str, List[str], List[str]]]:
    """Return the list of files and directories in a directory

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
    except (FileNotFoundError, IOError):
        yield "", [], []
        return

    for info in listing:
        # each info name must be at least [path]/part , but here
        # we check also for names like [path]/part/
        pathname = info["name"].rstrip("/")
        name = pathname.rsplit("/", 1)[-1]
        if info["type"] == "directory" and pathname != path:
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


def _available_workers(client: dask.distributed.Client) -> Set[str]:
    """Get the list of available workers.
    Args:
        client (dask.distributed.Client): Client connected to the Dask
        cluster.

    Returns:
        list: The list of available workers.
    """
    while True:
        info = client.scheduler_info()
        result = set(info["workers"]) - set(
            k for k, v in client.processing().items() if v)  # type: ignore
        if result:
            return result
        time.sleep(0.1)


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
        list: The list of results (the order is not guaranteed to be the same
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
        return max_workers or len(client.scheduler_info()["workers"])

    result = []
    workers = set()
    iterate = True

    while iterate:
        # As long as there are workers available, we can submit tasks.
        while completed.count() < n_workers(max_workers):
            try:
                if not workers:
                    workers = _available_workers(client)
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
                result += client.gather(completed.next_batch())  # type: ignore
            except StopIteration:
                pass
    result += [item.result() for item in completed]
    return result
