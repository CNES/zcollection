# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dask utilities
==============
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence
import asyncio
import itertools

import dask.distributed


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


async def _available_workers(client: dask.distributed.Client) -> set[str]:
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
        assert tasks is not None
        result = set(info['workers']) - {k
                                         for k, v in tasks.items()
                                         if v}  # type: ignore[arg-type]
        if result:
            return result
        await asyncio.sleep(0.1)


def calculation_stream(func: Callable,
                       seq: Iterator,
                       *args,
                       max_workers: int | None = None,
                       **kwargs) -> list[Any]:
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

    def n_workers(max_workers: int | None) -> int:
        """Limit the number of workers.

        Args:
            n_workers: The maximum number of workers allowed. If None, all
                available workers are used.
        Returns:
            int: The maximum number of workers allowed.
        """
        return max_workers or len(client.scheduler_info()['workers'])

    workers: set[str] = set()

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
                   sections: int | None = None) -> Iterator[Sequence[Any]]:
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
