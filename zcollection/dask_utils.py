# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dask utilities
==============
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import itertools
import uuid

from dask.delayed import Delayed as dask_Delayed
import dask.distributed
import dask.highlevelgraph

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


def dask_workers(client: dask.distributed.Client,
                 cores_only: bool = False) -> int:
    """Return the number of available Dask workers or threads.

    Args:
        client: The Dask client.
        cores_only: If ``True``, return the number of workers. Otherwise,
            return the total number of threads across all workers.

    Returns:
        The number of workers or threads, according to ``cores_only``.

    Raises:
        RuntimeError: If no Dask workers are available.
    """
    result: int = len(
        client.ncores()) if cores_only else sum(  # type: ignore[arg-type]
            item
            for item in client.nthreads().values())  # type: ignore[arg-type]
    if result == 0:
        raise RuntimeError('No dask workers available')
    return result


def get_client() -> dask.distributed.Client:
    """Return the current Dask client, creating a local one if needed.

    Returns:
        The current or newly created Dask client.
    """
    try:
        return dask.distributed.get_client()
    except ValueError:
        return dask.distributed.Client(
            processes=False,
            direct_to_workers=True,
        )


def split_sequence(sequence: Sequence[Any],
                   sections: int | None = None) -> Iterator[Sequence[Any]]:
    """Split a sequence into approximately equal sections.

    Args:
        sequence: The sequence to split.
        sections: The desired number of sections. By default, create one
            section per element.

    Yields:
        Each section of the sequence.

    Raises:
        ValueError: If ``sections`` is not positive.
    """
    sections = len(sequence) if sections is None else sections
    if sections <= 0:
        raise ValueError('The number of sections must be greater than zero.')
    length: int = len(sequence)
    sections = min(sections, length)

    size: int
    extras: int
    size, extras = divmod(length, sections)

    div = tuple(
        itertools.accumulate([0] + extras * [size + 1] +
                             (sections - extras) * [size]))
    yield from (sequence[item:div[ix + 1]] for ix, item in enumerate(div[:-1]))


def simple_delayed(name: str, func: Callable) -> dask_Delayed:
    """Create a single-task Dask Delayed object.

    Args:
        name: The task name.
        func: The function to defer.

    Returns:
        The Dask Delayed object.
    """
    name = f'{name}-{uuid.uuid4()!s}'
    return dask_Delayed(
        name,
        dask.highlevelgraph.HighLevelGraph({name: {
            name: func
        }}, {name: set()}),
        None,
    )
