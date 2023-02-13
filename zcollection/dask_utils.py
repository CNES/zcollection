# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dask utilities
==============
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence
import itertools
import uuid

from dask.delayed import Delayed as dask_Delayed
import dask.distributed
import dask.highlevelgraph


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


def simple_delayed(name: str, func: Callable) -> dask_Delayed:
    """Create a simple delayed function.

    Args:
        name: name of the function
        func: function to be delayed

    Returns:
        delayed function
    """
    name = f'{name}-{str(uuid.uuid4())}'
    return dask_Delayed(
        name,
        dask.highlevelgraph.HighLevelGraph({name: {
            name: func
        }}, {name: set()}),
        None,
    )
