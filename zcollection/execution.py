# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Execution of an operation on partitions
=======================================

An operation runs one task per partition. With Dask, at most
``npartitions`` tasks run at the same time: a task is submitted when
another one finishes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple
import contextlib
import traceback
import uuid

import distributed

from . import dask_utils

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

#: The largest number of failed partitions named in the message of
#: PartitionErrors.
_NAMED_FAILURES = 10


class PartitionErrors(Exception):
    """Error raised when partitions of an operation fail.

    A failed partition can be partly modified. The message names up to ten
    failed partitions, counts the others and gives the traceback of the
    first, in the order the operation processes the partitions.

    Attributes:
        completed: The paths of the partitions processed, relative to the
            collection or the view. Skipped partitions are left out.
        failed: The traceback of the error of each failed partition, keyed
            by path.
        not_attempted: The paths of the partitions the operation did not
            attempt.
    """

    def __init__(self, completed: Sequence[str], failed: Mapping[str, str],
                 not_attempted: Sequence[str]) -> None:
        self.completed = tuple(completed)
        self.failed = dict(failed)
        self.not_attempted = tuple(not_attempted)
        total = (len(self.completed) + len(self.failed) +
                 len(self.not_attempted))
        names = ', '.join(list(self.failed)[:_NAMED_FAILURES])
        if len(self.failed) > _NAMED_FAILURES:
            names += f' and {len(self.failed) - _NAMED_FAILURES} others'
        first = next(iter(self.failed.values()), '')
        super().__init__(
            f'{len(self.failed)} of {total} partitions failed: {names}\n\n'
            f'{first}')

    def __reduce__(self) -> tuple[Any, ...]:
        # Pickle rebuilds an exception from `args`, which holds only the
        # message here.
        return type(self), (self.completed, self.failed, self.not_attempted)


class Skipped:
    """The result of a task that skipped its partition."""


#: The value a task returns when it skips its partition. `run` leaves that
#: partition out of its result and out of PartitionErrors.
SKIPPED = Skipped()


class _Failure(NamedTuple):
    """The result of a task whose partition failed."""
    trace: str


def _call(func: Callable[..., Any], item: Any, kwargs: dict[str, Any]) -> Any:
    """Return `func(item, **kwargs)`, or the traceback of its exception as a
    _Failure."""
    try:
        return func(item, **kwargs)
    except Exception as err:
        return _Failure(''.join(traceback.format_exception(err)))


def run(func: Callable[..., Any],
        items: Iterable[Any],
        names: Sequence[str],
        *,
        client: distributed.Client | None,
        npartitions: int | None,
        stop_on_error: bool,
        kwargs: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Process each partition with `func`, one task per partition.

    With a client, `run` returns or raises once every submitted task has
    finished, unless a second interruption arrives while it waits for them.

    Args:
        func: The function processing one partition, called as
            ``func(item, **kwargs)``. It returns :data:`SKIPPED` when it
            skips the partition.
        items: The argument of `func` for each partition.
        names: The name of each partition, in the order of `items`.
        client: The Dask client, or ``None`` to process the partitions one
            after the other in this process.
        npartitions: The largest number of partitions processed at the same
            time with Dask. Defaults to the number of workers.
        stop_on_error: If True, no partition starts after the first
            failure, and the partitions in progress finish.
        kwargs: Keyword arguments passed to `func`.

    Returns:
        The names of the partitions processed, in the order of `items`.
        Skipped partitions are left out.

    Raises:
        ValueError: If `npartitions` is below one.
        PartitionErrors: If a partition failed.
    """
    check_npartitions(npartitions)
    items = list(items)
    kwargs = dict(kwargs or {})
    if client is None:
        outcomes = _run_sequential(func, items, stop_on_error, kwargs)
    else:
        outcomes = _run_distributed(
            client, func, items, npartitions
            or dask_utils.dask_workers(client, cores_only=True), stop_on_error,
            kwargs)
    completed, failed, not_attempted = _report(outcomes, names)
    if failed:
        raise PartitionErrors(completed, failed, not_attempted)
    return completed


def check_npartitions(npartitions: int | None) -> None:
    """Check that `npartitions` is None or at least one.

    Raises:
        ValueError: If `npartitions` is below one.
    """
    if npartitions is not None and npartitions < 1:
        raise ValueError('npartitions must be at least one')


def _report(
    outcomes: Mapping[int, Any], names: Sequence[str]
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    """Return the partitions completed, failed and not attempted.

    Args:
        outcomes: The result of each submitted partition, by its index.
        names: The name of each partition.
    """
    completed: list[str] = []
    failed: dict[str, str] = {}
    not_attempted: list[str] = []
    for index, name in enumerate(names):
        if index not in outcomes:
            not_attempted.append(name)
        elif isinstance(outcomes[index], _Failure):
            failed[name] = outcomes[index].trace
        elif not isinstance(outcomes[index], Skipped):
            completed.append(name)
    return tuple(completed), failed, tuple(not_attempted)


def _run_sequential(func: Callable[..., Any], items: list[Any],
                    stop_on_error: bool, kwargs: dict[str,
                                                      Any]) -> dict[int, Any]:
    """Process the partitions one after the other in this process."""
    outcomes: dict[int, Any] = {}
    for index, item in enumerate(items):
        outcomes[index] = _call(func, item, kwargs)
        if stop_on_error and isinstance(outcomes[index], _Failure):
            break
    return outcomes


def _run_distributed(client: distributed.Client, func: Callable[..., Any],
                     items: list[Any], width: int, stop_on_error: bool,
                     kwargs: dict[str, Any]) -> dict[int, Any]:
    """Process the partitions with Dask, at most `width` at a time."""
    outcomes: dict[int, Any] = {}
    in_flight: dict[distributed.Future, int] = {}
    pending = iter(range(len(items)))
    prefix = getattr(func, '__name__', 'partition')
    finished = distributed.as_completed()

    def submit() -> None:
        """Submit the next partition, if one is left."""
        next_index = next(pending, None)
        if next_index is not None:
            submitted = client.submit(_call,
                                      func,
                                      items[next_index],
                                      kwargs,
                                      key=f'{prefix}-{uuid.uuid4().hex}',
                                      pure=False)
            in_flight[submitted] = next_index
            finished.add(submitted)

    try:
        for _ in range(min(width, len(items))):
            submit()
        stop = False
        while in_flight:
            future = next(finished)
            index = in_flight.pop(future)
            try:
                outcomes[index] = future.result()
            except Exception as err:
                # Dask itself failed, for example when a worker is lost.
                outcomes[index] = _Failure(''.join(
                    traceback.format_exception(err)))
            if stop_on_error and isinstance(outcomes[index], _Failure):
                stop = True
            if not stop:
                submit()
    except BaseException:
        # An interruption, or an error of the client or of a submission:
        # wait for the tasks in flight, then raise the same exception. After
        # a second interruption during the wait, the function leaves at once.
        with contextlib.suppress(Exception):
            distributed.wait(list(in_flight))
        raise
    return outcomes
