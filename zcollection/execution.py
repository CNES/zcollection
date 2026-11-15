# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Executing operations on partitions
==================================

Process one partition per task. With Dask, keep at most ``npartitions``
tasks in flight, submitting a new task whenever one finishes.
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

#: Maximum number of failed partitions named in a PartitionErrors message.
_NAMED_FAILURES = 10


class PartitionErrors(Exception):
    """Error raised when an operation fails on one or more partitions.

    A failed partition may be partly modified. The message names up to ten
    failed partitions, counts any additional failures, and includes the
    traceback of the first failure in the order passed to the executor.

    Attributes:
        completed: The paths of processed partitions, relative to the
            collection or view. Skipped partitions are omitted.
        failed: Error tracebacks as text, keyed by partition path.
        not_attempted: The paths of partitions not attempted.
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
        # Pickle reconstructs an exception from ``args``, which contains only
        # the message here.
        return type(self), (self.completed, self.failed, self.not_attempted)


class Skipped:
    """The result of a task that skipped its partition."""


#: Result returned when a task skips its partition. ``run`` excludes the
#: partition from its return value and from PartitionErrors.
SKIPPED = Skipped()


class _Failure(NamedTuple):
    """The result of a task that failed to process its partition."""
    trace: str


def _call(func: Callable[..., Any], item: Any, kwargs: dict[str, Any]) -> Any:
    """Return ``func(item, **kwargs)`` or a ``_Failure`` with its traceback."""
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
    """Process each partition with ``func``, one task per partition.

    With a Dask client, wait for submitted tasks to finish, even after an
    error or interruption. A second interruption stops the wait, but tasks
    already submitted to Dask may continue running.

    Args:
        func: The function called for each partition as
            ``func(item, **kwargs)``. Return :data:`SKIPPED` to skip one.
        items: The argument passed to ``func`` for each partition.
        names: The partition names, in the order of ``items``.
        client: The Dask client, or ``None`` to process partitions
            sequentially in the calling process.
        npartitions: The maximum number of partition tasks in progress
            with Dask. Defaults to the number of workers.
        stop_on_error: Whether to stop submitting new partitions after
            the first failure. Already submitted tasks can finish.
        kwargs: Keyword arguments passed to ``func``.

    Returns:
        The names of processed partitions, in the order of ``items``.
        Skipped partitions are omitted.

    Raises:
        ValueError: If ``npartitions`` is below one.
        PartitionErrors: If a partition fails.
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
    """Process partitions with Dask, up to ``width`` tasks in flight."""
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
        # On interruption or client/submission error, wait for submitted tasks
        # to finish before re-raising the exception. A second interruption
        # during the wait exits immediately.
        with contextlib.suppress(Exception):
            distributed.wait(list(in_flight))
        raise
    return outcomes
