# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests of the execution of an operation, one task per partition."""
from __future__ import annotations

from typing import TYPE_CHECKING
import functools
import json
import os
import pickle
import time
import uuid

import dask.config
import dask.distributed
import pytest

from .. import execution
from .cluster import dask_client, dask_cluster  # noqa: F401

if TYPE_CHECKING:
    import pathlib

NAMES = tuple(f'p{index}' for index in range(10))


@pytest.fixture(params=['sequential', 'distributed'])
def client(request):
    """Run without Dask, or with the client of the test cluster."""
    if request.param == 'sequential':
        return None
    return request.getfixturevalue('dask_client')


def record(item: int,
           *,
           track: str,
           fail: tuple[int, ...] = (),
           delay: float = 0.0) -> None:
    """Fail on the items `fail`; otherwise sleep and record the call."""
    start = time.time()
    if item in fail:
        raise ValueError(f'item {item} failed')
    time.sleep(delay)
    with open(os.path.join(track, uuid.uuid4().hex), 'w') as stream:
        json.dump([item, start, time.time()], stream)


def recorded(track: pathlib.Path) -> list[tuple[int, float, float]]:
    """Return the item, start and stop of each recorded call."""
    return [
        tuple(json.loads(path.read_text()))  # type: ignore[misc]
        for path in track.iterdir()
    ]


def items(track: pathlib.Path) -> list[int]:
    """Return the items of the recorded calls."""
    return sorted(item for item, _, _ in recorded(track))


def peak(calls: list[tuple[int, float, float]]) -> int:
    """Return the largest number of calls running at the same time."""
    events = sorted([(start, 1)
                     for _, start, _ in calls] + [(stop, -1)
                                                  for _, _, stop in calls])
    current = largest = 0
    for _, change in events:
        current += change
        largest = max(largest, current)
    return largest


def test_window_bounds_the_partitions_in_flight(client, tmp_path) -> None:
    """At most `npartitions` partitions run at the same time, and one
    without Dask."""
    written = execution.run(functools.partial(record,
                                              track=str(tmp_path),
                                              delay=0.05),
                            range(10),
                            NAMES,
                            client=client,
                            npartitions=3,
                            stop_on_error=True)

    assert written == NAMES
    assert peak(recorded(tmp_path)) <= (3 if client else 1)


def test_npartitions_below_one_is_refused() -> None:
    """``npartitions`` below one is refused."""
    with pytest.raises(ValueError, match='npartitions'):
        execution.run(print,
                      range(10),
                      NAMES,
                      client=None,
                      npartitions=0,
                      stop_on_error=True)


@pytest.mark.parametrize(
    ('stop_on_error', 'completed', 'failed', 'not_attempted'), [
        (True, NAMES[:3], ['p3'], NAMES[4:]),
        (False, NAMES[:3] + NAMES[4:6] + NAMES[7:], ['p3', 'p6'], ()),
    ])
def test_report(client, tmp_path, stop_on_error, completed, failed,
                not_attempted) -> None:
    """The error gives the partitions completed, failed and not attempted,
    in the order of the operation, and the traceback of the first failure."""
    with pytest.raises(execution.PartitionErrors) as info:
        execution.run(functools.partial(record,
                                        track=str(tmp_path),
                                        fail=(3, 6)),
                      range(10),
                      NAMES,
                      client=client,
                      npartitions=1,
                      stop_on_error=stop_on_error)

    errors = info.value
    assert errors.completed == completed
    assert list(errors.failed) == failed
    assert 'ValueError: item 3 failed' in errors.failed['p3']
    assert errors.not_attempted == not_attempted
    assert items(tmp_path) == [int(name[1:]) for name in completed]
    assert errors.failed['p3'] in str(errors)
    copy = pickle.loads(pickle.dumps(errors))
    assert (copy.completed, copy.failed,
            copy.not_attempted) == (errors.completed, errors.failed,
                                    errors.not_attempted)


def test_returns_once_the_tasks_in_flight_finish(
        dask_client,  # noqa: F811
        tmp_path,
        monkeypatch) -> None:
    """After a failure, the tasks already running finish before the error
    reaches the caller."""
    release = tmp_path.parent / f'{tmp_path.name}.release'

    class ReleasingAsCompleted(dask.distributed.as_completed):
        """Release the waiting tasks once run receives a finished
        task."""

        def __next__(self):
            future = super().__next__()
            release.touch()
            return future

    monkeypatch.setattr(execution.distributed, 'as_completed',
                        ReleasingAsCompleted)
    try:
        with pytest.raises(execution.PartitionErrors) as info:
            execution.run(functools.partial(record_on_release,
                                            track=str(tmp_path),
                                            release=str(release),
                                            fail=(0, )),
                          range(10),
                          NAMES,
                          client=dask_client,
                          npartitions=4,
                          stop_on_error=True)
    finally:
        release.touch()

    assert info.value.completed == NAMES[1:4]
    assert info.value.not_attempted == NAMES[4:]


class InterruptedAsCompleted(dask.distributed.as_completed):
    """Raise KeyboardInterrupt when the first task finishes."""

    def __next__(self):
        super().__next__()
        raise KeyboardInterrupt


def interrupt(*args, **kwargs) -> None:
    """Raise KeyboardInterrupt."""
    raise KeyboardInterrupt


def record_on_release(item: int,
                      *,
                      track: str,
                      release: str,
                      fail: tuple[int, ...] = ()) -> None:
    """Pass item zero to `record` at once, and the other items once the
    release file exists."""
    if item:
        deadline = time.monotonic() + 30
        while not os.path.exists(release):
            if time.monotonic() > deadline:
                raise TimeoutError('The test did not release the tasks')
            time.sleep(0.01)
    record(item, track=track, fail=fail)


@pytest.mark.parametrize('twice', [False, True])
def test_interruption(
        dask_client,  # noqa: F811
        tmp_path,
        monkeypatch,
        twice) -> None:
    """A first interruption reaches the caller once the tasks in flight
    finish; a second one, during that wait, reaches it at once."""
    monkeypatch.setattr(execution.distributed, 'as_completed',
                        InterruptedAsCompleted)
    if twice:
        monkeypatch.setattr(execution.distributed, 'wait', interrupt)
    release = tmp_path.parent / f'{tmp_path.name}.release'
    callback = (functools.partial(
        record_on_release, track=str(tmp_path), release=str(release)) if twice
                else functools.partial(record, track=str(tmp_path), delay=0.5))
    try:
        with pytest.raises(KeyboardInterrupt):
            execution.run(callback,
                          range(10),
                          NAMES,
                          client=dask_client,
                          npartitions=4,
                          stop_on_error=True)

        if twice:
            assert items(tmp_path) == [0]
        else:
            assert items(tmp_path) == [0, 1, 2, 3]
    finally:
        release.touch()


def exit_on_one(item: int) -> None:
    """End the worker process running the item one."""
    if item == 1:
        os._exit(1)


def test_a_lost_worker_is_reported_on_its_partition() -> None:
    """A partition whose worker dies is reported as failed."""
    with dask.config.set({'distributed.scheduler.allowed-failures': 0}):
        with dask.distributed.LocalCluster(n_workers=2,
                                           threads_per_worker=1,
                                           processes=True,
                                           dashboard_address=':0') as cluster:
            with dask.distributed.Client(cluster) as lost_client:
                with pytest.raises(execution.PartitionErrors) as info:
                    execution.run(exit_on_one,
                                  range(3),
                                  NAMES[:3],
                                  client=lost_client,
                                  npartitions=1,
                                  stop_on_error=False)

    assert info.value.completed == ('p0', 'p2')
    assert list(info.value.failed) == ['p1']
