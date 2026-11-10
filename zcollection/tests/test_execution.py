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


@pytest.mark.parametrize('npartitions', [0, -1])
def test_npartitions_below_one_is_refused(client, npartitions) -> None:
    """A number of partitions below one is refused."""
    with pytest.raises(ValueError, match='npartitions'):
        execution.run(print,
                      range(10),
                      NAMES,
                      client=client,
                      npartitions=npartitions,
                      stop_on_error=True)


@pytest.mark.parametrize(
    ('stop_on_error', 'written', 'failed', 'not_attempted'), [
        (True, NAMES[:3], ['p3'], NAMES[4:]),
        (False, NAMES[:3] + NAMES[4:6] + NAMES[7:], ['p3', 'p6'], ()),
    ])
def test_report(client, tmp_path, stop_on_error, written, failed,
                not_attempted) -> None:
    """The error gives the partitions written, failed and not attempted,
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
    assert errors.written == written
    assert list(errors.failed) == failed
    assert 'ValueError: item 3 failed' in errors.failed['p3']
    assert errors.not_attempted == not_attempted
    assert items(tmp_path) == [int(name[1:]) for name in written]
    assert errors.failed['p3'] in str(errors)
    copy = pickle.loads(pickle.dumps(errors))
    assert (copy.written, copy.failed,
            copy.not_attempted) == (errors.written, errors.failed,
                                    errors.not_attempted)


def check_arguments(item: int, *, data: list[int], priority: int) -> None:
    """Check the keyword arguments a task receives."""
    assert data == [1, 2, 3]
    assert priority == 5


def test_keyword_arguments_reach_the_function(client) -> None:
    """The keyword arguments reach the function, scattered data included,
    whatever their names."""
    data = [1, 2, 3]
    assert execution.run(check_arguments,
                         range(2),
                         NAMES[:2],
                         client=client,
                         npartitions=1,
                         stop_on_error=True,
                         kwargs={
                             'data': client.scatter(data) if client else data,
                             'priority': 5
                         }) == NAMES[:2]


def test_returns_once_the_tasks_in_flight_finish(
        dask_client,  # noqa: F811
        tmp_path) -> None:
    """After a failure, the tasks already running finish before the error
    reaches the caller."""
    with pytest.raises(execution.PartitionErrors) as info:
        execution.run(functools.partial(record,
                                        track=str(tmp_path),
                                        fail=(0, ),
                                        delay=0.5),
                      range(10),
                      NAMES,
                      client=dask_client,
                      npartitions=4,
                      stop_on_error=True)

    assert items(tmp_path) == [1, 2, 3]
    assert info.value.not_attempted == NAMES[4:]


class InterruptedAsCompleted(dask.distributed.as_completed):
    """Raise KeyboardInterrupt when the first task finishes."""

    def __next__(self):
        super().__next__()
        raise KeyboardInterrupt


def interrupt(*args, **kwargs) -> None:
    """Raise KeyboardInterrupt."""
    raise KeyboardInterrupt


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
    with pytest.raises(KeyboardInterrupt):
        execution.run(functools.partial(record, track=str(tmp_path),
                                        delay=0.5),
                      range(10),
                      NAMES,
                      client=dask_client,
                      npartitions=4,
                      stop_on_error=True)

    if twice:
        assert len(items(tmp_path)) < 4
    else:
        assert items(tmp_path) == [0, 1, 2, 3]


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

    assert info.value.written == ('p0', 'p2')
    assert list(info.value.failed) == ['p1']
