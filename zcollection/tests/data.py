# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Make test datasets
==================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import functools
import itertools

import numcodecs
import numpy

from .. import collection, dataset, execution, partitioning, sync, view

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ..type_hints import NDArray

#: First date of the test dataset.
START_DATE = numpy.datetime64('2000-01-01', 'ns')
#: Last date of the test dataset.
END_DATE = numpy.datetime64('2000-06-30', 'ns')
#: Delta between two dates.
DELTA = numpy.timedelta64(72, 'h')
#: Fill value.
FILL_VALUE = 2147483647


def make_dataset(dates: numpy.ndarray,
                 measures: numpy.ndarray,
                 fill_value: float | None = None,
                 filters: tuple | None = None,
                 delayed: bool = True) -> dataset.Dataset:
    """Create a dataset."""
    array_class = (dataset.DelayedArray if delayed else dataset.Array)
    return dataset.Dataset(
        attrs=(dataset.Attribute(name='attr', value=1), ),
        variables=(
            array_class(
                name='time',
                data=dates,
                dimensions=('num_lines', ),
                attrs=(dataset.Attribute(name='attr', value=1), ),
                compressor=numcodecs.Blosc(),
            ),
            array_class(name='var1',
                        data=measures,
                        dimensions=('num_lines', 'num_pixels'),
                        attrs=(dataset.Attribute(name='attr', value=1), ),
                        fill_value=fill_value,
                        filters=filters),
            array_class(name='var2',
                        data=measures,
                        dimensions=('num_lines', 'num_pixels'),
                        attrs=(dataset.Attribute(name='attr', value=1), ),
                        fill_value=fill_value,
                        filters=filters),
        ),
    )


def create_test_dataset(delayed: bool = True) -> Iterator[dataset.Dataset]:
    """Create a temporal dataset."""

    dates: NDArray = numpy.arange(START_DATE, END_DATE, DELTA)
    indices: NDArray = numpy.arange(0, len(dates))

    for item in numpy.array_split(dates, 12):
        mask: NDArray = (dates >= item[0]) & (dates <= item[-1])
        measures = numpy.vstack((indices[mask], ) * 25).T

        yield make_dataset(item, measures, delayed=delayed)


def create_test_dataset_with_fillvalue(
        delayed: bool = True,
        rows: slice | None = None) -> Iterator[dataset.Dataset]:
    """Create a dataset with a fixed scale offset filter and fill values."""

    dates: NDArray = numpy.arange(START_DATE, END_DATE, DELTA)
    measures: NDArray = numpy.arange(0, len(dates), dtype=numpy.float64)
    measures[measures % 2 == 0] = FILL_VALUE
    if rows is not None:
        dates = dates[rows]
        measures = measures[rows]
    measures = numpy.vstack((measures, ) * 25).T * 1e-4

    yield make_dataset(dates,
                       measures,
                       delayed=delayed,
                       fill_value=FILL_VALUE * 1e-4,
                       filters=(numcodecs.FixedScaleOffset(scale=10000,
                                                           offset=0,
                                                           dtype='<f8',
                                                           astype='i4'), ))


def create_test_collection(
        tested_fs,
        with_fillvalue=False,
        delayed=True,
        distributed=True,
        fillvalue_rows: slice | None = None) -> collection.Collection:
    """Create a collection."""
    zds: dataset.Dataset = next(
        create_test_dataset_with_fillvalue(
            delayed=delayed, rows=fillvalue_rows
        ) if with_fillvalue else create_test_dataset(delayed=delayed))
    zcol = collection.Collection(
        axis='time',
        ds=zds.metadata(),
        partition_handler=partitioning.Date(('time', ), 'D'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs,
    )
    zcol.insert(zds, distributed=distributed)
    return zcol


#: List of filesystems and datasets to test.
FILE_SYSTEM_DATASET = list(
    itertools.product([
        'local_fs',
        's3_fs',
    ], [
        create_test_dataset,
        create_test_dataset_with_fillvalue,
    ]))


def with_process_sync(target, tmp_path):
    """Reopen the target with a synchronizer that reports its lock."""
    synchronizer = sync.ProcessSync(str(tmp_path / 'update.lock'))
    if isinstance(target, view.View):
        return type(target).from_config(target.base_dir,
                                        filesystem=target.fs,
                                        synchronizer=synchronizer)
    return type(target).from_config(target.partition_properties.dir,
                                    mode='w',
                                    filesystem=target.fs,
                                    synchronizer=synchronizer)


def _process_then_fail(task: Callable[..., Any], failed: Any, item: Any,
                       **kwargs: Any) -> Any:
    """Call `task`, then raise OSError if `item` is `failed`."""
    result = task(item, **kwargs)
    if item == failed:
        raise OSError('task failed')
    return result


def fail_after(monkeypatch, task: Callable[..., Any], failed: str) -> None:
    """Make `task` raise OSError after it handles the partition named
    `failed`, also when the Dask workers are separate processes."""
    run = execution.run

    def failing_run(func, items, names, **kwargs):
        if func is task:
            items, names = list(items), list(names)
            func = functools.partial(_process_then_fail, func,
                                     items[names.index(failed)])
        return run(func, items, names, **kwargs)

    monkeypatch.setattr(execution, 'run', failing_run)


def record_lock(monkeypatch, target) -> list[bool]:
    """Return a list that receives, before and after each run of tasks,
    whether `target` holds its lock."""
    run = execution.run
    held: list[bool] = []

    def recording_run(*args, **kwargs):
        held.append(target.is_locked())
        try:
            return run(*args, **kwargs)
        finally:
            held.append(target.is_locked())

    monkeypatch.setattr(execution, 'run', recording_run)
    return held
