# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test merging time series."""
from typing import TYPE_CHECKING
import copy

import numpy
import pytest

from .. import time_series
from ... import dataset
from ...tests import data
from ...variable import Array

if TYPE_CHECKING:
    from ...type_hints import NDArray


def test_merge_disjoint() -> None:
    """Test merging two disjoint time series."""
    generator = data.create_test_dataset()
    zds0 = next(generator)
    zds1 = next(generator)

    zds = time_series.merge_time_series(zds1, zds0, 'time', 'num_lines')
    assert numpy.all(zds.variables['time'].values == numpy.concatenate((
        zds0.variables['time'].values, zds1.variables['time'].values)))

    zds = time_series.merge_time_series(zds0, zds1, 'time', 'num_lines')
    assert numpy.all(zds.variables['time'].values == numpy.concatenate((
        zds0.variables['time'].values, zds1.variables['time'].values)))

    zds = time_series.merge_time_series(zds0, zds0, 'time', 'num_lines')
    assert numpy.all(
        zds.variables['time'].values == zds0.variables['time'].values)


def test_merge_intersection() -> None:
    """Test merging two overlapping time series."""
    generator = data.create_test_dataset()
    zds0 = next(generator)
    # ds0.variables["time"].values => numpy.array([
    #     "2000-01-01T00:00:00.000000", "2000-01-04T00:00:00.000000",
    #     "2000-01-07T00:00:00.000000", "2000-01-10T00:00:00.000000",
    #     "2000-01-13T00:00:00.000000", "2000-01-16T00:00:00.000000"
    # ])
    zds1 = next(generator)
    # ds1.variables["time"].values => numpy.array([
    #     "2000-01-19T00:00:00.000000", "2000-01-22T00:00:00.000000",
    #     "2000-01-25T00:00:00.000000", "2000-01-28T00:00:00.000000",
    #     "2000-01-31T00:00:00.000000"])

    existing_zds = zds1
    new_zds = copy.deepcopy(zds0)
    new_zds.variables['time'] = zds0.variables['time'].duplicate(
        zds0.variables['time'].values + numpy.timedelta64(9, 'D'))

    zds = time_series.merge_time_series(existing_zds, new_zds, 'time',
                                        'num_lines')
    assert numpy.all(zds.variables['time'].values == numpy.concatenate((
        zds0.variables['time'].values[3:], zds1.variables['time'].values[:])))

    existing_zds = zds0
    new_zds = copy.deepcopy(zds1)
    new_zds.variables['time'] = zds1.variables['time'].duplicate(
        zds1.variables['time'].values - numpy.timedelta64(9, 'D'))
    zds = time_series.merge_time_series(existing_zds, new_zds, 'time',
                                        'num_lines')
    assert numpy.all(zds.variables['time'].values == numpy.concatenate((
        zds0.variables['time'].values[:], zds1.variables['time'].values[:2])))

    existing_zds = zds0
    new_zds = zds0.isel({'num_lines': slice(1, -1)})
    new_zds.variables['var1'] = new_zds.variables['var1'].duplicate(
        new_zds.variables['var1'].values + 100)
    zds = time_series.merge_time_series(existing_zds, new_zds, 'time',
                                        'num_lines')
    assert numpy.all(zds.variables['var1'].values == numpy.concatenate((
        zds0.variables['var1'].values[:1],
        zds0.variables['var1'].values[1:-1] + 100,
        zds0.variables['var1'].values[-1:])))


def test_intersection_with_tolerance() -> None:
    """Test merging overlapping time series with a data gap."""
    axis: NDArray = numpy.arange(numpy.datetime64('2000-01-01', 'ns'),
                                 numpy.datetime64('2000-01-01T23:59:59', 'ns'),
                                 numpy.timedelta64(1, 's'))
    measures = numpy.vstack((numpy.arange(axis.size), ) * 25).T
    zds0 = data.make_dataset(axis, measures, delayed=False)

    dates: NDArray = numpy.arange(
        numpy.datetime64('2000-01-01T10:00:00', 'ns'),
        numpy.datetime64('2000-01-01T14:59:59', 'ns'),
        numpy.timedelta64(1, 's'))

    # Create a gap in the data by removing the data between 11:00 and 13:00
    mask = (dates > numpy.datetime64('2000-01-01T11:00:00', 'ns')) & (
        dates < numpy.datetime64('2000-01-01T13:00:00', 'ns'))
    dates = dates[~mask]
    measures = numpy.vstack((numpy.full(dates.size, -1), ) * 25).T
    zds1 = data.make_dataset(dates, measures, delayed=False)

    # Merge the two datasets with a tolerance of 1 minute to keep the
    # data gap in the existing dataset.
    zds_gap_filled = time_series.merge_time_series(zds0,
                                                   zds1,
                                                   'time',
                                                   'num_lines',
                                                   tolerance=numpy.timedelta64(
                                                       1, 'm'))
    # Merge the two datasets without a tolerance. The data gap is
    # kept and stored in the new dataset.
    zds_with_gap = time_series.merge_time_series(
        zds0,
        zds1,
        'time',
        'num_lines',
    )
    assert zds_with_gap.time.size == zds0.time.size - mask.sum()

    mask = (axis > numpy.datetime64('2000-01-01T11:00:00', 'ns')) & (
        axis < numpy.datetime64('2000-01-01T13:00:00', 'ns'))
    assert numpy.all(zds_gap_filled.variables['time'].values ==
                     zds0.variables['time'].values)
    assert numpy.all((zds_gap_filled.variables['var1'].values[:, 0] < 0
                      ).sum() == zds1.dimensions['num_lines'])

    # Create gaps from 11:00 to 13:00, 15:00 to 17:00 and 19:00 to 21:00.
    mask = (axis > numpy.datetime64('2000-01-01T11:00:00', 'ns')) & (
        axis < numpy.datetime64('2000-01-01T13:00:00', 'ns'))
    mask |= (axis > numpy.datetime64('2000-01-01T15:00:00', 'ns')) & (
        axis < numpy.datetime64('2000-01-01T17:00:00', 'ns'))
    mask |= (axis > numpy.datetime64('2000-01-01T19:00:00', 'ns')) & (
        axis < numpy.datetime64('2000-01-01T21:00:00', 'ns'))

    dates = axis[~mask]

    measures = numpy.vstack((numpy.full(dates.size, -1), ) * 25).T
    zds1 = data.make_dataset(dates, measures, delayed=False)

    # Merge the two datasets with a tolerance of 1 minute to keep the
    # data gaps in the existing dataset.
    zds_gap_filled = time_series.merge_time_series(zds0,
                                                   zds1,
                                                   'time',
                                                   'num_lines',
                                                   tolerance=numpy.timedelta64(
                                                       1, 'm'))
    # Merge the two datasets without a tolerance. The data gaps are
    # kept and stored in the new dataset.
    zds_with_gap = time_series.merge_time_series(
        zds0,
        zds1,
        'time',
        'num_lines',
    )

    assert numpy.all(zds_gap_filled.variables['time'].values ==
                     zds0.variables['time'].values)
    assert zds_with_gap.time.size == zds0.time.size - mask.sum()
    assert numpy.all((zds_gap_filled.variables['var1'].values[:, 0] < 0
                      ).sum() == zds1.dimensions['num_lines'])


def _hourly(first: int, last: int, value: int) -> dataset.Dataset:
    """Return hourly records from ``first`` to ``last`` holding ``value``."""
    time = numpy.arange(first, last + 1).astype('datetime64[h]')
    return dataset.Dataset([
        Array('time', time, ('num_lines', )),
        Array('var', numpy.full(time.size, value), ('num_lines', )),
    ])


@pytest.mark.parametrize(
    ('existing', 'inserted'),
    [
        ((1, 5), (0, 2)),  # The inserted series starts before.
        ((0, 5), (1, 2)),  # The inserted series lies inside.
        ((0, 3), (1, 2)),  # A single existing record follows it.
    ],
    ids=['before-overlapping', 'inside', 'inside-one-after'])
def test_merge_keeps_record_following_insert(existing, inserted) -> None:
    """The existing record one step after the inserted series is kept.

    The step of the series equals the unit of its axis.
    """
    zds = time_series.merge_time_series(_hourly(existing[0], existing[1], 0),
                                        _hourly(inserted[0], inserted[1], 1),
                                        'time', 'num_lines')

    first = min(existing[0], inserted[0])
    numpy.testing.assert_array_equal(
        zds.variables['time'].values,
        numpy.arange(first, existing[1] + 1).astype('datetime64[h]'))
    numpy.testing.assert_array_equal(zds.variables['var'].values, [
        int(inserted[0] <= hour <= inserted[1])
        for hour in range(first, existing[1] + 1)
    ])
