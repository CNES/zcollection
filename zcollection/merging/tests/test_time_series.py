# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test the time series merging."""
import copy

import numpy

from .. import time_series
from ...tests import data
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster

# pylint: enable=unused-import # Need to import for fixtures


def test_merge_disjoint(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test the update of two disjoint time series."""
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


def test_merge_intersection(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test the update of two intersecting time series."""
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
