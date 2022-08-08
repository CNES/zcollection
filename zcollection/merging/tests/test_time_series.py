# Copyright (c) 2022 CNES
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
):
    """Test the update of two disjoint time series."""
    generator = data.create_test_dataset()
    ds0 = next(generator)
    ds1 = next(generator)

    ds = time_series.merge_time_series(ds1, ds0, 'time', 'num_lines')
    assert numpy.all(ds.variables['time'].values == numpy.concatenate((
        ds0.variables['time'].values, ds1.variables['time'].values)))

    ds = time_series.merge_time_series(ds0, ds1, 'time', 'num_lines')
    assert numpy.all(ds.variables['time'].values == numpy.concatenate((
        ds0.variables['time'].values, ds1.variables['time'].values)))

    ds = time_series.merge_time_series(ds0, ds0, 'time', 'num_lines')
    assert numpy.all(
        ds.variables['time'].values == ds0.variables['time'].values)


def test_merge_intersection(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the update of two intersecting time series."""
    generator = data.create_test_dataset()
    ds0 = next(generator)
    # ds0.variables["time"].values => numpy.array([
    #     "2000-01-01T00:00:00.000000", "2000-01-04T00:00:00.000000",
    #     "2000-01-07T00:00:00.000000", "2000-01-10T00:00:00.000000",
    #     "2000-01-13T00:00:00.000000", "2000-01-16T00:00:00.000000"
    # ])
    ds1 = next(generator)
    # ds1.variables["time"].values => numpy.array([
    #     "2000-01-19T00:00:00.000000", "2000-01-22T00:00:00.000000",
    #     "2000-01-25T00:00:00.000000", "2000-01-28T00:00:00.000000",
    #     "2000-01-31T00:00:00.000000"])

    existing_ds = ds1
    new_ds = copy.deepcopy(ds0)
    new_ds.variables['time'] = ds0.variables['time'].duplicate(
        ds0.variables['time'].values + numpy.timedelta64(9, 'D'))

    ds = time_series.merge_time_series(existing_ds, new_ds, 'time',
                                       'num_lines')
    assert numpy.all(ds.variables['time'].values == numpy.concatenate((
        ds0.variables['time'].values[3:], ds1.variables['time'].values[:])))

    existing_ds = ds0
    new_ds = copy.deepcopy(ds1)
    new_ds.variables['time'] = ds1.variables['time'].duplicate(
        ds1.variables['time'].values - numpy.timedelta64(9, 'D'))
    ds = time_series.merge_time_series(existing_ds, new_ds, 'time',
                                       'num_lines')
    assert numpy.all(ds.variables['time'].values == numpy.concatenate((
        ds0.variables['time'].values[:], ds1.variables['time'].values[:2])))

    existing_ds = ds0
    new_ds = ds0.isel(dict(num_lines=slice(1, -1)))
    new_ds.variables['var1'] = new_ds.variables['var1'].duplicate(
        new_ds.variables['var1'].values + 100)
    ds = time_series.merge_time_series(existing_ds, new_ds, 'time',
                                       'num_lines')
    assert numpy.all(ds.variables['var1'].values == numpy.concatenate((
        ds0.variables['var1'].values[:1],
        ds0.variables['var1'].values[1:-1] + 100,
        ds0.variables['var1'].values[-1:])))
