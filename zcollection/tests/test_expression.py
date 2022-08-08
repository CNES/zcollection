# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Tests of the expression evaluation
==================================
"""
from typing import Optional
import timeit

import numpy
import pytest
import xarray

from .. import dataset
from ..expression import Expression
from ..partitioning import Date
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint enable=unused-import


def make_dataset(num_samples: Optional[int] = None) -> dataset.Dataset:
    """Creation of a data set for testing purposes."""
    dates = numpy.arange(numpy.datetime64('2000-01-01'),
                         numpy.datetime64('2009-12-31'),
                         numpy.timedelta64(1, 'h')).astype('datetime64[us]')
    if num_samples is not None:
        dates = dates[:num_samples + 1]
    observation = numpy.random.rand(dates.size)  # type: ignore
    return dataset.Dataset.from_xarray(
        xarray.Dataset(
            dict(dates=xarray.DataArray(dates, dims=('num_lines', )),
                 observation=xarray.DataArray(observation,
                                              dims=('num_lines', )))))


def test_expression():
    """Test of the creation of expressions."""
    expr = Expression('a == b')
    assert expr(dict(a=1, b=1))
    assert not expr(dict(a=1, b=2))

    with pytest.raises(SyntaxError):
        Expression('a==')

    with pytest.raises(NameError):
        assert expr(dict(a=1, c=1))


def test_date_expression(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test of expressions handling dates.."""
    ds = make_dataset(5 * 24)
    partitioning = Date(('dates', ), 'D')

    for partition, _ in partitioning.split_dataset(ds, 'num_lines'):
        variables = dict(partitioning.parse('/'.join(partition)))
        expr = Expression('year==2000')
        assert expr(variables)
        expr = Expression('year==2000 and month==1')
        assert expr(variables)
        expr = Expression('year==2000 and month==1 and day in range(1, 12)')
        assert expr(variables)


def test_bench_expression(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Benchmark of expressions."""
    partitioning = Date(('dates', ), 'D')
    ds = make_dataset()
    expr = Expression('year==2000 and month==1 and day in range(1, 12)')
    times = []
    number = 100
    for partition, _ in partitioning.split_dataset(ds, 'num_lines'):
        variables = dict(partitioning.parse('/'.join(partition)))
        times.append(
            timeit.timeit('expr(variables)',
                          globals=dict(expr=expr, variables=variables),
                          number=number))

    assert sum(times) / (len(times) * number) < 1e-5
