# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Tests of the expression evaluation
==================================
"""
from __future__ import annotations

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


def make_dataset(num_samples: int | None = None) -> dataset.Dataset:
    """Creation of a data set for testing purposes."""
    dates = numpy.arange(numpy.datetime64('2000-01-01', 'ns'),
                         numpy.datetime64('2009-12-31', 'ns'),
                         numpy.timedelta64(1, 'h')).astype('datetime64[ns]')
    if num_samples is not None:
        dates = dates[:num_samples + 1]
    observation = numpy.random.rand(dates.size)  # type: ignore
    return dataset.Dataset.from_xarray(
        xarray.Dataset({
            'dates':
            xarray.DataArray(dates, dims=('num_lines', )),
            'observation':
            xarray.DataArray(observation, dims=('num_lines', ))
        }))


def test_expression() -> None:
    """Test of the creation of expressions."""
    expr = Expression('a == b')
    assert expr({'a': 1, 'b': 1})
    assert not expr({'a': 1, 'b': 2})

    with pytest.raises(SyntaxError):
        Expression('a==')

    with pytest.raises(NameError):
        assert expr({'a': 1, 'c': 1})


def test_date_expression(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test of expressions handling dates.."""
    zds = make_dataset(5 * 24)
    partitioning = Date(('dates', ), 'D')

    for partition, _ in partitioning.split_dataset(zds, 'num_lines'):
        variables = dict(partitioning.parse('/'.join(partition)))
        expr = Expression('year==2000')
        assert expr(variables)
        expr = Expression('year==2000 and month==1')
        assert expr(variables)
        expr = Expression('year==2000 and month==1 and day in range(1, 12)')
        assert expr(variables)


def test_bench_expression(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Benchmark of expressions."""
    partitioning = Date(('dates', ), 'D')
    zds = make_dataset()
    expr = Expression('year==2000 and month==1 and day in range(1, 12)')
    times = []
    number = 100
    for partition, _ in partitioning.split_dataset(zds, 'num_lines'):
        variables = dict(partitioning.parse('/'.join(partition)))
        times.append(
            timeit.timeit('expr(variables)',
                          globals={
                              'expr': expr,
                              'variables': variables
                          },
                          number=number))

    assert sum(times) / (len(times) * number) < 1e-5
