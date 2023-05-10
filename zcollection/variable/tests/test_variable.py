# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing variables
=================
"""
from typing import Any
import pickle

import dask.array.core
import dask.array.ma
import numpy
import pytest
import xarray
import zarr

from ... import meta
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ..abc import Variable
from ..array import Array
from ..delayed_array import DelayedArray
from .data import array, delayed_array

# pylint enable=unused-import

#: Create an Array for testing
ARRAY = array

#: Create a DelayedArray for testing
DELAYED_ARRAY = delayed_array


@pytest.mark.parametrize('factory', [ARRAY, DELAYED_ARRAY])
def test_masked_array(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test masked array."""
    var1: Variable = factory()
    var2: Variable = var1.rename('var2')
    assert var2.array is var1.array
    assert var2.name == 'var2'
    assert var2.dimensions == var1.dimensions
    assert var2.attrs == var1.attrs
    assert var2.compressor == var1.compressor
    assert var2.filters == var1.filters
    assert var2.fill_value == var1.fill_value
    assert var2.dtype == var1.dtype
    assert var2.shape == var1.shape
    assert var2.size == var1.size
    assert var2.ndim == var1.ndim


@pytest.mark.parametrize('factory', [ARRAY, DELAYED_ARRAY])
def test_constructor(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test variable creation."""
    var1: Variable = factory()
    assert var1.name == 'var1'
    assert var1.dtype == numpy.dtype('int64')
    assert var1.shape == (5, 2)
    assert var1.dimensions == ('x', 'y')
    assert var1.attrs == (meta.Attribute(name='attr', value=1), )
    assert var1.compressor.cname == 'zstd'  # type: ignore
    assert var1.compressor.clevel == 1  # type: ignore
    assert var1.fill_value == 0
    assert var1.size == 10
    assert var1.nbytes == 80
    assert var1.filters == (
        zarr.Delta('int64', 'int32'),
        zarr.Delta('int32', 'int32'),
    )
    assert numpy.all(var1.values == numpy.arange(10).reshape(5, 2))
    assert numpy.all(var1.values == var1.values)
    assert tuple(var1.dimension_index()) == (('x', 0), ('y', 1))
    assert isinstance(var1.metadata(), meta.Variable)
    assert isinstance(str(var1), str)
    assert isinstance(repr(var1), str)

    var2: Variable = pickle.loads(pickle.dumps(var1))
    assert var2.name == 'var1'
    assert var2.dtype == numpy.dtype('int64')
    assert var2.shape == (5, 2)
    assert var2.dimensions == ('x', 'y')
    assert var2.attrs == (meta.Attribute(name='attr', value=1), )
    assert var2.compressor.cname == 'zstd'  # type: ignore
    assert var2.compressor.clevel == 1  # type: ignore
    assert var2.fill_value == 0
    assert var2.size == 10
    assert var2.nbytes == 80
    assert var2.filters == (
        zarr.Delta('int64', 'int32'),
        zarr.Delta('int32', 'int32'),
    )
    assert numpy.all(var1.values == var2.values)

    def add_arrays(arr1, arr2) -> Any:
        return arr1 + arr2

    assert numpy.all(
        add_arrays(var1, var1.values) == numpy.arange(10).reshape(5, 2) +
        numpy.arange(10).reshape(5, 2))

    assert numpy.all(
        add_arrays(var1, var1.data).compute() ==
        numpy.arange(10).reshape(5, 2) + numpy.arange(10).reshape(5, 2))

    var1.values = numpy.ones((10, 4), dtype='int64')
    assert var1.data.shape == (10, 4)
    assert isinstance(var1.data, dask.array.core.Array)
    assert numpy.all(var1.values == 1)

    with pytest.raises(ValueError):
        var1.values = numpy.ones((10, 4, 2), dtype='int64')


@pytest.mark.parametrize('factory', [ARRAY, DELAYED_ARRAY])
def test_duplicate(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test of the duplication of variables."""
    var1: Variable = factory()
    var2: Variable = var1.duplicate(var1.array * 2)
    assert var2.name == 'var1'
    assert var2.dtype == numpy.dtype('int64')
    assert var2.shape == (5, 2)
    assert var2.dimensions == ('x', 'y')
    assert var2.attrs == (meta.Attribute(name='attr', value=1), )
    assert var2.compressor.cname == 'zstd'  # type: ignore
    assert var2.compressor.clevel == 1  # type: ignore
    assert var2.fill_value == 0
    assert var2.filters == (
        zarr.Delta('int64', 'int32'),
        zarr.Delta('int32', 'int32'),
    )
    assert numpy.all(var1.values == var2.values / 2)  # type: ignore
    assert var1.metadata() == var2.metadata()

    with pytest.raises(ValueError):
        var1.duplicate(numpy.ones((10, 4, 2), dtype='int64'))


@pytest.mark.parametrize('factory', [ARRAY, DELAYED_ARRAY])
def test_concatenate(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test concatenation of variables."""
    var1: Variable = factory()
    var2: Variable = factory()
    var3: Variable = factory()

    var4: Variable = var1.concat((var2, var3), 'x')
    assert numpy.all(var4.values == numpy.concatenate(
        (var1.values, var2.values, var3.values), axis=0))

    var4 = var1.concat(var2, 'x')
    assert numpy.all(
        var4.values == numpy.concatenate((var1.values, var2.values), axis=0))

    with pytest.raises(ValueError):
        var1.concat([], 'y')


@pytest.mark.parametrize('factory', [ARRAY, DELAYED_ARRAY])
def test_getitem(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test getting of variables."""
    var: Variable = factory()
    values: numpy.ndarray = var.values
    result: numpy.ndarray = var[0]
    assert numpy.all(result == values[0])
    result = var[0:2]
    assert numpy.all(result == values[0:2])
    result = var[0:2, 0]
    assert numpy.all(result == values[0:2, 0])
    result = var[0:2, 0:2]
    assert numpy.all(result == values[0:2, 0:2])


@pytest.mark.parametrize('factory', [ARRAY, DELAYED_ARRAY])
def test_fill(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test filling of variables."""
    var: Variable = factory()
    assert not var.values.all() is numpy.ma.masked
    var.fill()
    assert var.values.all() is numpy.ma.masked


@pytest.mark.parametrize('factory', [ARRAY, DELAYED_ARRAY])
def test_rechunk(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test rechunking of variables."""
    var: Variable = factory()
    values: numpy.ndarray = var.values
    var = var.rechunk()
    assert numpy.all(var.values == values)


@pytest.mark.parametrize('factory', [Array, DelayedArray])
def test_dimension_less(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Concatenate two dimensionless variables."""
    data: numpy.ndarray = numpy.array([0, 1], dtype=numpy.int32)
    var1: Variable = factory('nv',
                             data, ('nv', ),
                             attrs=(meta.Attribute('comment', 'vertex'),
                                    meta.Attribute('units', '1')))
    assert var1.fill_value is None
    metadata: meta.Variable = var1.metadata()
    assert metadata.fill_value is None
    assert meta.Variable.from_config(metadata.get_config()) == metadata

    var2: Variable = factory('nv',
                             data, ('nv', ),
                             attrs=(meta.Attribute('comment', 'vertex'),
                                    meta.Attribute('units', '1')))

    concatenated: Variable = var1.concat((var2, ), 'time')
    assert numpy.all(concatenated.values == var1.values)
    assert concatenated.metadata() == var1.metadata()


@pytest.mark.parametrize('factory', [Array, DelayedArray])
def test_timedelta64_to_xarray(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test conversion to xarray."""
    delta: numpy.ndarray = numpy.diff(
        numpy.arange(
            numpy.datetime64('2000-01-01', 'ns'),
            numpy.datetime64('2000-02-01', 'ns'),
            numpy.timedelta64('1', 'h'),
        ))

    var: Variable = factory(
        name='timedelta',
        data=delta,
        dimensions=('num_lines', ),
        attrs=(meta.Attribute(name='attr', value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta('int64', 'int64'), ),
    )
    xr_var: xarray.Variable = var.to_xarray()
    assert xr_var.dims == ('num_lines', )
    assert xr_var.attrs == {'attr': 1}
    assert xr_var.dtype.kind == 'm'


@pytest.mark.parametrize('factory', [Array, DelayedArray])
def test_datetime64_to_xarray(
        factory,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test conversion to xarray."""
    dates: numpy.ndarray = numpy.arange(
        numpy.datetime64('2000-01-01', 'ns'),
        numpy.datetime64('2000-02-01', 'ns'),
        numpy.timedelta64('1', 'h'),
    )
    var: Variable = factory(
        name='time',
        data=dates,
        dimensions=('num_lines', ),
        attrs=(meta.Attribute(name='attr', value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta('int64', 'int64'), ),
    )
    xr_var: xarray.Variable = var.to_xarray()
    assert xr_var.dims == ('num_lines', )
    assert xr_var.attrs == {'attr': 1}
    assert xr_var.dtype == 'datetime64[ns]'
