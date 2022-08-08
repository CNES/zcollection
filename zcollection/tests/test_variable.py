# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing variables
=================
"""
import pickle

import dask.array.core
import dask.array.ma
import numpy
import pytest
import zarr

from .. import meta, variable
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint enable=unused-import


def test_maybe_truncate():
    """Test the truncation of a string to a given length."""
    data = list(range(1000))
    # pylint: disable=protected-access
    assert variable._maybe_truncate(data, 10) == '[0, 1, ...'
    assert variable._maybe_truncate(data, len(str(data))) == str(data)
    # pylint: enable=protected-access


def test_variable_masked_array(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test masked array."""
    var = create_test_variable()
    var2 = var.rename('var2')
    assert var2.array is var.array
    assert var2.name == 'var2'
    assert var2.dimensions == var.dimensions
    assert var2.attrs == var.attrs
    assert var2.compressor == var.compressor
    assert var2.filters == var.filters
    assert var2.fill_value == var.fill_value
    assert var2.dtype == var.dtype
    assert var2.shape == var.shape
    assert var2.size == var.size
    assert var2.ndim == var.ndim


def test_variable_not_equal():
    """Test if two values are different."""
    assert variable._not_equal(1, 2) is True
    assert variable._not_equal(1, 1) is False
    assert variable._not_equal(1, '1') is True
    assert variable._not_equal(1, numpy.nan) is True
    assert variable._not_equal(numpy.nan, numpy.nan) is False
    assert variable._not_equal(numpy.nan, 1) is True
    assert variable._not_equal(numpy.datetime64('NaT'),
                               numpy.datetime64('NaT')) is False
    assert variable._not_equal(numpy.datetime64('NaT'), 1) is True


def test_variable_as_asarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test converting array like to a dask array."""
    arr = numpy.arange(10)
    da, fill_value = variable._asarray(arr)
    assert isinstance(da, dask.array.core.Array)
    assert fill_value is None

    arr = numpy.ma.masked_equal(arr, 5)
    da, fill_value = variable._asarray(arr)
    assert isinstance(da, dask.array.core.Array)
    assert fill_value == 5

    da, fill_value = variable._asarray(dask.array.ma.masked_equal(arr, 5))
    assert isinstance(da, dask.array.core.Array)
    assert fill_value == 5

    with pytest.raises(ValueError):
        variable._asarray(numpy.ma.masked_equal(arr, 5), fill_value=6)

    with pytest.raises(ValueError):
        variable._asarray(numpy.ma.masked_equal(
            numpy.arange(numpy.datetime64(0, 'Y'),
                         numpy.datetime64(10, 'Y'),
                         dtype='M8[Y]'), numpy.datetime64(5, 'Y')),
                          fill_value=numpy.datetime64('NaT'))

    da = variable._asarray(numpy.ma.masked_equal(
        numpy.arange(numpy.datetime64(0, 'Y'),
                     numpy.datetime64(10, 'Y'),
                     dtype='M8[Y]'), numpy.datetime64('NaT')),
                           fill_value=numpy.datetime64('NaT'))


def create_test_variable(name='var1', fill_value=0):
    """Create a test variable."""
    return variable.Variable(name=name,
                             data=numpy.arange(10,
                                               dtype='int64').reshape(5, 2),
                             dimensions=('x', 'y'),
                             attrs=(variable.Attribute(name='attr',
                                                       value=1), ),
                             compressor=zarr.Blosc(cname='zstd', clevel=1),
                             fill_value=fill_value,
                             filters=(zarr.Delta('int64', 'int32'),
                                      zarr.Delta('int32', 'int32')))


def test_variable(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test variable creation."""
    var = create_test_variable()
    assert var.name == 'var1'
    assert var.dtype == numpy.dtype('int64')
    assert var.shape == (5, 2)
    assert var.dimensions == ('x', 'y')
    assert var.attrs == (variable.Attribute(name='attr', value=1), )
    assert var.compressor.cname == 'zstd'  # type: ignore
    assert var.compressor.clevel == 1  # type: ignore
    assert var.fill_value == 0
    assert var.size == 10
    assert var.nbytes == 80
    assert var.filters == (
        zarr.Delta('int64', 'int32'),
        zarr.Delta('int32', 'int32'),
    )
    assert numpy.all(var.values == numpy.arange(10).reshape(5, 2))
    assert numpy.all(var.values == var.values)
    assert tuple(var.dimension_index()) == (('x', 0), ('y', 1))
    assert isinstance(var.metadata(), meta.Variable)
    assert isinstance(str(var), str)
    assert isinstance(repr(var), str)

    other = pickle.loads(pickle.dumps(var))
    assert other.name == 'var1'
    assert other.dtype == numpy.dtype('int64')
    assert other.shape == (5, 2)
    assert other.dimensions == ('x', 'y')
    assert other.attrs == (variable.Attribute(name='attr', value=1), )
    assert other.compressor.cname == 'zstd'  # type: ignore
    assert other.compressor.clevel == 1  # type: ignore
    assert other.fill_value == 0
    assert other.size == 10
    assert other.nbytes == 80
    assert other.filters == (
        zarr.Delta('int64', 'int32'),
        zarr.Delta('int32', 'int32'),
    )
    assert numpy.all(var.values == other.values)

    def foo(a, b):
        return a + b

    assert numpy.all(
        foo(var, var.values) == numpy.arange(10).reshape(5, 2) +
        numpy.arange(10).reshape(5, 2))

    assert numpy.all(
        foo(var, var.data).compute() == numpy.arange(10).reshape(5, 2) +
        numpy.arange(10).reshape(5, 2))

    var.data = numpy.ones((10, 4), dtype='int64')
    assert var.data.shape == (10, 4)
    assert isinstance(var.data, dask.array.core.Array)
    assert numpy.all(var.values == 1)

    with pytest.raises(ValueError):
        var.data = numpy.ones((10, 4, 2), dtype='int64')


def test_variable_duplicate(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test of the duplication of variables."""
    var = create_test_variable()
    other = var.duplicate(var.array * 2)
    assert other.name == 'var1'
    assert other.dtype == numpy.dtype('int64')
    assert other.shape == (5, 2)
    assert other.dimensions == ('x', 'y')
    assert other.attrs == (variable.Attribute(name='attr', value=1), )
    assert other.compressor.cname == 'zstd'  # type: ignore
    assert other.compressor.clevel == 1  # type: ignore
    assert other.fill_value == 0
    assert other.filters == (
        zarr.Delta('int64', 'int32'),
        zarr.Delta('int32', 'int32'),
    )
    assert numpy.all(var.values == other.values / 2)  # type: ignore
    assert var.have_same_properties(other)

    with pytest.raises(ValueError):
        var.duplicate(numpy.ones((10, 4, 2), dtype='int64'))


def test_variable_concat(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test concatenation of variables."""
    var_a = create_test_variable()
    var_b = create_test_variable()
    var_c = create_test_variable()

    vard = var_a.concat((var_b, var_c), 'x')
    assert numpy.all(vard.values == numpy.concatenate(
        (var_a.values, var_b.values, var_c.values), axis=0))

    vard = var_a.concat(var_b, 'x')
    assert numpy.all(
        vard.values == numpy.concatenate((var_a.values, var_b.values), axis=0))

    with pytest.raises(ValueError):
        var_a.concat([], 'y')


def test_variable_datetime64_to_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test conversion to xarray."""
    dates = numpy.arange(
        numpy.datetime64('2000-01-01', 'ms'),
        numpy.datetime64('2000-02-01', 'ms'),
        numpy.timedelta64('1', 'h'),
    )
    var = variable.Variable(
        name='time',
        data=dates,
        dimensions=('num_lines', ),
        attrs=(variable.Attribute(name='attr', value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta('int64', 'int64'), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ('num_lines', )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype == 'datetime64[ns]'


def test_variable_timedelta64_to_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test conversion to xarray."""
    delta = numpy.diff(
        numpy.arange(
            numpy.datetime64('2000-01-01', 'ms'),
            numpy.datetime64('2000-02-01', 'ms'),
            numpy.timedelta64('1', 'h'),
        ))

    var = variable.Variable(
        name='timedelta',
        data=delta,
        dimensions=('num_lines', ),
        attrs=(variable.Attribute(name='attr', value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta('int64', 'int64'), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ('num_lines', )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype.kind == 'm'


def test_variable_dimension_less(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Concatenate two dimensionless variables."""
    data = numpy.array([0, 1], dtype=numpy.int32)
    args = ('nv', data, ('nv', ), (variable.Attribute('comment', 'vertex'),
                                   variable.Attribute('units', '1')))
    n_vertex = variable.Variable(*args)
    assert n_vertex.fill_value is None
    metadata = n_vertex.metadata()
    assert metadata.fill_value is None
    assert meta.Variable.from_config(metadata.get_config()) == metadata

    other = variable.Variable(*args)

    concatenated = n_vertex.concat((other, ), 'time')
    assert numpy.all(concatenated.values == n_vertex.values)
    assert concatenated.metadata() == n_vertex.metadata()


def test_variable_getitem(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    var = create_test_variable()
    values = var.values
    result = var[0].compute()
    assert numpy.all(result == values[0])
    result = var[0:2].compute()
    assert numpy.all(result == values[0:2])
    result = var[0:2, 0].compute()
    assert numpy.all(result == values[0:2, 0])
    result = var[0:2, 0:2].compute()
    assert numpy.all(result == values[0:2, 0:2])


def test_variable_fill(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test filling of variables."""
    var = create_test_variable()
    assert not var.values.all() is numpy.ma.masked
    var.fill()
    assert var.values.all() is numpy.ma.masked
