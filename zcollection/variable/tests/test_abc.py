# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing interface for the variable module
=========================================
"""
import dask
import numpy
import pytest
import zarr

from ..abc import FILL_VALUE, not_equal
from ..array import Array
from ..delayed_array import DelayedArray


def test_variable_not_equal() -> None:
    """Test if two values are different."""
    assert not_equal(1, 2) is True
    assert not_equal(1, 1) is False
    assert not_equal(1, '1') is True
    assert not_equal(1, numpy.nan) is True
    assert not_equal(numpy.nan, numpy.nan) is False
    assert not_equal(numpy.nan, 1) is True
    assert not_equal(numpy.datetime64('NaT'), numpy.datetime64('NaT')) is False
    assert not_equal(numpy.datetime64('NaT'), 1) is True


#: Name of the attribute holding the dimensions of a Zarr array.
DIMENSIONS = '_ARRAY_DIMENSIONS'


def make_zarr_array(fill_value, attributes):
    """Create a Zarr array carrying the given attributes."""
    array = zarr.array(numpy.arange(4, dtype='float64'), fill_value=fill_value)
    array.attrs[DIMENSIONS] = ['num_lines']
    array.attrs.update(attributes)
    return array


@pytest.mark.parametrize('cls', [Array, DelayedArray])
def test_from_zarr_reads_the_declared_fill_value(cls) -> None:
    """The value marking absence is read from its own attribute."""
    array = make_zarr_array(None, {FILL_VALUE: -999.0, 'units': 'm'})

    variable = cls.from_zarr(array, 'var', DIMENSIONS)

    assert variable.fill_value == -999.0
    assert {attr.name for attr in variable.attrs} == {'units'}


@pytest.mark.parametrize('cls', [Array, DelayedArray])
def test_from_zarr_reads_no_fill_value_when_none_is_declared(cls) -> None:
    """Without the attribute, and without a fill value, nothing is masked."""
    array = make_zarr_array(None, {'units': 'm'})

    variable = cls.from_zarr(array, 'var', DIMENSIONS)

    assert variable.fill_value is None
    # A closed client may still be the default scheduler at this point.
    with dask.config.set(scheduler='synchronous'):
        assert not numpy.ma.getmaskarray(variable.values).any()


@pytest.mark.parametrize('cls', [Array, DelayedArray])
def test_from_zarr_falls_back_to_the_fill_value_of_the_array(cls) -> None:
    """Collections written before the attribute existed keep their meaning.

    Their value marking absence sits in the fill value of the Zarr array,
    which is where it is read from when the attribute is absent.
    """
    array = make_zarr_array(-999.0, {'units': 'm'})

    variable = cls.from_zarr(array, 'var', DIMENSIONS)

    assert variable.fill_value == -999.0


@pytest.mark.parametrize('cls', [Array, DelayedArray])
def test_from_zarr_prefers_the_attribute_over_the_array(cls) -> None:
    """The attribute wins: the fill value of the array is storage only."""
    array = make_zarr_array(0.0, {FILL_VALUE: -999.0})

    variable = cls.from_zarr(array, 'var', DIMENSIONS)

    assert variable.fill_value == -999.0
