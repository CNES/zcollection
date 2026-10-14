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


#: Name of the attribute a Zarr 2 array holds its dimensions in.
DIMENSIONS = '_ARRAY_DIMENSIONS'

#: The variable implementations the rules hold for.
IMPLEMENTATIONS = [Array, DelayedArray]

#: The versions of Zarr a collection is read from.
FORMATS = [2, 3]


def make_zarr_array(fill_value, attributes, zarr_format=3):
    """Create a Zarr array carrying the given attributes."""
    named = {'dimension_names': ('num_lines', )} if zarr_format == 3 else {}
    array = zarr.create_array(store={},
                              shape=(4, ),
                              dtype='float64',
                              fill_value=fill_value,
                              zarr_format=zarr_format,
                              attributes=dict(attributes),
                              **named)
    array[...] = numpy.arange(4, dtype='float64')
    if zarr_format == 2:
        array.attrs[DIMENSIONS] = ['num_lines']
    return array


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
@pytest.mark.parametrize('zarr_format', FORMATS)
def test_from_zarr_reads_the_dimensions(cls, zarr_format) -> None:
    """A Zarr 2 array names its dimensions in an attribute, a Zarr 3 array
    in a field of its own."""
    array = make_zarr_array(None, {'units': 'm'}, zarr_format)

    variable = cls.from_zarr(array, 'var')

    assert variable.dimensions == ('num_lines', )
    assert {attr.name for attr in variable.attrs} == {'units'}


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
@pytest.mark.parametrize('zarr_format', FORMATS)
def test_from_zarr_reads_the_declared_fill_value(cls, zarr_format) -> None:
    """The value marking absence is read from its own attribute."""
    array = make_zarr_array(None, {
        FILL_VALUE: -999.0,
        'units': 'm'
    }, zarr_format)

    variable = cls.from_zarr(array, 'var')

    assert variable.fill_value == -999.0
    assert {attr.name for attr in variable.attrs} == {'units'}


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
@pytest.mark.parametrize('zarr_format', FORMATS)
def test_from_zarr_reads_no_fill_value_when_none_is_declared(
        cls, zarr_format) -> None:
    """Without the attribute, and without a fill value, nothing is masked."""
    array = make_zarr_array(None, {'units': 'm'}, zarr_format)

    variable = cls.from_zarr(array, 'var')

    assert variable.fill_value is None
    # A closed client may still be the default scheduler at this point.
    with dask.config.set(scheduler='synchronous'):
        assert not numpy.ma.getmaskarray(variable.values).any()


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_from_zarr_falls_back_to_the_fill_value_of_a_zarr_2_array(cls) -> None:
    """A Zarr 2 array carrying no attribute holds the value marking absence
    in its fill value."""
    array = make_zarr_array(-999.0, {'units': 'm'}, zarr_format=2)

    variable = cls.from_zarr(array, 'var')

    assert variable.fill_value == -999.0


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_from_zarr_ignores_the_fill_value_of_a_zarr_3_array(cls) -> None:
    """The fill value of a Zarr 3 array marks nothing."""
    array = make_zarr_array(-999.0, {'units': 'm'}, zarr_format=3)

    variable = cls.from_zarr(array, 'var')

    assert variable.fill_value is None


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
@pytest.mark.parametrize('zarr_format', FORMATS)
def test_from_zarr_prefers_the_attribute_over_the_array(cls,
                                                        zarr_format) -> None:
    """The attribute is read in preference to the fill value of the
    array."""
    array = make_zarr_array(0.0, {FILL_VALUE: -999.0}, zarr_format)

    variable = cls.from_zarr(array, 'var')

    assert variable.fill_value == -999.0
