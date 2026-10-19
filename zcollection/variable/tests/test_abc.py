# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing interface for the variable module
=========================================
"""
import dask
import numcodecs
import numpy
import pytest
import zarr

from ...meta import Attribute
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

#: Fill values whose encoded form differs from the value itself.
ENCODED_FILL_VALUES = [
    ('float64', 'NaN', numpy.float64('nan')),
    ('float64', 'Infinity', numpy.float64('inf')),
    ('float64', '-Infinity', numpy.float64('-inf')),
    ('datetime64[s]', 0, numpy.datetime64('1970-01-01', 's')),
    ('timedelta64[s]', 3, numpy.timedelta64(3, 's')),
]


def make_zarr_array(fill_value, attributes, zarr_format=3, dtype='float64'):
    """Create a Zarr array carrying the given attributes."""
    named = {'dimension_names': ('num_lines', )} if zarr_format == 3 else {}
    array = zarr.create_array(store={},
                              shape=(4, ),
                              dtype=dtype,
                              fill_value=fill_value,
                              zarr_format=zarr_format,
                              attributes=dict(attributes),
                              **named)
    array[...] = numpy.arange(4).astype(dtype)
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
    """The fill value is read from its own attribute."""
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
    """A Zarr 2 array carrying no attribute takes the fill value of the
    array."""
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


#: The properties a variable carries from one rebuild to the next.
PROPERTIES = ('name', 'dimensions', 'attrs', 'compressor', 'fill_value',
              'filler', 'filters')


def make_variable(cls):
    """Create a variable whose every property holds a distinctive value."""
    return cls(name='var',
               data=numpy.arange(8.0).reshape(2, 4),
               dimensions=('num_lines', 'num_pixels'),
               attrs=(Attribute('units', 'm'), ),
               compressor=numcodecs.Blosc(),
               fill_value=-999.0,
               filler=True,
               filters=(numcodecs.Delta('float64'), ))


def properties(variable, **changed):
    """Return the properties of a variable, with the given ones replaced."""
    return {
        name: changed.get(name, getattr(variable, name))
        for name in PROPERTIES
    }


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_isel_keeps_the_properties(cls) -> None:
    """Selecting along a dimension keeps every property."""
    variable = make_variable(cls)

    other = variable.isel((slice(0, 1), slice(None)))

    assert properties(other) == properties(variable)
    assert other.shape == (1, 4)


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_rename_keeps_the_other_properties(cls) -> None:
    """Renaming a variable keeps every other property."""
    variable = make_variable(cls)

    other = variable.rename('other')

    assert properties(other) == properties(variable, name='other')


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_concat_keeps_the_properties(cls) -> None:
    """Concatenating a variable with itself keeps every property."""
    variable = make_variable(cls)

    other = variable.concat(variable, 'num_lines')

    assert properties(other) == properties(variable)
    assert other.shape == (4, 4)


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_concat_outside_the_dimensions_keeps_the_properties(cls) -> None:
    """A dimension the variable does not span leaves it as it is."""
    variable = make_variable(cls)

    other = variable.concat(variable, 'elsewhere')

    assert properties(other) == properties(variable)
    assert other.shape == variable.shape


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_duplicate_keeps_the_properties(cls) -> None:
    """A variable built from the data of another keeps every property."""
    variable = make_variable(cls)

    other = variable.duplicate(numpy.ones((2, 4)))

    assert properties(other) == properties(variable)


def test_rechunk_keeps_the_properties() -> None:
    """Rechunking a delayed variable keeps every property."""
    variable = make_variable(DelayedArray)

    other = variable.rechunk(chunks=(1, 2))

    assert properties(other) == properties(variable)


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
def test_concat_marks_what_holds_a_filled_variable(cls) -> None:
    """A concatenation is filled when one of its variables is."""
    filled = make_variable(cls)
    measured = filled.duplicate(filler=False)

    assert filled.concat(measured, 'num_lines').filler is True
    assert measured.concat(filled, 'num_lines').filler is True
    assert measured.concat(measured, 'num_lines').filler is False


@pytest.mark.parametrize('cls', IMPLEMENTATIONS)
@pytest.mark.parametrize(('dtype', 'encoded', 'expected'), ENCODED_FILL_VALUES)
def test_from_zarr_decodes_the_declared_fill_value(cls, dtype, encoded,
                                                   expected) -> None:
    """The attribute holds the encoded form of the fill value."""
    array = make_zarr_array(None, {FILL_VALUE: encoded}, dtype=dtype)

    variable = cls.from_zarr(array, 'var')

    assert not_equal(variable.fill_value, expected) is False
