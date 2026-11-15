# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of the serialization of data types and fill values
=======================================================
"""
from __future__ import annotations

from typing import Any
import json

import numpy
import pytest

from .. import serialization

# Metadata forms from Zarr 2.18.7 Metadata2:
# https://github.com/zarr-developers/zarr-python/blob/v2.18.7/zarr/meta.py

#: Fill values, their data types and their stored JSON forms.
FILL_VALUES: list[tuple[Any, str, Any]] = [
    (None, 'float64', None),
    (-999.0, 'float64', -999.0),
    (0.0, 'float32', 0.0),
    (numpy.nan, 'float64', 'NaN'),
    (numpy.inf, 'float64', 'Infinity'),
    (-numpy.inf, 'float32', '-Infinity'),
    (-99, 'int32', -99),
    (0, 'int8', 0),
    (2147483647, 'uint32', 2147483647),
    (True, 'bool', True),
    (False, 'bool', False),
    (complex(1.5, -2.5), 'complex128', [1.5, -2.5]),
    (b'\x00\x01', 'S2', 'AAE='),
    ('text', 'U8', 'text'),
    (numpy.datetime64('2000-01-01',
                      'ns'), 'datetime64[ns]', 946684800000000000),
    (numpy.timedelta64(6, 'h'), 'timedelta64[h]', 6),
]

#: Declared data types and their metadata forms for little-endian storage.
DTYPES: list[tuple[str, str]] = [
    ('float64', '<f8'),
    ('float32', '<f4'),
    ('int8', '|i1'),
    ('uint64', '<u8'),
    ('bool', '|b1'),
    ('complex128', '<c16'),
    ('S4', '|S4'),
    ('U8', '<U8'),
    ('datetime64[ns]', '<M8[ns]'),
    ('timedelta64[h]', '<m8[h]'),
]


@pytest.mark.parametrize(('value', 'dtype', 'expected'), FILL_VALUES)
def test_fill_value_serialization(value, dtype, expected) -> None:
    """Fill values encode to and decode from the stored JSON form."""
    dtype = numpy.dtype(dtype)
    encoded = serialization.encode_fill_value(value, dtype)
    assert json.loads(json.dumps(encoded, allow_nan=False)) == expected
    decoded = serialization.decode_fill_value(expected, dtype)

    if value is None:
        assert decoded is None
    elif isinstance(value, float) and numpy.isnan(value):
        assert numpy.isnan(decoded)
    else:
        assert decoded == value


@pytest.mark.parametrize(('dtype', 'expected'), DTYPES)
def test_dtype_serialization(dtype, expected) -> None:
    """Data types encode to and decode from the stored metadata form."""
    dtype = numpy.dtype(dtype).newbyteorder('<')
    assert serialization.encode_dtype(dtype) == expected
    assert serialization.decode_dtype(expected) == dtype


def test_structured_dtype_serialization() -> None:
    """Structured fields encode to and decode from the stored JSON form."""
    dtype = numpy.dtype([('a', '<i4'), ('b', '<f8')])
    expected = [['a', '<i4'], ['b', '<f8']]
    encoded = serialization.encode_dtype(dtype)
    assert json.loads(json.dumps(encoded)) == expected
    assert serialization.decode_dtype(expected) == dtype


@pytest.mark.parametrize(
    'function',
    [serialization.encode_fill_value, serialization.decode_fill_value])
def test_object_data_type_is_refused(function) -> None:
    """Object data types are not supported by the serializer."""
    with pytest.raises(TypeError, match='Unsupported data type'):
        function(0, numpy.dtype('O'))
