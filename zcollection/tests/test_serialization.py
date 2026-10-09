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

import numpy
import pytest

from .. import serialization

#: Fill values and the data type they belong to. Collections on disk hold
#: values encoded this way, so the encoding is a format, not a detail.
FILL_VALUES: list[tuple[Any, str]] = [
    (None, 'float64'),
    (-999.0, 'float64'),
    (0.0, 'float32'),
    (numpy.nan, 'float64'),
    (numpy.inf, 'float64'),
    (-numpy.inf, 'float32'),
    (-99, 'int32'),
    (0, 'int8'),
    (2147483647, 'uint32'),
    (True, 'bool'),
    (False, 'bool'),
    (complex(1.5, -2.5), 'complex128'),
    (b'\x00\x01', 'S2'),
    ('text', 'U8'),
    (numpy.datetime64('2000-01-01', 'ns'), 'datetime64[ns]'),
    (numpy.timedelta64(6, 'h'), 'timedelta64[h]'),
]

#: Data types a variable is declared with.
DTYPES: list[str] = [
    'float64',
    'float32',
    'int8',
    'uint64',
    'bool',
    'complex128',
    'S4',
    'U8',
    'datetime64[ns]',
    'timedelta64[h]',
]


@pytest.mark.parametrize(('value', 'dtype'), FILL_VALUES)
def test_fill_value_round_trips(value, dtype) -> None:
    """A fill value survives the trip through its encoded form."""
    dtype = numpy.dtype(dtype)
    decoded = serialization.decode_fill_value(
        serialization.encode_fill_value(value, dtype), dtype)

    if value is None:
        assert decoded is None
    elif isinstance(value, float) and numpy.isnan(value):
        assert numpy.isnan(decoded)
    else:
        assert decoded == value


@pytest.mark.parametrize(('value', 'dtype'), FILL_VALUES)
def test_fill_value_encodes_as_json(value, dtype) -> None:
    """The encoded form is what a JSON document can hold."""
    import json

    encoded = serialization.encode_fill_value(value, numpy.dtype(dtype))
    assert json.loads(json.dumps(encoded)) == (list(encoded) if isinstance(
        encoded, tuple) else encoded)


@pytest.mark.parametrize('dtype', DTYPES)
def test_dtype_round_trips(dtype) -> None:
    """A data type survives the trip through its encoded form."""
    expected = numpy.dtype(dtype)
    assert serialization.decode_dtype(
        serialization.encode_dtype(expected)) == expected


def test_structured_dtype_round_trips() -> None:
    """A structured data type keeps its fields."""
    expected = numpy.dtype([('a', 'int32'), ('b', 'float64')])
    assert serialization.decode_dtype(
        serialization.encode_dtype(expected)) == expected


@pytest.mark.parametrize(
    'function',
    [serialization.encode_fill_value, serialization.decode_fill_value])
def test_object_data_type_is_refused(function) -> None:
    """A data type holding Python objects has no encoding here."""
    with pytest.raises(TypeError, match='Unsupported data type'):
        function(0, numpy.dtype('O'))


class TestAgainstZarrV2:
    """Compare with the implementation these functions replace.

    Collections written before this module existed were encoded by
    ``zarr.meta``. These tests state that the encoding did not move. They
    are skipped once that module is gone, which is what Zarr v3 does to it.
    """

    @staticmethod
    def metadata():
        """Return the Zarr v2 metadata encoder, or skip."""
        return pytest.importorskip(
            'zarr.meta', reason='zarr.meta is gone from Zarr v3').Metadata2

    @pytest.mark.parametrize(('value', 'dtype'), FILL_VALUES)
    def test_fill_value_encoding_is_unchanged(self, value, dtype) -> None:
        """A fill value encodes the way it used to."""
        dtype = numpy.dtype(dtype)
        assert serialization.encode_fill_value(
            value, dtype) == self.metadata().encode_fill_value(value, dtype)

    @pytest.mark.parametrize(('value', 'dtype'), FILL_VALUES)
    def test_fill_value_decoding_is_unchanged(self, value, dtype) -> None:
        """A fill value decodes the way it used to."""
        dtype = numpy.dtype(dtype)
        encoded = serialization.encode_fill_value(value, dtype)

        ours = serialization.decode_fill_value(encoded, dtype)
        theirs = self.metadata().decode_fill_value(encoded, dtype)

        if value is None:
            assert ours is None
            assert theirs is None
        elif isinstance(value, float) and numpy.isnan(value):
            assert numpy.isnan(ours)
            assert numpy.isnan(theirs)
        else:
            assert ours == theirs

    @pytest.mark.parametrize('dtype', DTYPES)
    def test_dtype_encoding_is_unchanged(self, dtype) -> None:
        """A data type encodes the way it used to."""
        dtype = numpy.dtype(dtype)
        assert serialization.encode_dtype(
            dtype) == self.metadata().encode_dtype(dtype)
