# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Serialization of data types and fill values
===========================================

Collections store the data type and the fill value of their variables as
JSON. These functions are the encoding they are stored with, owned here so
that it stays the same whatever the storage layer underneath becomes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import base64
import binascii

import numpy

if TYPE_CHECKING:
    from collections.abc import Callable


def encode_dtype(dtype: numpy.dtype) -> str | list:
    """Encode a data type.

    Args:
        dtype: The data type.

    Returns:
        Its string form, or the list of its fields when it is structured.
    """
    return dtype.str if dtype.fields is None else dtype.descr


def decode_dtype(data: Any) -> numpy.dtype:
    """Decode a data type.

    Args:
        data: The encoded data type.

    Returns:
        The data type.
    """
    if isinstance(data, list):
        data = [tuple(item) for item in data]
    return numpy.dtype(data)


def _encode_float(value: Any, _dtype: numpy.dtype) -> Any:
    """Encode a floating point fill value, naming the values JSON lacks."""
    if numpy.isnan(value):
        return 'NaN'
    if numpy.isposinf(value):
        return 'Infinity'
    if numpy.isneginf(value):
        return '-Infinity'
    return float(value)


def _encode_complex(value: Any, _dtype: numpy.dtype) -> Any:
    """Encode a complex fill value as its two parts."""
    scalar = numpy.dtype(complex).type()
    return (encode_fill_value(value.real, scalar.real.dtype),
            encode_fill_value(value.imag, scalar.imag.dtype))


#: How a fill value is encoded, by the kind of its data type.
_ENCODERS: dict[str, Callable[[Any, numpy.dtype], Any]] = {
    'f': _encode_float,
    'u': lambda value, _dtype: int(value),
    'i': lambda value, _dtype: int(value),
    'b': lambda value, _dtype: bool(value),
    'c': _encode_complex,
    'S': lambda value, _dtype: str(base64.standard_b64encode(value), 'ascii'),
    'V': lambda value, _dtype: str(base64.standard_b64encode(value), 'ascii'),
    'm': lambda value, _dtype: int(value.view('i8')),
    'M': lambda value, _dtype: int(value.view('i8')),
}


def _decode_float(data: Any, dtype: numpy.dtype) -> Any:
    """Decode a floating point fill value."""
    named = {'NaN': numpy.nan, 'Infinity': numpy.inf, '-Infinity': -numpy.inf}
    if isinstance(data, str) and data in named:
        return named[data]
    return numpy.array(data, dtype=dtype)[()]


def _decode_complex(data: Any, dtype: numpy.dtype) -> Any:
    """Decode a complex fill value from its two parts."""
    scalar = dtype.type()
    value = (decode_fill_value(data[0], scalar.real.dtype) +
             1j * decode_fill_value(data[1], scalar.imag.dtype))
    return numpy.array(value, dtype=dtype)[()]


def _decode_bytes(data: Any, dtype: numpy.dtype) -> Any:
    """Decode a byte string fill value."""
    try:
        data = base64.standard_b64decode(data)
    except binascii.Error:
        # Values written before the encoding was base64, such as a bare
        # zero, stay usable.
        pass
    return numpy.array(data, dtype=dtype)[()]


def _decode_void(data: Any, dtype: numpy.dtype) -> Any:
    """Decode a raw fill value."""
    return numpy.array(base64.standard_b64decode(data),
                       dtype=dtype.str).view(dtype)[()]


#: How a fill value is decoded, by the kind of its data type.
_DECODERS: dict[str, Callable[[Any, numpy.dtype], Any]] = {
    'f': _decode_float,
    'c': _decode_complex,
    'S': _decode_bytes,
    'V': _decode_void,
    'U': lambda data, _dtype: data,
}


def encode_fill_value(value: Any, dtype: numpy.dtype) -> Any:
    """Encode a fill value.

    Args:
        value: The fill value, or None when the variable declares none.
        dtype: The data type of the variable.

    Returns:
        A JSON-serializable form of the fill value.

    Raises:
        TypeError: If the data type holds Python objects.
    """
    if value is None:
        return None
    if dtype.hasobject:
        raise TypeError(f'Unsupported data type: {dtype!r}')
    encode = _ENCODERS.get(dtype.kind)
    return value if encode is None else encode(value, dtype)


def decode_fill_value(data: Any, dtype: numpy.dtype) -> Any:
    """Decode a fill value.

    Args:
        data: The encoded fill value, or None when the variable declares
            none.
        dtype: The data type of the variable.

    Returns:
        The fill value.

    Raises:
        TypeError: If the data type holds Python objects.
    """
    if data is None:
        return None
    if dtype.hasobject:
        raise TypeError(f'Unsupported data type: {dtype!r}')
    decode = _DECODERS.get(dtype.kind)
    return (numpy.array(data, dtype=dtype)[()] if decode is None else decode(
        data, dtype))
