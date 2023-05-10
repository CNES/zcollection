# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing variables
=================
"""
from typing import Any

import dask.array.core
import dask.array.ma
import numpy
import pytest

from ..array import _as_numpy_array


def test_as_numpy_array() -> None:
    """Test converting array like to a dask array."""
    array: numpy.ndarray
    fill_value: Any
    np_array: numpy.ndarray

    np_array = numpy.arange(10)
    array, fill_value = _as_numpy_array(np_array)
    assert isinstance(array, numpy.ndarray)
    assert not isinstance(array, numpy.ma.MaskedArray)
    assert fill_value is None

    np_array = numpy.ma.masked_equal(np_array, 5)
    array, fill_value = _as_numpy_array(np_array)
    assert isinstance(array, numpy.ndarray)
    assert not isinstance(array, numpy.ma.MaskedArray)
    assert fill_value == 5

    array, fill_value = _as_numpy_array(dask.array.ma.masked_equal(
        np_array, 5))
    assert isinstance(array, numpy.ndarray)
    assert not isinstance(array, numpy.ma.MaskedArray)
    assert fill_value == 5

    with pytest.raises(ValueError):
        _as_numpy_array(numpy.ma.masked_equal(np_array, 5), fill_value=6)

    with pytest.raises(ValueError):
        _as_numpy_array(numpy.ma.masked_equal(
            numpy.arange(numpy.datetime64(0, 'Y'),
                         numpy.datetime64(10, 'Y'),
                         dtype='M8[Y]'), numpy.datetime64(5, 'Y')),
                        fill_value=numpy.datetime64('NaT'))

    array, fill_value = _as_numpy_array(numpy.ma.masked_equal(
        numpy.arange(numpy.datetime64(0, 'Y'),
                     numpy.datetime64(10, 'Y'),
                     dtype='M8[Y]'), numpy.datetime64('NaT')),
                                        fill_value=numpy.datetime64('NaT'))
    assert isinstance(array, numpy.ndarray)
    assert not isinstance(array, numpy.ma.MaskedArray)
