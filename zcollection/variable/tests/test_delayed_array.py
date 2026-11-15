# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dask-backed variable tests
==========================
"""
from typing import Any

import dask.array.core
import dask.array.ma
import numpy
import pytest

from ..delayed_array import _as_dask_array
from .data import delayed_array

# pylint enable=unused-import


def test_as_dask_array() -> None:
    """Test converting array-like data to a Dask array."""
    dask_array: dask.array.core.Array
    fill_value: Any
    np_array: numpy.ndarray

    np_array = numpy.arange(10)
    dask_array, fill_value = _as_dask_array(np_array)
    assert isinstance(dask_array, dask.array.core.Array)
    assert fill_value is None

    np_array = numpy.ma.masked_equal(np_array, 5)
    dask_array, fill_value = _as_dask_array(np_array)
    assert isinstance(dask_array, dask.array.core.Array)
    assert fill_value == 5

    dask_array, fill_value = _as_dask_array(
        dask.array.ma.masked_equal(np_array, 5))
    assert isinstance(dask_array, dask.array.core.Array)
    assert fill_value == 5

    with pytest.raises(ValueError, match='does not match the fill value'):
        _as_dask_array(numpy.ma.masked_equal(np_array, 5), fill_value=6)

    with pytest.raises(ValueError, match='does not match the fill value'):
        _as_dask_array(numpy.ma.masked_equal(
            numpy.arange(numpy.datetime64(0, 'Y'),
                         numpy.datetime64(10, 'Y'),
                         dtype='M8[Y]'), numpy.datetime64(5, 'Y')),
                       fill_value=numpy.datetime64('NaT', 'Y'))

    dask_array, fill_value = _as_dask_array(numpy.ma.masked_equal(
        numpy.arange(numpy.datetime64(0, 'Y'),
                     numpy.datetime64(10, 'Y'),
                     dtype='M8[Y]'), numpy.datetime64('NaT', 'Y')),
                                            fill_value=numpy.datetime64(
                                                'NaT', 'Y'))
    assert isinstance(dask_array, dask.array.core.Array)


def test_persist() -> None:
    """A persisted variable keeps a Dask array holding the same values."""
    var = delayed_array()
    expected = var.values

    assert var.persist() is var
    assert isinstance(var.array, dask.array.core.Array)
    assert var.shape == expected.shape
    numpy.testing.assert_array_equal(var.values, expected)
