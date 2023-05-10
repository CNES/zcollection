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

# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ..delayed_array import _as_dask_array

# pylint enable=unused-import


def test_as_dask_array(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test converting array like to a dask array."""
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

    with pytest.raises(ValueError):
        _as_dask_array(numpy.ma.masked_equal(np_array, 5), fill_value=6)

    with pytest.raises(ValueError):
        _as_dask_array(numpy.ma.masked_equal(
            numpy.arange(numpy.datetime64(0, 'Y'),
                         numpy.datetime64(10, 'Y'),
                         dtype='M8[Y]'), numpy.datetime64(5, 'Y')),
                       fill_value=numpy.datetime64('NaT'))

    dask_array, fill_value = _as_dask_array(numpy.ma.masked_equal(
        numpy.arange(numpy.datetime64(0, 'Y'),
                     numpy.datetime64(10, 'Y'),
                     dtype='M8[Y]'), numpy.datetime64('NaT')),
                                            fill_value=numpy.datetime64('NaT'))
    assert isinstance(dask_array, dask.array.core.Array)
