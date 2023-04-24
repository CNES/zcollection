# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing interface for the variable module
=========================================
"""
import numpy

# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ..abc import not_equal

# pylint enable=unused-import

# def test_maybe_truncate():
#     """Test the truncation of a string to a given length."""
#     data = list(range(1000))
#     # pylint: disable=protected-access
#     assert variable._maybe_truncate(data, 10) == '[0, 1, ...'
#     assert variable._maybe_truncate(data, len(str(data))) == str(data)
#     # pylint: enable=protected-access


def test_variable_not_equal():
    """Test if two values are different."""
    assert not_equal(1, 2) is True
    assert not_equal(1, 1) is False
    assert not_equal(1, '1') is True
    assert not_equal(1, numpy.nan) is True
    assert not_equal(numpy.nan, numpy.nan) is False
    assert not_equal(numpy.nan, 1) is True
    assert not_equal(numpy.datetime64('NaT'), numpy.datetime64('NaT')) is False
    assert not_equal(numpy.datetime64('NaT'), 1) is True
