# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing interface for the variable module
=========================================
"""
import numpy

from ..abc import not_equal


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
