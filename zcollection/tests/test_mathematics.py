# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Mathematics testing.
====================
"""
from .. import mathematics


def test_prod():
    """Test the product of an iterable."""
    assert mathematics.prod([]) == 1
    assert mathematics.prod([1]) == 1
    assert mathematics.prod([1, 2, 3]) == 6
    assert mathematics.prod([1, 2, 3, 4, 5]) == 120
    assert mathematics.prod([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 362880
    assert mathematics.prod([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 3628800
