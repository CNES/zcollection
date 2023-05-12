# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Mathematical functions.
=======================
"""
from typing import Iterable
import functools
import operator


def prod(iterable: Iterable) -> int:
    """Return the product of all elements in the given iterable.

    Args:
        iterable: An iterable containing numeric values.

    Returns:
        The product of all elements in the iterable. If the iterable is empty,
        returns 1.
    """
    return functools.reduce(operator.mul, iterable, 1)
