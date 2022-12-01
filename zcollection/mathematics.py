"""
Mathematical functions.
=======================
"""
from typing import Iterable
import functools
import operator


def prod(iterable: Iterable) -> int:
    """Get the product of an iterable.

    Args:
        iterable: An iterable.

    Returns:
        The product of the iterable.
    """
    return functools.reduce(operator.mul, iterable, 1)
