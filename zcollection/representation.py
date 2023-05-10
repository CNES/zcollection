# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Representation of dataset objects.
==================================
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

from .meta import Attribute


def dimensions(dims: dict[str, int]) -> str:
    """Returns a string representation of the dimensions.

    Args:
        dims: A dictionary containing the dimensions.

    Returns:
        A string representation of the dimensions in the form of a tuple, where
        each element of the tuple is a string containing the dimension name and
        its corresponding value.
    """
    return str(tuple(f'{name}: {value}' for name, value in dims.items()))


def _maybe_truncate(obj: Any, max_size: int) -> str:
    """Truncate the string representation of an object to the given length.

    Args:
        obj: An object.
        max_size: The maximum length of the string representation.

    Returns:
        The string representation of the object, truncated to the given length
        if necessary.
    """
    result = str(obj)
    if len(result) > max_size:
        return result[:max_size - 3] + '...'
    return result


def pretty_print(obj: Any, num_characters: int = 120) -> str:
    """Return a pretty printed string representation of the given object.

    Args:
        obj:
            An object to be pretty printed.
        num_characters:
            An integer representing the maximum number of
            characters per line.

    Returns:
        A string representation of the object, pretty printed with a maximum of
        `num_characters` characters per line.
    """
    result: str = _maybe_truncate(obj, num_characters)
    return result + ' ' * max(num_characters - len(result), 0)


def calculate_column_width(items: Iterable) -> int:
    """Calculate the maximum width of a column.

    Args:
        items: An iterable of items.

    Returns:
        The maximum width of a column.
    """
    max_name: int = max(len(str(name)) for name in items)
    return max(max_name, 7)


def attributes(attrs: Sequence[Attribute]) -> Iterator[str]:
    """Get the string representation of the attributes.

    Args:
        attrs: The attributes.

    Returns:
        The string representation of the attributes.
    """
    width: int = calculate_column_width(item.name for item in attrs)
    for attr in attrs:
        name_str: str = f'    {attr.name:<{width}s}'
        yield pretty_print(f'{name_str}: {attr.value!r}')
