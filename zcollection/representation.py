# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Representation of dataset objects.
==================================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from .meta import Attribute


def dimensions(dims: dict[str, int]) -> str:
    """Format dataset dimensions as a string.

    Args:
        dims: A mapping of dimension names to sizes.

    Returns:
        A string containing the dimension names and sizes as a tuple.
    """
    return str(tuple(f'{name}: {value}' for name, value in dims.items()))


def _maybe_truncate(obj: Any, max_size: int) -> str:
    """Return an object's string representation, truncated if needed.

    Args:
        obj: The object to format.
        max_size: The maximum length.

    Returns:
        The string representation, limited to ``max_size`` characters.
    """
    result = str(obj)
    if len(result) > max_size:
        return result[:max_size - 3] + '...'
    return result


def pretty_print(obj: Any, num_characters: int = 120) -> str:
    """Format an object as a fixed-width string.

    Args:
        obj: The object to format.
        num_characters: The maximum width of the string.

    Returns:
        The formatted string, truncated or padded to the specified width.
    """
    result: str = _maybe_truncate(obj, num_characters)
    return result + ' ' * max(num_characters - len(result), 0)


def calculate_column_width(items: Iterable) -> int:
    """Calculate the width needed for a column.

    Args:
        items: The items to measure.

    Returns:
        The maximum item width, with a minimum of seven characters.
    """
    max_name: int = max(len(str(name)) for name in items)
    return max(max_name, 7)


def attributes(attrs: Sequence[Attribute]) -> Iterator[str]:
    """Format dataset attributes as strings.

    Args:
        attrs: The attributes to format.

    Yields:
        A string representation of each attribute.
    """
    width: int = calculate_column_width(item.name for item in attrs)
    for attr in attrs:
        name_str: str = f'    {attr.name:<{width}s}'
        yield pretty_print(f'{name_str}: {attr.value!r}')
