from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

from .meta import Attribute


def dimensions(dimensions: dict[str, int]) -> str:
    """Get the string representation of the dimensions.

    Args:
        dimensions: The dimensions.

    Returns:
        The string representation of the dimensions.
    """
    return str(tuple(f'{name}: {value}' for name, value in dimensions.items()))


def _maybe_truncate(obj: Any, max_size: int) -> str:
    """Truncate the string representation of an object to the given length.

    Args:
        obj: An object.
        max_size: The maximum length of the string representation.

    Returns:
        The string representation of the object.
    """
    result = str(obj)
    if len(result) > max_size:
        return result[:max_size - 3] + '...'
    return result


def pretty_print(obj: Any, num_characters: int = 120) -> str:
    """Pretty print the object.

    Args:
        obj: An object.
        num_characters: The maximum number of characters per line.

    Returns:
        The pretty printed string representation of the object.
    """
    result = _maybe_truncate(obj, num_characters)
    return result + ' ' * max(num_characters - len(result), 0)


def calculate_column_width(items: Iterable) -> int:
    """Calculate the maximum width of a column.

    Args:
        items: An iterable of items.

    Returns:
        The maximum width of a column.
    """
    max_name = max(len(str(name)) for name in items)
    return max(max_name, 7)


def attributes(attrs: Sequence[Attribute]) -> Iterator[str]:
    """Get the string representation of the attributes.

    Args:
        attrs: The attributes.

    Returns:
        The string representation of the attributes.
    """
    width = calculate_column_width(item.name for item in attrs)
    for attr in attrs:
        name_str = f'    {attr.name:<{width}s}'
        yield pretty_print(f'{name_str}: {attr.value!r}')
