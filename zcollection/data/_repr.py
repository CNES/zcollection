# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Multi-line, xarray-like ``__repr__`` for :class:`Group` and :class:`Variable`.

The output mirrors xarray's tabular layout while exposing a synthetic byte
size for the dataset, each child group, and each variable so the user can
gauge memory/disk footprint at a glance.
"""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Iterator


if TYPE_CHECKING:
    from .group import Group
    from .variable import Variable


_BYTES_PER_KILOBYTE: int = 1024
_LINE_WIDTH: int = 120


def format_bytes(size: int) -> str:
    """Return a human-readable byte-size string.

    Uses base-1024 stepping (kB, MB, GB, TB, PB) to match the
    convention used by Zarr's ``nbytes_stored`` reporting. Examples::

        format_bytes(40)  # '40 B'   (no decimals under 1 kB)
        format_bytes(2 * 1024)  # '2.00 kB'
        format_bytes(5 * 1024**2)  # '5.00 MB'
        format_bytes(1024**5)  # '1.00 PB'
    """
    value = float(size)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < _BYTES_PER_KILOBYTE:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= _BYTES_PER_KILOBYTE
    return f"{value:.2f} PB"


def format_dimensions(dims: dict[str, int]) -> str:
    """Format a dimension mapping as ``"(time: 100, x_ac: 240)"``."""
    if not dims:
        return "()"
    return (
        "(" + ", ".join(f"{name}: {size}" for name, size in dims.items()) + ")"
    )


def calculate_column_width(items: Iterable[str]) -> int:
    """Return the column width needed to align ``items`` left-padded.

    The result is ``max(7, longest_item_length)``. The 7-char floor
    matches the convention used by xarray's repr — short names like
    ``"time"`` still get a comfortable column even when they're the
    only key in the table.
    """
    items = list(items)
    if not items:
        return 7
    return max(7, *(len(str(name)) for name in items))


def pretty_print(line: str, num_characters: int = _LINE_WIDTH) -> str:
    """Truncate ``line`` to ``num_characters`` with an ellipsis if too long."""
    if len(line) > num_characters:
        return line[: num_characters - 3] + "..."
    return line


def format_attributes(
    attrs: dict[str, Any], indent: str = "    "
) -> Iterator[str]:
    """Yield aligned ``"name : value"`` lines for an attribute mapping."""
    width = calculate_column_width(attrs)
    for name, value in attrs.items():
        line = f"{indent}{name:<{width}s} : {value!r}"
        yield pretty_print(line)


def _data_repr(var: Variable) -> str:
    """Return a short tag for the underlying array backend of ``var``."""
    data = var.data
    if data is None:
        return "<no data>"
    cls = type(data)
    qualname = f"{cls.__module__}.{cls.__qualname__}"
    if qualname.startswith("numpy."):
        return f"numpy.ndarray<size={format_bytes(var.nbytes)}>"
    if "dask" in cls.__module__:
        chunks = getattr(data, "chunksize", None) or getattr(
            data, "chunks", None
        )
        return f"dask.Array<chunks={chunks}>"
    return f"{cls.__name__}<size={format_bytes(var.nbytes)}>"


def variable_repr(var: Variable) -> str:
    """Return a multi-line representation of a single :class:`Variable`.

    The output has a header line (qualified class name, dimensions,
    dtype, total bytes), a backend tag (numpy / dask / other), and an
    optional ``Attributes:`` block listing the variable's attrs.
    """
    dims_str = format_dimensions(
        dict(zip(var.dimensions, var.shape, strict=False))
    )
    size_str = format_bytes(var.nbytes)
    head = (
        f"<{var.__module__}.{var.__class__.__name__} {dims_str} "
        f"{var.dtype}> Size: {size_str}"
    )
    lines = [head, f"  {_data_repr(var)}"]
    if var.attrs:
        lines.append("  Attributes:")
        lines.extend(format_attributes(var.attrs))
    return "\n".join(lines)


def _variable_line(name: str, var: Variable, width: int) -> str:
    """Format one variable as a single aligned line."""
    dims_str = "(" + ", ".join(var.dimensions) + ")"
    size_str = format_bytes(var.nbytes)
    line = (
        f"    {name:<{width}s} {dims_str:<24s} "
        f"{var.dtype!s:<10s} {size_str:>10s}  {_data_repr(var)}"
    )
    return pretty_print(line)


def group_repr(group: Group) -> str:
    """Return a multi-line, xarray-like representation of a :class:`Group`.

    The output has four sections:

    - a header line with the group's qualified class name, absolute
      path, and recursive byte size;
    - a ``Dimensions:`` line listing the dims declared on this group;
    - a ``Data variables:`` block (one line per own variable, or
      ``<empty>``);
    - an ``Attributes:`` block (only if the group has attrs);
    - a ``Groups:`` block summarising direct children (size + variable
      and subgroup counts; only if the group has children).

    Long lines are truncated by :func:`pretty_print`.
    """
    cls = group.__class__
    head = (
        f"<{cls.__module__}.{cls.__qualname__} {group.long_name()!r}> "
        f"Size: {format_bytes(group.nbytes)}"
    )
    lines = [head, f"  Dimensions: {format_dimensions(dict(group.dimensions))}"]

    # Data variables
    lines.append("Data variables:")
    if not group.variables:
        lines.append("    <empty>")
    else:
        width = calculate_column_width(group.variables)
        for name, var in group.variables.items():
            lines.append(_variable_line(name, var, width))

    # Attributes
    if group.attrs:
        lines.append("Attributes:")
        lines.extend(format_attributes(group.attrs))

    # Child groups (one summary line each)
    if group.groups:
        lines.append("Groups:")
        width = calculate_column_width(group.groups)
        for name, child in group.groups.items():
            size_str = format_bytes(child.nbytes)
            n_vars = len(child.variables)
            n_subs = len(child.groups)
            line = (
                f"    {name:<{width}s} {size_str:>10s}  "
                f"({n_vars} variable{'s' if n_vars != 1 else ''}, "
                f"{n_subs} subgroup{'s' if n_subs != 1 else ''})"
            )
            lines.append(pretty_print(line))

    return "\n".join(lines)
