"""Merge strategies for inserting into an already-existing partition.

A :data:`MergeCallable` takes the existing partition and the inserted dataset
and returns the dataset that should land on disk. The built-in strategies are:

- :func:`replace` (default): the new data wins, no merge.
- :func:`concat`: concatenate inserted after existing along the axis.
- :func:`time_series`: time-aware merge for monotonic datetime axes —
  the existing range overlapping ``[inserted_min, inserted_max]`` is dropped
  in bulk and replaced by the inserted block.
- :func:`upsert`: row-wise replace-or-add by axis equality. Existing rows
  whose axis value is **not** present in the inserted batch are preserved;
  matching axis values get the inserted row; new axis values are appended.
  Result is sorted by axis. Designed for re-acquisition workflows where a
  new product partially supplements an existing partition.
"""

from typing import Any, Protocol
from collections.abc import Callable

import numpy

from ..data import Dataset, Variable


class MergeCallable(Protocol):
    """Signature for a merge strategy."""

    def __call__(
        self,
        existing: Dataset,
        inserted: Dataset,
        *,
        axis: str,
        partitioning_dim: str,
    ) -> Dataset: ...


def replace(
    existing: Dataset,
    inserted: Dataset,
    *,
    axis: str,
    partitioning_dim: str,
) -> Dataset:
    """Drop ``existing`` entirely; only ``inserted`` is written."""
    return inserted


def concat(
    existing: Dataset,
    inserted: Dataset,
    *,
    axis: str,
    partitioning_dim: str,
) -> Dataset:
    """Append ``inserted`` after ``existing`` along ``partitioning_dim``."""
    return _concat_along(existing, inserted, partitioning_dim)


def time_series(
    existing: Dataset,
    inserted: Dataset,
    *,
    axis: str,
    partitioning_dim: str,
) -> Dataset:
    """Time-aware merge for monotonic datetime axes.

    The axis variable on each side is read; rows in ``existing`` that fall
    inside ``[inserted_min, inserted_max]`` are dropped, then the remaining
    existing slice is concatenated with ``inserted`` and re-sorted by axis.
    """
    if axis not in existing or axis not in inserted:
        raise ValueError(
            f"time_series merge requires axis variable {axis!r} on both sides"
        )

    existing_axis = existing[axis].to_numpy()
    inserted_axis = inserted[axis].to_numpy()
    if inserted_axis.size == 0:
        return existing
    if existing_axis.size == 0:
        return inserted

    lo, hi = inserted_axis.min(), inserted_axis.max()
    keep = (existing_axis < lo) | (existing_axis > hi)

    trimmed = _slice_dataset_bool(existing, partitioning_dim, keep)
    merged = _concat_along(trimmed, inserted, partitioning_dim)

    merged_axis = merged[axis].to_numpy()
    order = numpy.argsort(merged_axis, kind="stable")
    return _index_dataset(merged, partitioning_dim, order)


def upsert(
    existing: Dataset,
    inserted: Dataset,
    *,
    axis: str,
    partitioning_dim: str,
    tolerance: Any = None,
) -> Dataset:
    """Row-wise replace-or-add by axis proximity.

    For each row in ``existing``: keep it if its axis value has no match
    in ``inserted[axis]``. The kept rows are then concatenated with
    ``inserted`` and the result is sorted by axis. Use this when a new
    acquisition may partially overlap (replace) and partially extend
    (add) an existing partition without wiping the gaps in between.

    ``tolerance`` controls what counts as a match:

    * ``None`` (default) — exact equality.
    * a scalar (e.g. ``numpy.timedelta64(500, "ms")`` for datetime axes,
      or a float for numeric axes) — an existing row matches the nearest
      inserted row when ``|existing - nearest_inserted| <= tolerance``.
      Useful when re-acquired timestamps are jittered by clock drift.

    Use :func:`upsert_within` to build a ``MergeCallable`` with a fixed
    tolerance for use with the string-based registry::

        col.insert(
            ds,
            merge=zcollection.merge.upsert_within(numpy.timedelta64(500, "ms")),
        )
    """
    if axis not in existing or axis not in inserted:
        raise ValueError(
            f"upsert merge requires axis variable {axis!r} on both sides"
        )

    existing_axis = existing[axis].to_numpy()
    inserted_axis = inserted[axis].to_numpy()
    if inserted_axis.size == 0:
        return existing
    if existing_axis.size == 0:
        return inserted

    if tolerance is None:
        keep = ~numpy.isin(existing_axis, inserted_axis)
    else:
        sorted_ins = numpy.sort(inserted_axis)
        n = sorted_ins.size
        idx = numpy.searchsorted(sorted_ins, existing_axis)
        left = numpy.clip(idx - 1, 0, n - 1)
        right = numpy.clip(idx, 0, n - 1)
        # numpy handles |a - b| correctly for both numeric and datetime64.
        left_dist = numpy.abs(existing_axis - sorted_ins[left])
        right_dist = numpy.abs(existing_axis - sorted_ins[right])
        nearest = numpy.minimum(left_dist, right_dist)
        keep = nearest > tolerance

    trimmed = _slice_dataset_bool(existing, partitioning_dim, keep)
    merged = _concat_along(trimmed, inserted, partitioning_dim)

    merged_axis = merged[axis].to_numpy()
    order = numpy.argsort(merged_axis, kind="stable")
    return _index_dataset(merged, partitioning_dim, order)


def upsert_within(tolerance: Any) -> MergeCallable:
    """Return an :func:`upsert` strategy bound to ``tolerance``.

    Pass the result as ``merge=`` to :py:meth:`Collection.insert`.
    """

    def _strategy(
        existing: Dataset,
        inserted: Dataset,
        *,
        axis: str,
        partitioning_dim: str,
    ) -> Dataset:
        return upsert(
            existing,
            inserted,
            axis=axis,
            partitioning_dim=partitioning_dim,
            tolerance=tolerance,
        )

    return _strategy


# Helpers --------------------------------------------------------


def _concat_along(left: Dataset, right: Dataset, dim: str) -> Dataset:
    if not left.variables:
        return right
    if not right.variables:
        return left

    new_vars: dict[str, Variable] = {}
    for name, lvar in left.variables.items():
        if name not in right.variables:
            new_vars[name] = lvar
            continue
        rvar = right.variables[name]
        if dim in lvar.dimensions:
            ax = lvar.dimensions.index(dim)
            data = numpy.concatenate(
                [lvar.to_numpy(), rvar.to_numpy()], axis=ax
            )
        else:
            data = lvar.to_numpy()
        new_vars[name] = Variable(lvar.schema, data)
    for name, rvar in right.variables.items():
        if name not in new_vars:
            new_vars[name] = rvar
    return Dataset(schema=left.schema, variables=new_vars, attrs=left.attrs)


def _slice_dataset_bool(
    dataset: Dataset,
    dim: str,
    mask: numpy.ndarray,
) -> Dataset:
    new_vars: dict[str, Variable] = {}
    for name, var in dataset.variables.items():
        if dim in var.dimensions:
            ax = var.dimensions.index(dim)
            slicer: list[slice | numpy.ndarray] = [slice(None)] * var.ndim
            slicer[ax] = mask
            data = var.to_numpy()[tuple(slicer)]
        else:
            data = var.to_numpy()
        new_vars[name] = Variable(var.schema, data)
    return Dataset(
        schema=dataset.schema, variables=new_vars, attrs=dataset.attrs
    )


def _index_dataset(
    dataset: Dataset,
    dim: str,
    idx: numpy.ndarray,
) -> Dataset:
    new_vars: dict[str, Variable] = {}
    for name, var in dataset.variables.items():
        if dim in var.dimensions:
            ax = var.dimensions.index(dim)
            slicer: list[slice | numpy.ndarray] = [slice(None)] * var.ndim
            slicer[ax] = idx
            data = var.to_numpy()[tuple(slicer)]
        else:
            data = var.to_numpy()
        new_vars[name] = Variable(var.schema, data)
    return Dataset(
        schema=dataset.schema, variables=new_vars, attrs=dataset.attrs
    )


_BUILTIN: dict[str, MergeCallable] = {
    "replace": replace,
    "concat": concat,
    "time_series": time_series,
    "upsert": upsert,
}


def resolve(strategy: MergeCallable | str | None) -> MergeCallable:
    """Resolve ``strategy`` to a callable; ``None`` means :func:`replace`."""
    if strategy is None:
        return replace
    if callable(strategy):
        return strategy  # type: ignore[return-value]
    if isinstance(strategy, str):
        if strategy not in _BUILTIN:
            raise KeyError(
                f"unknown merge strategy {strategy!r}; choose from {tuple(_BUILTIN)!r}"
            )
        return _BUILTIN[strategy]
    raise TypeError(f"unsupported merge strategy: {strategy!r}")


__all__ = (
    "Callable",
    "MergeCallable",
    "concat",
    "replace",
    "resolve",
    "time_series",
    "upsert",
    "upsert_within",
)
