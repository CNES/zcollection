# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Merge strategies for inserting into an already-existing partition.

A :class:`MergeCallable` is the contract: given the partition already
on disk and the dataset just sliced for that partition, return the
dataset to actually write. ``Collection.insert(merge=…)`` looks up
strategies through :func:`resolve`, which accepts either a built-in
string alias or a :class:`MergeCallable` directly.

Built-in strategies:

- :func:`replace` *(default)* — the inserted dataset wins outright;
  the existing partition is overwritten.
- :func:`concat` — append inserted after existing along the
  partitioning dimension. No deduplication, no sorting.
- :func:`time_series` — drop the existing rows whose axis value falls
  in ``[inserted_min, inserted_max]``, concatenate, and sort the merged
  result by axis. Useful when a re-acquired block fully covers a
  contiguous time window.
- :func:`upsert` — row-wise replace-or-add by axis equality. Existing
  rows whose axis value matches an inserted row are dropped; the rest
  are kept; the result is concatenated with the inserted block and
  sorted by axis. The optional ``tolerance`` argument relaxes the
  match to "nearest within window", useful when re-acquired
  timestamps are jittered by clock drift.

Building a tolerance-aware strategy: :func:`upsert_within` returns a
:class:`MergeCallable` with the tolerance baked in, suitable for
passing directly to ``Collection.insert(merge=…)`` (the string alias
``"upsert"`` only covers exact-equality matching).
"""

from typing import Any, Protocol

import numpy

from ..data import Dataset, Variable


class MergeCallable(Protocol):
    """Contract for a merge strategy.

    Implementations are invoked by ``Collection.insert_async`` whenever
    the dataset slice destined for a partition collides with an
    existing on-disk partition. The callable receives the two datasets
    plus two metadata strings, and returns the dataset that should
    actually be written.
    """

    def __call__(
        self,
        existing: Dataset,
        inserted: Dataset,
        *,
        axis: str,
        partitioning_dim: str,
    ) -> Dataset:
        """Compute the dataset to write for one partition collision.

        Args:
            existing: The partition's current on-disk content, already
                read into memory.
            inserted: The slice of the user-supplied dataset that maps
                to the same partition key.
            axis: Name of the variable used for row-wise comparisons
                (typically the time variable). Both ``existing`` and
                ``inserted`` are expected to expose it as a 1-D
                variable.
            partitioning_dim: Dimension along which slicing and
                concatenation happen — the same axis the collection
                is partitioned over.

        Returns:
            The dataset to be written. Its variables must match the
            collection's schema; its length along
            ``partitioning_dim`` may differ from either input (e.g.
            longer for :func:`concat`, sorted-and-deduped for
            :func:`upsert`).

        """
        ...


def replace(
    existing: Dataset,
    inserted: Dataset,
    *,
    axis: str,
    partitioning_dim: str,
) -> Dataset:
    """Drop ``existing`` entirely; only ``inserted`` is written.

    Args:
        existing: Dataset already on disk (ignored).
        inserted: New dataset to write.
        axis: Unused; kept for protocol compatibility.
        partitioning_dim: Unused; kept for protocol compatibility.

    Returns:
        ``inserted`` unchanged.

    """
    return inserted


def concat(
    existing: Dataset,
    inserted: Dataset,
    *,
    axis: str,
    partitioning_dim: str,
) -> Dataset:
    """Append ``inserted`` after ``existing`` along ``partitioning_dim``.

    Args:
        existing: Dataset already on disk for this partition.
        inserted: New dataset to append.
        axis: Unused; kept for protocol compatibility.
        partitioning_dim: Dimension along which to concatenate.

    Returns:
        The concatenation ``existing || inserted``. No deduplication or
        sorting is performed.

    """
    return _concat_along(existing, inserted, partitioning_dim)


def time_series(
    existing: Dataset,
    inserted: Dataset,
    *,
    axis: str,
    partitioning_dim: str,
) -> Dataset:
    """Time-aware merge: drop the existing window covered by ``inserted``.

    Rows in ``existing`` whose axis value falls inside
    ``[inserted_min, inserted_max]`` are dropped wholesale, the
    remaining rows are concatenated with ``inserted``, and the result
    is sorted by axis. The input axes do **not** need to be
    monotonic — the function sorts on the way out.

    Empty-input short-circuits: if ``inserted`` has zero rows along
    ``axis`` the function returns ``existing`` unchanged; if
    ``existing`` is empty it returns ``inserted`` unchanged.

    Args:
        existing: Dataset already on disk.
        inserted: New dataset.
        axis: Name of the variable used for the time-window comparison.
            Must be a 1-D variable present on both sides; comparable
            types include numeric and ``datetime64``.
        partitioning_dim: Dimension along which to slice and concat.

    Returns:
        The merged, axis-sorted dataset.

    Raises:
        ValueError: If ``axis`` is not a variable on both sides.

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

    For each row in ``existing``: keep it iff its axis value has no
    match in ``inserted[axis]``. The kept rows are concatenated with
    ``inserted`` and the result is sorted by axis. Use this when a new
    acquisition may partially overlap (replace) and partially extend
    (add) an existing partition without wiping the gaps in between.

    Empty-input short-circuits: if ``inserted`` has zero rows along
    ``axis`` the function returns ``existing`` unchanged; if
    ``existing`` is empty it returns ``inserted`` unchanged.

    ``tolerance`` controls what counts as a match:

    * ``None`` (default) — exact equality (``numpy.isin``).
    * a scalar (e.g. ``numpy.timedelta64(500, "ms")`` for datetime
      axes, or a float for numeric axes) — an existing row matches
      the nearest inserted row when
      ``|existing - nearest_inserted| <= tolerance``. Useful when
      re-acquired timestamps are jittered by clock drift.

    The string alias ``"upsert"`` covers only exact equality. For a
    tolerance-aware merge passed via ``Collection.insert(merge=…)``
    — which accepts a string alias *or* a callable, but not a string
    plus an argument — wrap the tolerance with :func:`upsert_within`
    and pass the returned callable::

        col.insert(
            ds,
            merge=zcollection.merge.upsert_within(numpy.timedelta64(500, "ms")),
        )

    Args:
        existing: Dataset already on disk.
        inserted: New dataset.
        axis: Name of the matching variable. Must be a 1-D variable
            present on both sides; comparable types include numeric
            and ``datetime64``.
        partitioning_dim: Dimension along which to slice rows.
        tolerance: ``None`` for exact equality, or a scalar for
            nearest-neighbour matching. For datetime axes pass a
            ``numpy.timedelta64`` (e.g. ``timedelta64(500, "ms")``);
            for numeric axes pass a plain float.

    Returns:
        The merged, axis-sorted dataset.

    Raises:
        ValueError: If ``axis`` is not a variable on both sides.

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

    The returned :class:`MergeCallable` is suitable for passing to
    ``Collection.insert(merge=…)``::

        import numpy
        from zcollection.collection import merge

        # 500 ms clock-drift tolerance on a datetime axis.
        col.insert(ds, merge=merge.upsert_within(numpy.timedelta64(500, "ms")))

        # Or on a numeric axis:
        col.insert(ds, merge=merge.upsert_within(1e-6))

    Args:
        tolerance: ``None`` for exact equality, or a scalar (numeric
            or ``numpy.timedelta64``) controlling the nearest-neighbour
            match window in :func:`upsert`.

    Returns:
        A :class:`MergeCallable` that calls :func:`upsert` with the
        given tolerance.

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
    """Concatenate ``left`` and ``right`` along ``dim``.

    Both datasets are expected to share the same schema (the result
    inherits ``left.schema`` and ``left.attrs`` without checking).
    Walks the full group tree, so nested-group variables that span
    ``dim`` via dimension inheritance are concatenated too. Variables
    that don't span ``dim`` are static across partitions by the
    schema's partitioned-or-immutable contract — they are passed
    through unchanged from the left side. Variables only present on
    one side are passed through unchanged from that side.
    """
    if not left.variables and not left.groups:
        return right
    if not right.variables and not right.groups:
        return left

    left_all = left.all_variables()
    right_all = right.all_variables()

    new_vars: dict[str, Variable] = {}
    for path, lvar in left_all.items():
        rvar = right_all.get(path)
        if rvar is None or dim not in lvar.dimensions:
            new_vars[path] = lvar
            continue
        ax = lvar.dimensions.index(dim)
        data = numpy.concatenate([lvar.to_numpy(), rvar.to_numpy()], axis=ax)
        new_vars[path] = Variable(lvar.schema, data)
    for path, rvar in right_all.items():
        if path not in new_vars:
            new_vars[path] = rvar
    return Dataset(schema=left.schema, variables=new_vars, attrs=left.attrs)


def _slice_dataset_bool(
    dataset: Dataset,
    dim: str,
    mask: numpy.ndarray,
) -> Dataset:
    """Boolean-slice every variable along ``dim`` (root + nested groups).

    Variables in nested groups whose own dim differs from ``dim`` are
    passed through unchanged — their length is independent of the
    partitioning axis.
    """
    new_vars: dict[str, Variable] = {}
    for path, var in dataset.all_variables().items():
        if dim in var.dimensions:
            ax = var.dimensions.index(dim)
            slicer: list[slice | numpy.ndarray] = [slice(None)] * var.ndim
            slicer[ax] = mask
            data = var.to_numpy()[tuple(slicer)]
        else:
            data = var.to_numpy()
        new_vars[path] = Variable(var.schema, data)
    return Dataset(
        schema=dataset.schema, variables=new_vars, attrs=dataset.attrs
    )


def _index_dataset(
    dataset: Dataset,
    dim: str,
    idx: numpy.ndarray,
) -> Dataset:
    """Reorder rows along ``dim`` using an integer index array.

    Applies to root and nested-group variables alike. Same fall-through
    as :func:`_slice_dataset_bool`: variables whose primary dim is
    independent of ``dim`` are kept as-is.
    """
    new_vars: dict[str, Variable] = {}
    for path, var in dataset.all_variables().items():
        if dim in var.dimensions:
            ax = var.dimensions.index(dim)
            slicer: list[slice | numpy.ndarray] = [slice(None)] * var.ndim
            slicer[ax] = idx
            data = var.to_numpy()[tuple(slicer)]
        else:
            data = var.to_numpy()
        new_vars[path] = Variable(var.schema, data)
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
    """Resolve ``strategy`` to a :class:`MergeCallable`.

    Used by ``Collection.insert_async`` to turn the user-supplied
    ``merge=`` argument into a callable.

    Args:
        strategy: One of:

            - ``None`` — falls back to :func:`replace` (the default
              behaviour when no merge is requested).
            - A string alias of a built-in strategy: one of
              ``"replace"``, ``"concat"``, ``"time_series"``,
              ``"upsert"``.
            - A :class:`MergeCallable` (any callable matching the
              :class:`MergeCallable` protocol, including the result
              of :func:`upsert_within`).

    Returns:
        The resolved :class:`MergeCallable`.

    Raises:
        KeyError: If ``strategy`` is a string that doesn't match a
            built-in alias.
        TypeError: If ``strategy`` is neither ``None``, a string, nor
            a callable.

    """
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
    "MergeCallable",
    "concat",
    "replace",
    "resolve",
    "time_series",
    "upsert",
    "upsert_within",
)
