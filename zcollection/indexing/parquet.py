# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Parquet-backed indexer.

An index is a single Parquet table holding ``(<key cols…>, partition,
start, stop)`` rows. ``builder`` produces the key columns for one
partition; ``Indexer.build`` walks every (filtered) partition and
concatenates the rows into the table.

Querying with a dict of equality filters yields a ``{partition: [(start,
stop), ...]}`` mapping that callers feed to per-partition slicing.
"""

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterable

import numpy
import pyarrow
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq


if TYPE_CHECKING:
    from ..collection import Collection
    from ..data import Dataset

#: Reserved column names in the index.
PARTITION_COL: str = "_partition"
START_COL: str = "_start"
STOP_COL: str = "_stop"

_RESERVED = (PARTITION_COL, START_COL, STOP_COL)

#: Signature of the per-partition row generator.
#:
#: Returns a structured numpy array (or dict of equal-length arrays) with
#: the key columns plus integer ``_start`` / ``_stop`` columns delineating
#: contiguous row ranges within the partition.
IndexBuilder = Callable[["Dataset"], numpy.ndarray | dict[str, numpy.ndarray]]


class Indexer:
    """Lookup table over a :class:`~zcollection.collection.base.Collection`'s rows."""

    def __init__(self, table: pyarrow.Table) -> None:
        """Initialize the indexer with an underlying Parquet table.

        Args:
            table: PyArrow table containing key columns plus the reserved
                ``_partition``, ``_start`` and ``_stop`` columns.

        """
        self._table = table
        for col in (PARTITION_COL, START_COL, STOP_COL):
            if col not in table.column_names:
                raise ValueError(
                    f"index table missing required column {col!r}; "
                    f"got {table.column_names!r}"
                )

    # --- construction -----------------------------------------------

    @classmethod
    def build(
        cls,
        collection: Collection,
        *,
        builder: IndexBuilder,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Indexer:
        """Build an index by walking the collection's partitions.

        Args:
            collection: The :class:`~zcollection.Collection` to index.
            builder: Per-partition row generator. It receives a Dataset
                slice and must return either a structured numpy array
                (with named fields, including ``_start`` and ``_stop``)
                or a dict ``{column_name: 1-D numpy.ndarray}`` of equal
                length. The user-defined columns become the queryable
                key columns; ``_start`` / ``_stop`` carry contiguous row
                ranges within the partition.
            filters: Partition-key predicate forwarded to
                :meth:`Collection.partitions` to skip whole partitions.
            variables: Optional whitelist of variables to load before
                calling ``builder`` — keep this tight on cloud stores.

        Returns:
            An :class:`Indexer` whose Parquet table is the concatenation
            of every non-empty per-partition output, with a
            ``_partition`` column added by the builder pipeline.

        Raises:
            ValueError: If ``builder``'s output for any partition omits
                a ``_start`` or ``_stop`` column.
            TypeError: If ``builder`` returns something other than a
                structured array or a dict of arrays.

        """

        # Push the numpy->arrow conversion onto the per-partition step so
        # it runs in parallel on workers; the client only adds the
        # (dictionary-encoded, O(1)) ``_partition`` column at the end.
        def _worker(ds: Any) -> pyarrow.Table:
            return _payload_to_table_no_partition(builder(ds))

        rows: list[pyarrow.Table] = []
        per_partition = collection.map(
            _worker,
            filters=filters,
            variables=variables,
        )
        for path, tbl in per_partition.items():
            if tbl.num_rows:
                rows.append(_attach_partition_column(tbl, partition=path))
        if not rows:
            empty = pyarrow.table(
                {
                    PARTITION_COL: pyarrow.array([], type=pyarrow.string()),
                    START_COL: pyarrow.array([], type=pyarrow.int64()),
                    STOP_COL: pyarrow.array([], type=pyarrow.int64()),
                }
            )
            return cls(empty)
        return cls(pyarrow.concat_tables(rows, promote_options="default"))

    @classmethod
    def update(
        cls,
        collection: Collection,
        *,
        builder: IndexBuilder,
        existing: Indexer | str,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Indexer:
        """Rebuild the rows of partitions matching ``filters`` and merge them.

        Issue #33 fix: ``build`` always walks every partition and
        rematerialises the whole index — unworkable when the index is
        larger than RAM and only a handful of partitions changed. This
        method walks only the filtered partitions, drops every existing
        row whose ``_partition`` matches one of them (a left-anti merge
        on the partition column), then concatenates the fresh rows.

        Args:
            collection: The :class:`~zcollection.Collection` to re-index.
            builder: Same contract as :meth:`build`.
            existing: Either a loaded :class:`Indexer` or a path to an
                on-disk Parquet file. Rows for untouched partitions are
                copied straight through from this input.
            filters: Partition-key predicate forwarded to
                :meth:`Collection.partitions` — the set of partitions to
                re-index.
            variables: Optional whitelist of variables to load before
                calling ``builder``.

        Returns:
            A new :class:`Indexer` whose table is
            ``kept_existing + fresh_rows``. Column order follows the
            fresh build; schema promotion handles minor dtype drift.

        Raises:
            ValueError: If ``builder``'s output for any partition omits
                a ``_start`` or ``_stop`` column.
            TypeError: If ``builder`` returns something other than a
                structured array or a dict of arrays.

        """
        prior = existing if isinstance(existing, cls) else cls.read(existing)

        def _worker(ds: Any) -> pyarrow.Table:
            return _payload_to_table_no_partition(builder(ds))

        fresh_rows: list[pyarrow.Table] = []
        touched_partitions: list[str] = []
        per_partition = collection.map(
            _worker,
            filters=filters,
            variables=variables,
        )
        for path, tbl in per_partition.items():
            touched_partitions.append(path)
            if tbl.num_rows:
                fresh_rows.append(_attach_partition_column(tbl, partition=path))

        # Left-anti filter on the existing table: drop rows whose
        # ``_partition`` is in the set we just rebuilt.
        prior_tbl = prior.table
        if touched_partitions:
            mask = pc.invert(  # type: ignore[attr-defined]
                pc.is_in(  # type: ignore[attr-defined]
                    prior_tbl.column(PARTITION_COL),
                    value_set=pyarrow.array(touched_partitions),
                ),
            )
            kept = prior_tbl.filter(mask)
        else:
            kept = prior_tbl

        if not fresh_rows:
            return cls(kept)
        combined = pyarrow.concat_tables(
            [kept, *fresh_rows], promote_options="default"
        )
        return cls(combined)

    @classmethod
    def read(cls, path: str) -> Indexer:
        """Load an indexer from a Parquet file at ``path``."""
        return cls(pq.read_table(path))

    def write(self, path: str) -> None:
        """Write the index to ``path`` as a Parquet file."""
        pq.write_table(self._table, path)

    # --- accessors --------------------------------------------------

    @property
    def table(self) -> pyarrow.Table:
        """Return the underlying PyArrow table."""
        return self._table

    @property
    def key_columns(self) -> tuple[str, ...]:
        """Return the user-defined key column names (excluding reserved ones)."""
        return tuple(c for c in self._table.column_names if c not in _RESERVED)

    def __len__(self) -> int:
        """Return the number of rows in the index."""
        return self._table.num_rows

    # --- query ------------------------------------------------------

    def lookup(self, **predicates: Any) -> dict[str, list[tuple[int, int]]]:
        """Return ``{partition: [(start, stop), ...]}`` for matching rows.

        Filters are AND-ed: a row matches when every named predicate
        matches. A scalar predicate is equality; a list / tuple / set /
        ndarray predicate is set membership (``IN``).

        Args:
            **predicates: One ``key_column=value`` per filter. Allowed
                column names are exactly :attr:`key_columns`.

        Returns:
            A mapping from partition path to the list of contiguous
            ``(start, stop)`` row ranges that satisfy the predicates.
            Partitions with no matching row are absent from the mapping.

        Raises:
            KeyError: If a predicate names an unknown column.

        """
        unknown = set(predicates) - set(self.key_columns)
        if unknown:
            raise KeyError(
                f"unknown index columns: {sorted(unknown)}; "
                f"available: {self.key_columns!r}"
            )
        mask = None
        for col, value in predicates.items():
            arr = self._table.column(col)
            if isinstance(value, (list, tuple, set, numpy.ndarray)):
                cond = pc.is_in(arr, value_set=pyarrow.array(list(value)))  # type: ignore[attr-defined]
            else:
                cond = pc.equal(arr, value)  # type: ignore[attr-defined]
            mask = cond if mask is None else pc.and_(mask, cond)  # type: ignore[attr-defined]
        if mask is None:
            filtered = self._table
        else:
            filtered = self._table.filter(mask)

        out: dict[str, list[tuple[int, int]]] = {}
        partitions = filtered.column(PARTITION_COL).to_pylist()
        starts = filtered.column(START_COL).to_pylist()
        stops = filtered.column(STOP_COL).to_pylist()
        for p, s, e in zip(partitions, starts, stops, strict=True):
            out.setdefault(p, []).append((int(s), int(e)))
        return out


# --- helpers --------------------------------------------------------


def _payload_to_table_no_partition(
    payload: numpy.ndarray | dict[str, numpy.ndarray],
) -> pyarrow.Table:
    """Convert a builder payload to a PyArrow table (no partition column)."""
    if isinstance(payload, numpy.ndarray) and payload.dtype.names is not None:
        cols = {name: payload[name] for name in payload.dtype.names}
    elif isinstance(payload, dict):
        cols = dict(payload)
    else:
        raise TypeError(
            f"builder must return a structured array or dict; got {type(payload)}",
        )

    for col in (START_COL, STOP_COL):
        if col not in cols:
            raise ValueError(
                f"builder output missing required column {col!r}",
            )
    return pyarrow.table(cols)


def _attach_partition_column(
    table: pyarrow.Table, *, partition: str
) -> pyarrow.Table:
    """Append a dictionary-encoded ``_partition`` column with a single value.

    Cardinality of this column across the full index is ``#partitions``,
    not ``#rows``, so dictionary encoding keeps both in-memory and
    on-disk size proportional to the number of partitions.
    """
    n = table.num_rows
    partition_col = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array(numpy.zeros(n, dtype=numpy.int32)),
        pyarrow.array([partition], type=pyarrow.string()),
    )
    return table.append_column(PARTITION_COL, partition_col)
