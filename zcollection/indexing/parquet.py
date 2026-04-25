"""Parquet-backed indexer.

An index is a single Parquet table holding ``(<key cols…>, partition,
start, stop)`` rows. ``builder`` produces the key columns for one
partition; ``Indexer.build`` walks every (filtered) partition and
concatenates the rows into the table.

Querying with a dict of equality filters yields a ``{partition: [(start,
stop), ...]}`` mapping that callers feed to per-partition slicing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy
import pyarrow
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
    """Lookup table over a :class:`Collection`'s rows."""

    def __init__(self, table: pyarrow.Table) -> None:
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
    ) -> "Indexer":
        """Build an index by walking the collection's partitions."""
        rows: list[pyarrow.Table] = []
        per_partition = collection.map(
            builder, filters=filters, variables=variables,
        )
        for path, payload in per_partition.items():
            tbl = _payload_to_table(payload, partition=path)
            if tbl.num_rows:
                rows.append(tbl)
        if not rows:
            empty = pyarrow.table({
                PARTITION_COL: pyarrow.array([], type=pyarrow.string()),
                START_COL: pyarrow.array([], type=pyarrow.int64()),
                STOP_COL: pyarrow.array([], type=pyarrow.int64()),
            })
            return cls(empty)
        return cls(pyarrow.concat_tables(rows, promote_options="default"))

    @classmethod
    def read(cls, path: str) -> "Indexer":
        return cls(pq.read_table(path))

    def write(self, path: str) -> None:
        pq.write_table(self._table, path)

    # --- accessors --------------------------------------------------

    @property
    def table(self) -> pyarrow.Table:
        return self._table

    @property
    def key_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self._table.column_names if c not in _RESERVED)

    def __len__(self) -> int:
        return self._table.num_rows

    # --- query ------------------------------------------------------

    def lookup(self, **predicates: Any) -> dict[str, list[tuple[int, int]]]:
        """Return ``{partition: [(start, stop), ...]}`` for matching rows."""
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
                cond = pyarrow.compute.is_in(arr, value_set=pyarrow.array(list(value)))
            else:
                cond = pyarrow.compute.equal(arr, value)
            mask = cond if mask is None else pyarrow.compute.and_(mask, cond)
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


def _payload_to_table(
    payload: numpy.ndarray | dict[str, numpy.ndarray],
    *,
    partition: str,
) -> pyarrow.Table:
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
    n = len(next(iter(cols.values())))
    cols[PARTITION_COL] = numpy.array([partition] * n, dtype=object)
    return pyarrow.table(cols)
