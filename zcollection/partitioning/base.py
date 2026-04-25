"""Partitioning Protocol — pure-numpy partition key extraction."""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from collections.abc import Iterable, Iterator

import numpy

if TYPE_CHECKING:
    from ..data import Dataset

PartitionKey = tuple[tuple[str, int], ...]


@runtime_checkable
class Partitioning(Protocol):
    """Maps dataset rows along a partitioning axis to partition keys.

    Implementations operate on plain numpy — Dask is layered higher up.
    """

    name: str

    @property
    def axis(self) -> tuple[str, ...]:
        """Variables (along the partitioning dimension) used to derive the key."""
        ...

    @property
    def dimension(self) -> str:
        """The dataset dimension this partitioning splits."""
        ...

    def split(self, dataset: Dataset) -> Iterator[tuple[PartitionKey, slice]]:
        """Yield (partition_key, slice) for each contiguous run."""
        ...

    def encode(self, key: PartitionKey) -> str:
        """Encode a key as a relative storage path."""
        ...

    def decode(self, path: str) -> PartitionKey:
        """Decode a relative storage path into a key."""
        ...

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the partitioning."""
        ...


def keys_from_columns(
    columns: dict[str, numpy.ndarray],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return (unique_rows, inverse) for the stacked columns.

    ``unique_rows`` has shape (n_unique, n_cols); ``inverse`` maps each input
    row to its unique index. Sort order is lexicographic and stable.
    """
    if not columns:
        raise ValueError("at least one column is required")
    arrays = [numpy.asarray(c) for c in columns.values()]
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("all partition columns must share length")
    stacked = numpy.column_stack(arrays)
    # numpy.unique with axis=0 gives sorted unique rows + inverse.
    unique, inverse = numpy.unique(stacked, axis=0, return_inverse=True)
    return unique, inverse


def runs_from_inverse(inverse: numpy.ndarray) -> Iterable[tuple[int, slice]]:
    """Yield (group_id, slice) for each contiguous run of equal labels.

    A row that re-appears later produces a separate run — partitions can be
    fragmented. Callers concatenate fragments belonging to the same group.
    """
    if inverse.size == 0:
        return
    starts = numpy.flatnonzero(numpy.diff(inverse, prepend=inverse[0] - 1))
    starts = numpy.concatenate([starts, [inverse.size]])
    for i in range(len(starts) - 1):
        s, e = int(starts[i]), int(starts[i + 1])
        yield int(inverse[s]), slice(s, e)
