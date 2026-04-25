"""GroupedSequence — bucket the last ``Sequence`` variable into fixed runs.

Useful when, e.g., altimetry passes 1..100 should all live in one partition
rather than producing 100 partitions per cycle.
"""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import numpy

from ..errors import PartitionError
from .base import PartitionKey, keys_from_columns, runs_from_inverse
from .sequence import Sequence

if TYPE_CHECKING:
    from ..data import Dataset

_MIN_GROUP_SIZE = 2


class GroupedSequence(Sequence):
    """Like :class:`Sequence`, but groups the *last* variable into buckets of ``size``.

    The first ``len(variables) - 1`` variables continue to act as exact keys;
    the last variable is mapped to ``(value - start) // size * size + start``
    before grouping. ``size`` must be ≥ 2 (otherwise prefer :class:`Sequence`).
    """

    name = "grouped-sequence"

    def __init__(
        self,
        variables: tuple[str, ...],
        *,
        dimension: str,
        size: int,
        start: int = 0,
    ) -> None:
        """Initialize the grouped-sequence partitioning.

        Args:
            variables: The variable(s) to partition by; must be at least one.
            dimension: The dimension to partition along; if ``None``, inferred
                from the variable name.
            size: The bucket size for the last variable; must be ≥ 2.
            start: The bucket origin for the last variable; defaults to 0.

        """
        super().__init__(variables, dimension=dimension)
        if size < _MIN_GROUP_SIZE:
            raise PartitionError(
                f"GroupedSequence requires size >= {_MIN_GROUP_SIZE}; got {size}"
            )
        self._size = int(size)
        self._start = int(start)

    @property
    def size(self) -> int:
        """Return the bucket size for the grouped variable."""
        return self._size

    @property
    def start(self) -> int:
        """Return the bucket origin for the grouped variable."""
        return self._start

    def split(self, dataset: Dataset) -> Iterator[tuple[PartitionKey, slice]]:
        """Yield ``(key, slice)`` for each contiguous bucket in ``dataset``."""
        cols: dict[str, numpy.ndarray] = {}
        names = list(self.axis)
        for name in names:
            if name not in dataset:
                raise PartitionError(
                    f"variable {name!r} required for partitioning is missing"
                )
            var = dataset[name]
            if var.dimensions != (self.dimension,):
                raise PartitionError(
                    f"variable {name!r} must be 1-D along {self.dimension!r}; "
                    f"got dims={var.dimensions}"
                )
            cols[name] = var.to_numpy()

        last = names[-1]
        last_values = numpy.asarray(cols[last])
        if not numpy.issubdtype(last_values.dtype, numpy.integer):
            raise PartitionError(
                f"GroupedSequence requires integer values for {last!r}; "
                f"got dtype {last_values.dtype}"
            )
        cols[last] = (
            last_values - self._start
        ) // self._size * self._size + self._start

        unique, inverse = keys_from_columns(cols)
        for gid, sl in runs_from_inverse(inverse):
            row = unique[gid]
            key = tuple((n, int(row[i])) for i, n in enumerate(names))
            yield key, sl

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the partitioning."""
        return {
            "name": self.name,
            "variables": list(self.axis),
            "dimension": self.dimension,
            "size": self._size,
            "start": self._start,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> GroupedSequence:
        """Reconstruct a GroupedSequence from its JSON payload."""
        return cls(
            tuple(payload["variables"]),
            dimension=payload["dimension"],
            size=int(payload["size"]),
            start=int(payload.get("start", 0)),
        )
