"""Public type aliases and protocols."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeAlias

if TYPE_CHECKING:
    import numpy

#: Mapping from dimension name to size.
DimSizes: TypeAlias = dict[str, int]

#: Concrete partition key: ordered tuple of (name, integer-value) pairs.
PartitionKey: TypeAlias = tuple[tuple[str, int], ...]

#: User-facing partition filter: either a boolean expression string
#: or a callable from a decoded partition dict to bool.
PartitionFilter: TypeAlias = str | Callable[[dict[str, int]], bool] | None


class IndexerCallable(Protocol):
    """Maps a partition key to a slice of rows within that partition."""

    def __call__(
        self,
        partition: dict[str, int],
    ) -> "slice | numpy.ndarray | None":
        ...


#: An indexer is either a callable or an iterable of (partition, slice) pairs.
Indexer: TypeAlias = IndexerCallable | None


class JSONSerializable(Protocol):
    """Protocol for objects with a JSON config round-trip."""

    def to_json(self) -> Any: ...
    @classmethod
    def from_json(cls, payload: Any) -> "JSONSerializable": ...
