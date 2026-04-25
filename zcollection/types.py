"""Public type aliases and protocols."""

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias
from collections.abc import Callable


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
    ) -> slice | numpy.ndarray | None:
        """Return the rows (slice or fancy index) selected within ``partition``."""
        ...


#: An indexer is either a callable or an iterable of (partition, slice) pairs.
Indexer: TypeAlias = IndexerCallable | None


class JSONSerializable(Protocol):
    """Protocol for objects with a JSON config round-trip."""

    def to_json(self) -> Any:
        """Return a JSON-compatible representation of this object."""
        ...

    @classmethod
    def from_json(cls, payload: Any) -> JSONSerializable:
        """Build an instance from a JSON-compatible payload."""
        ...
