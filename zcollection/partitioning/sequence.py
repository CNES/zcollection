"""Sequence partitioning — one partition per unique value tuple."""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import numpy

from ..errors import PartitionError
from .base import PartitionKey, keys_from_columns, runs_from_inverse

if TYPE_CHECKING:
    from ..data import Dataset


class Sequence:
    """Partition by the unique value tuples of one or more variables.

    The variables must all share the same single dimension, which becomes the
    partitioning axis.
    """

    name = "sequence"

    def __init__(self, variables: tuple[str, ...], *, dimension: str) -> None:
        if not variables:
            raise PartitionError("Sequence requires at least one variable")
        self._axis = tuple(variables)
        self._dimension = dimension

    @property
    def axis(self) -> tuple[str, ...]:
        return self._axis

    @property
    def dimension(self) -> str:
        return self._dimension

    def split(self, dataset: Dataset) -> Iterator[tuple[PartitionKey, slice]]:
        cols: dict[str, numpy.ndarray] = {}
        for name in self._axis:
            if name not in dataset:
                raise PartitionError(
                    f"variable {name!r} required for partitioning is missing"
                )
            var = dataset[name]
            if var.dimensions != (self._dimension,):
                raise PartitionError(
                    f"variable {name!r} must be 1-D along {self._dimension!r}; "
                    f"got dims={var.dimensions}"
                )
            cols[name] = var.to_numpy()

        unique, inverse = keys_from_columns(cols)
        for gid, sl in runs_from_inverse(inverse):
            row = unique[gid]
            key = tuple((n, _to_py(row[i])) for i, n in enumerate(self._axis))
            yield key, sl

    def encode(self, key: PartitionKey) -> str:
        return "/".join(f"{name}={value}" for name, value in key)

    def decode(self, path: str) -> PartitionKey:
        parts: list[tuple[str, int]] = []
        for token in path.strip("/").split("/"):
            if "=" not in token:
                raise PartitionError(
                    f"invalid partition path segment: {token!r}"
                )
            name, raw = token.split("=", 1)
            parts.append((name, int(raw)))
        return tuple(parts)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variables": list(self._axis),
            "dimension": self._dimension,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Sequence:
        return cls(
            variables=tuple(payload["variables"]),
            dimension=payload["dimension"],
        )


def _to_py(value: Any) -> int:
    arr = numpy.asarray(value)
    if numpy.issubdtype(arr.dtype, numpy.integer):
        return int(arr)
    raise PartitionError(
        f"Sequence partitioning requires integer keys; got dtype {arr.dtype}"
    )
