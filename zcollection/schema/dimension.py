"""Dimension metadata."""

from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Dimension:
    """A named axis of a dataset.

    ``size`` is ``None`` when unknown (typical for the partitioning axis).
    ``chunks`` is ``None`` when unconstrained (use full extent on write).
    ``shards`` is ``None`` when unsharded; otherwise the shard size in
    elements along this dimension.
    """

    name: str
    size: int | None = None
    chunks: int | None = None
    shards: int | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("size", self.size),
            ("chunks", self.chunks),
            ("shards", self.shards),
        ):
            if value is not None and value <= 0:
                raise ValueError(
                    f"Dimension.{field} must be positive or None, got {value!r}"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "chunks": self.chunks,
            "shards": self.shards,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Dimension:
        return cls(
            name=payload["name"],
            size=payload.get("size"),
            chunks=payload.get("chunks"),
            shards=payload.get("shards"),
        )
