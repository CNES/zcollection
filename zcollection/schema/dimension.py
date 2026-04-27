# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
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

    #: Dimension name, unique within the dataset.
    name: str
    #: Fixed size, or ``None`` if unknown (e.g. partitioning axis).
    size: int | None = None
    #: Chunk size along this dimension; ``None`` to use the full extent.
    chunks: int | None = None
    #: Shard size along this dimension; ``None`` for no sharding.
    shards: int | None = None

    def __post_init__(self) -> None:
        """Validate that size, chunks and shards are positive when set."""
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
        """Serialize the dimension to a JSON-compatible dict."""
        return {
            "name": self.name,
            "size": self.size,
            "chunks": self.chunks,
            "shards": self.shards,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Dimension:
        """Build a dimension from a JSON-compatible dict."""
        return cls(
            name=payload["name"],
            size=payload.get("size"),
            chunks=payload.get("chunks"),
            shards=payload.get("shards"),
        )
