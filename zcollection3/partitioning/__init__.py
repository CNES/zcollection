"""Partitioning strategies and the typed filter expression engine."""
from __future__ import annotations

from typing import Any

from .base import Partitioning, PartitionKey
from .expression import compile_filter, key_to_dict
from .sequence import Sequence

__all__ = (
    "PartitionKey",
    "Partitioning",
    "Sequence",
    "compile_filter",
    "from_json",
    "key_to_dict",
)


def from_json(payload: dict[str, Any]) -> Partitioning:
    name = payload.get("name")
    if name == "sequence":
        return Sequence.from_json(payload)
    raise ValueError(f"unknown partitioning {name!r}")
