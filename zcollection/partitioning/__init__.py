"""Partitioning strategies and the typed filter expression engine."""

from typing import Any

from .base import Partitioning, PartitionKey
from .catalog import Catalog, CatalogState, reconcile as reconcile_catalog
from .date import Date
from .expression import compile_filter, key_to_dict
from .grouped import GroupedSequence
from .sequence import Sequence


__all__ = (
    "Catalog",
    "CatalogState",
    "Date",
    "GroupedSequence",
    "PartitionKey",
    "Partitioning",
    "Sequence",
    "compile_filter",
    "from_json",
    "key_to_dict",
    "reconcile_catalog",
)


def from_json(payload: dict[str, Any]) -> Partitioning:
    """Reconstruct a partitioning instance from its JSON payload.

    Args:
        payload: The JSON payload containing the partitioning information.

    Returns:
        An instance of a partitioning strategy based on the provided payload.

    Raises:
        ValueError: If the partitioning name in the payload is unknown.

    """
    name = payload.get("name")
    if name == "sequence":
        return Sequence.from_json(payload)
    if name == "grouped-sequence":
        return GroupedSequence.from_json(payload)
    if name == "date":
        return Date.from_json(payload)
    raise ValueError(f"unknown partitioning {name!r}")
