"""Attribute metadata (JSON-clean key/value pairs)."""
from __future__ import annotations

from typing import Any

import numpy


def encode_value(value: Any) -> Any:
    """Coerce a value to something JSON can serialize."""
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    return value


def encode_attrs(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """Encode a full attribute dict for storage."""
    if not attrs:
        return {}
    return {str(k): encode_value(v) for k, v in attrs.items()}


# Backwards-compatible alias for callers that still want a class form.
class Attribute:
    """Convenience wrapper for a single (name, value) pair.

    The schema itself stores attributes as plain dicts; this class is offered
    for users porting v2 code that constructed ``Attribute`` objects.
    """

    __slots__ = ("name", "value")

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = encode_value(value)

    def __repr__(self) -> str:
        return f"Attribute({self.name!r}, {self.value!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Attribute)
            and self.name == other.name
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.name, repr(self.value)))
