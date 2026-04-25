"""Attribute metadata (JSON-clean key/value pairs)."""

from typing import Any

import numpy


def encode_value(value: Any) -> Any:
    """Coerce a value to something JSON can serialize.

    Args:
        value: The value to encode.

    Returns:
        A JSON-serializable representation of the value.

    """
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
    """Encode a full attribute dict for storage.

    Args:
        attrs: The attribute dict to encode, or ``None`` for no attributes.

    Returns:
        A JSON-serializable dict with all values encoded.

    """
    if not attrs:
        return {}
    return {str(k): encode_value(v) for k, v in attrs.items()}
