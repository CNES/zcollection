"""Serialisation of the root ``_zcollection.json`` config.

The schema lives in one file at the collection root. This module owns the
JSON schema for that file. Per-partition Zarr v3 metadata (``zarr.json``) is
written by the io layer.
"""

from typing import TYPE_CHECKING, Any
import json

from .versioning import FORMAT_VERSION

if TYPE_CHECKING:
    from .dataset import DatasetSchema

#: Filename of the root config under the collection directory.
CONFIG_FILE: str = "_zcollection.json"


def encode_root(
    *,
    schema: DatasetSchema,
    axis: str,
    partitioning: dict[str, Any],
    catalog_enabled: bool,
    extras: dict[str, Any] | None = None,
) -> bytes:
    """Serialise the root config payload to bytes (UTF-8 JSON)."""
    doc: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "axis": axis,
        "partitioning": partitioning,
        "catalog": {"enabled": bool(catalog_enabled)},
        "schema": schema.to_json(),
    }
    if extras:
        doc["extras"] = extras
    return json.dumps(doc, indent=2, sort_keys=False).encode("utf-8")


def decode_root(payload: bytes | str) -> dict[str, Any]:
    """Parse the root config payload."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    doc = json.loads(payload)
    return doc
