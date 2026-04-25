"""On-disk format versioning for ``_zcollection.json``.

Phase 1 ships ``format_version=1``. Future breaking changes register an
``Upgrader`` callable that lifts an older payload into the current schema.
"""
from __future__ import annotations

from typing import Any, Callable

from ..errors import FormatVersionError

#: Current format version written to ``_zcollection.json``.
FORMAT_VERSION: int = 1

Upgrader = Callable[[dict[str, Any]], dict[str, Any]]
_REGISTRY: dict[tuple[int, int], Upgrader] = {}


def register(from_version: int, to_version: int) -> Callable[[Upgrader], Upgrader]:
    """Decorator to register an upgrader for a (from, to) version pair."""

    def wrap(fn: Upgrader) -> Upgrader:
        _REGISTRY[(from_version, to_version)] = fn
        return fn

    return wrap


def upgrade(payload: dict[str, Any]) -> dict[str, Any]:
    """Walk the upgrader chain to bring ``payload`` to ``FORMAT_VERSION``."""
    version = int(payload.get("format_version", 1))
    while version < FORMAT_VERSION:
        try:
            upgrader = _REGISTRY[(version, version + 1)]
        except KeyError as exc:
            raise FormatVersionError(
                f"no upgrader registered from version {version} to "
                f"{version + 1}"
            ) from exc
        payload = upgrader(payload)
        version += 1
    if version > FORMAT_VERSION:
        raise FormatVersionError(
            f"on-disk format version {version} is newer than supported "
            f"{FORMAT_VERSION}; please upgrade zcollection"
        )
    return payload
