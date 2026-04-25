"""On-disk format versioning for ``_zcollection.json``.

The current on-disk format is ``format_version=1``. Future breaking changes
register an ``Upgrader`` callable that lifts an older payload into the
current schema.
"""

from typing import Any
from collections.abc import Callable

from ..errors import FormatVersionError


#: Current format version written to ``_zcollection.json``.
FORMAT_VERSION: int = 1

Upgrader = Callable[[dict[str, Any]], dict[str, Any]]
_REGISTRY: dict[tuple[int, int], Upgrader] = {}


def register(
    from_version: int, to_version: int
) -> Callable[[Upgrader], Upgrader]:
    """Register an upgrader for a (from, to) version pair."""

    def wrap(fn: Upgrader) -> Upgrader:
        _REGISTRY[(from_version, to_version)] = fn
        return fn

    return wrap


def upgrade(payload: dict[str, Any]) -> dict[str, Any]:
    """Walk the upgrader chain to bring ``payload`` to ``FORMAT_VERSION``.

    Args:
        payload: The JSON-decoded contents of an on-disk schema.

    Returns:
        An upgraded payload compatible with the current ``FORMAT_VERSION``.

    Raises:
        FormatVersionError: If the payload's format version is newer than
            supported, or if no upgrader is registered for an intermediate version.

    """
    version = int(payload.get("format_version", 1))
    while version < FORMAT_VERSION:
        try:
            upgrader = _REGISTRY[(version, version + 1)]
        except KeyError as exc:
            raise FormatVersionError(
                f"no upgrader registered from version {version} to {version + 1}"
            ) from exc
        payload = upgrader(payload)
        version += 1
    if version > FORMAT_VERSION:
        raise FormatVersionError(
            f"on-disk format version {version} is newer than supported "
            f"{FORMAT_VERSION}; please upgrade zcollection"
        )
    return payload
