"""Runtime configuration for zcollection v3.

A thin wrapper around :mod:`zarr.config` that adds a few zcollection-specific
keys. Defaults are tuned for cloud workloads; bring them down for local FS.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import zarr.config


_DEFAULTS: dict[str, Any] = {
    # Concurrent partition writes/reads driven by the sync facade.
    "partition.concurrency": 8,
    # Default codec profile applied when the user doesn't override.
    "codec.profile": "cloud-balanced",
    # Whether to maintain the partition catalog at the collection root.
    "catalog.enabled": True,
}


def get(key: str) -> Any:
    """Return the current value for ``key``."""
    return _DEFAULTS[key]


def set(**kwargs: Any) -> None:
    """Update one or more zcollection config keys."""
    for k, v in kwargs.items():
        if k not in _DEFAULTS:
            raise KeyError(f"unknown zcollection config key: {k!r}")
        _DEFAULTS[k] = v


@contextmanager
def override(**kwargs: Any) -> Iterator[None]:
    """Temporarily override config values within a context."""
    saved = {k: _DEFAULTS[k] for k in kwargs}
    set(**kwargs)
    try:
        yield
    finally:
        _DEFAULTS.update(saved)


def configure_zarr(async_concurrency: int | None = None) -> None:
    """Plumb async concurrency into zarr.config."""
    if async_concurrency is not None:
        zarr.config.set({"async.concurrency": async_concurrency})
