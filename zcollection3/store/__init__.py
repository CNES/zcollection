"""Store abstraction over Zarr v3 stores.

Phase 1 ships :class:`LocalStore` and :class:`MemoryStore`. The factory
function :func:`open_store` selects an implementation from a URL.
"""
from __future__ import annotations

from .base import Store, StoreSession
from .factory import open_store
from .layout import (
    CATALOG_DIR,
    IMMUTABLE_DIR,
    join_path,
    parent_path,
    relative_path,
)
from .local import LocalStore
from .memory import MemoryStore


def __getattr__(name: str):  # pragma: no cover — lazy optional dep
    if name == "IcechunkStore":
        from .icechunk_store import IcechunkStore  # noqa: PLC0415
        return IcechunkStore
    raise AttributeError(name)

__all__ = (
    "CATALOG_DIR",
    "IMMUTABLE_DIR",
    "LocalStore",
    "MemoryStore",
    "Store",
    "StoreSession",
    "join_path",
    "open_store",
    "parent_path",
    "relative_path",
)
