"""Public sync facade for zcollection v3 (Phase 1)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .collection import Collection
from .errors import CollectionNotFoundError, ReadOnlyError
from .store import Store, open_store

if TYPE_CHECKING:
    from .partitioning import Partitioning
    from .schema import DatasetSchema


def create_collection(
    path: str | Store,
    *,
    schema: DatasetSchema,
    axis: str,
    partitioning: Partitioning,
    catalog_enabled: bool = False,
    overwrite: bool = False,
) -> Collection:
    """Create a new collection at ``path`` and return its handle."""
    store = path if isinstance(path, Store) else open_store(path)
    return Collection.create(
        store,
        schema=schema,
        axis=axis,
        partitioning=partitioning,
        catalog_enabled=catalog_enabled,
        overwrite=overwrite,
    )


def open_collection(
    path: str | Store,
    *,
    mode: str = "r",
) -> Collection:
    """Open an existing collection in ``r`` (read-only) or ``rw`` mode."""
    if mode not in {"r", "rw"}:
        raise ValueError(f"mode must be 'r' or 'rw'; got {mode!r}")
    read_only = mode == "r"
    store = (
        path if isinstance(path, Store)
        else open_store(path, read_only=read_only)
    )
    return Collection.open(store, read_only=read_only)


__all__ = (
    "Collection",
    "CollectionNotFoundError",
    "ReadOnlyError",
    "create_collection",
    "open_collection",
)
