"""Public async facade for zcollection v3.

Mirrors :mod:`zcollection.api` but every entry point is a coroutine. This
module is the recommended surface when running inside an event loop (e.g.
FastAPI handlers, Jupyter ``%await`` cells, or other async workloads).
"""

from typing import TYPE_CHECKING

from .collection import Collection
from .store import Store, open_store


if TYPE_CHECKING:
    from .partitioning import Partitioning
    from .schema import DatasetSchema


async def create_collection(
    path: str | Store,
    *,
    schema: DatasetSchema,
    axis: str,
    partitioning: Partitioning,
    catalog_enabled: bool = False,
    overwrite: bool = False,
) -> Collection:
    """Async create. Returns the same :class:`~zcollection.collection.base.Collection` as the sync API."""
    store = path if isinstance(path, Store) else open_store(path)
    return Collection.create(
        store,
        schema=schema,
        axis=axis,
        partitioning=partitioning,
        catalog_enabled=catalog_enabled,
        overwrite=overwrite,
    )


async def open_collection(
    path: str | Store,
    *,
    mode: str = "r",
) -> Collection:
    """Async open. Returns a :class:`~zcollection.collection.base.Collection`; ``read_only`` flag follows ``mode``."""
    if mode not in {"r", "rw"}:
        raise ValueError(f"mode must be 'r' or 'rw'; got {mode!r}")
    read_only = mode == "r"
    store = (
        path
        if isinstance(path, Store)
        else open_store(path, read_only=read_only)
    )
    return Collection.open(store, read_only=read_only)


__all__ = (
    "create_collection",
    "open_collection",
)
