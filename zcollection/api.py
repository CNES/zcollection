"""Public sync facade for zcollection."""

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
    """Create a new collection at ``path`` and return its handle.

    Convenience wrapper around :meth:`Collection.create`. If ``path`` is
    a string it is dispatched through :func:`~zcollection.open_store`
    using its URL scheme (``file://``, ``memory://``, ``s3://``,
    ``icechunk://``).

    Args:
        path: A URL or a pre-built :class:`~zcollection.store.Store`.
        schema: The dataset schema. Variables not depending on ``axis``
            become immutable.
        axis: Name of the partition axis (a dimension of ``schema``).
        partitioning: A :class:`~zcollection.partitioning.Partitioning`
            instance (e.g. ``Date``, ``Sequence``, ``GroupedSequence``).
        catalog_enabled: If ``True``, maintain a sharded ``_catalog/``
            of partition paths to make cold opens and listings O(1).
        overwrite: If ``True``, replace any existing root at ``path``.

    Returns:
        A writable :class:`~zcollection.collection.base.Collection` ready to ``insert``.

    Raises:
        ~zcollection.errors.CollectionExistsError: If a collection already exists at ``path``
            and ``overwrite=False``.

    """
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
    """Open an existing collection in ``r`` (read-only) or ``rw`` mode.

    Args:
        path: A URL or a pre-built :class:`~zcollection.store.Store`.
        mode: ``"r"`` for read-only access (default) or ``"rw"`` for
            full read-write.

    Returns:
        A :class:`~zcollection.collection.base.Collection` bound to the existing root. In ``"r"`` mode
        all mutating methods raise
        :class:`~zcollection.errors.ReadOnlyError`.

    Raises:
        ValueError: If ``mode`` is not one of ``"r"`` or ``"rw"``.
        ~zcollection.errors.CollectionNotFoundError: If no collection exists at ``path``.

    """
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
    "Collection",
    "CollectionNotFoundError",
    "ReadOnlyError",
    "create_collection",
    "open_collection",
)
