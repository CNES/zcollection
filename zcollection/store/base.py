"""Store ABC.

Phase 1 stays sync at this layer: the io layer uses Zarr's sync API. When
we cut over to Zarr's async core in Phase 2 the same Store implementations
will be wrapped with the AsyncStore protocol. We keep the surface narrow on
purpose.
"""

from typing import Any, Self
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager


class StoreSession:
    """Per-operation session.

    For non-transactional stores this is a no-op. Icechunk's store will
    return a real session that supports commit/rollback.
    """

    transactional: bool = False

    def commit(self, message: str | None = None) -> None:
        return None

    def rollback(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


class Store(ABC):
    """Capability surface for a backend that can hold a Zarr v3 hierarchy.

    Implementations wrap a concrete :mod:`zarr.storage` Store and expose:

    - :meth:`zarr_store` to hand the raw Zarr store to the io layer.
    - :meth:`exists` / :meth:`delete_prefix` / :meth:`list_prefix` for the
      collection-level operations that don't go through Zarr.
    - :meth:`read_bytes` / :meth:`write_bytes` for the small JSON config
      file.
    """

    transactional: bool = False

    @property
    @abstractmethod
    def root_uri(self) -> str:
        """Human-readable URI for diagnostics."""

    @abstractmethod
    def zarr_store(self) -> Any:
        """Return the underlying zarr.abc.store.Store instance."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes | None: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield child keys (one path segment) under ``prefix``."""

    @abstractmethod
    def list_dir(self, prefix: str) -> Iterator[str]:
        """Yield direct children (groups + arrays) under ``prefix``."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    @contextmanager
    def session(self) -> Iterator[StoreSession]:
        sess = StoreSession()
        with sess:
            yield sess
