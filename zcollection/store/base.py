# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Store ABC — narrow synchronous surface; the async path lives in ``io``."""

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
        """Commit the session; the base implementation is a no-op."""
        return None

    def rollback(self) -> None:
        """Roll back the session; the base implementation is a no-op."""
        return None

    def __enter__(self) -> Self:
        """Enter the session context."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Commit on clean exit, roll back on exception."""
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
    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present in the store."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes | None:
        """Return the raw bytes stored at ``key`` or ``None`` if absent."""

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key`` (overwriting any existing value)."""

    @abstractmethod
    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield child keys (one path segment) under ``prefix``."""

    @abstractmethod
    def list_dir(self, prefix: str) -> Iterator[str]:
        """Yield direct children (groups + arrays) under ``prefix``."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Recursively delete everything under ``prefix``."""

    @contextmanager
    def session(self) -> Iterator[StoreSession]:
        """Yield a :class:`StoreSession` for the duration of a write block."""
        sess = StoreSession()
        with sess:
            yield sess
