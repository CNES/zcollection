"""In-memory store for tests and ephemeral collections."""

from typing import Any
import asyncio
from collections.abc import Iterator

import zarr.storage

from .base import Store


class MemoryStore(Store):
    """All keys held in a process-local dict; chiefly for tests."""

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._store = zarr.storage.MemoryStore()
        self._extras: dict[str, bytes] = {}

    @property
    def root_uri(self) -> str:
        """Return a synthetic ``memory://`` URI keyed off the instance id."""
        return f"memory://{id(self):x}"

    def zarr_store(self) -> Any:
        """Return the underlying :class:`zarr.storage.MemoryStore`."""
        return self._store

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists in either the extras dict or zarr storage."""
        if key in self._extras:
            return True
        # Fall through to the zarr-managed namespace.
        return _zarr_contains(self._store, key)

    def read_bytes(self, key: str) -> bytes | None:
        """Return the raw bytes stored at ``key`` or ``None`` if absent."""
        return self._extras.get(key)

    def write_bytes(self, key: str, data: bytes) -> None:
        """Store ``data`` at ``key`` in the extras dict."""
        self._extras[key] = data

    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield direct children under ``prefix``."""
        yield from self.list_dir(prefix)

    def list_dir(self, prefix: str) -> Iterator[str]:
        """Yield direct children under ``prefix`` from extras and zarr storage."""
        prefix_norm = prefix.strip("/")
        seen: set[str] = set()
        for key in self._extras:
            head = _head_under(key, prefix_norm)
            if head and head not in seen:
                seen.add(head)
                yield head
        for head in _zarr_list_dir(self._store, prefix_norm):
            if head not in seen:
                seen.add(head)
                yield head

    def delete_prefix(self, prefix: str) -> None:
        """Recursively delete extras and zarr keys under ``prefix``."""
        prefix_slash = prefix.rstrip("/") + "/"
        for key in list(self._extras):
            if key == prefix or key.startswith(prefix_slash):
                self._extras.pop(key, None)
        _run_sync(self._store.delete_dir(prefix.strip("/")))

    def __repr__(self) -> str:
        """Return the store's root URI as its representation."""
        return self.root_uri


def _head_under(key: str, prefix: str) -> str | None:
    if not prefix:
        return key.split("/", 1)[0] or None
    if key == prefix or not key.startswith(prefix + "/"):
        return None
    rest = key[len(prefix) + 1 :]
    return rest.split("/", 1)[0] or None


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is running in this thread; offload to a fresh one in another thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _zarr_contains(store: Any, key: str) -> bool:
    async def _check() -> bool:
        if await store.exists(key):
            return True
        async for _ in store.list_dir(key):
            return True
        return False

    return _run_sync(_check())


def _zarr_list_dir(store: Any, prefix: str) -> Iterator[str]:
    async def _collect() -> list[str]:
        return [name async for name in store.list_dir(prefix)]

    return iter(_run_sync(_collect()))
