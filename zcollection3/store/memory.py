"""In-memory store for tests and ephemeral collections."""
from __future__ import annotations

import asyncio
from typing import Any, Iterator

import zarr.storage

from .base import Store


class MemoryStore(Store):
    """All keys held in a process-local dict; chiefly for tests."""

    def __init__(self) -> None:
        self._store = zarr.storage.MemoryStore()
        self._extras: dict[str, bytes] = {}

    @property
    def root_uri(self) -> str:
        return f"memory://{id(self):x}"

    def zarr_store(self) -> Any:
        return self._store

    def exists(self, key: str) -> bool:
        if key in self._extras:
            return True
        # Fall through to the zarr-managed namespace.
        return _zarr_contains(self._store, key)

    def read_bytes(self, key: str) -> bytes | None:
        return self._extras.get(key)

    def write_bytes(self, key: str, data: bytes) -> None:
        self._extras[key] = data

    def list_prefix(self, prefix: str) -> Iterator[str]:
        yield from self.list_dir(prefix)

    def list_dir(self, prefix: str) -> Iterator[str]:
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
        prefix_slash = prefix.rstrip("/") + "/"
        for key in list(self._extras):
            if key == prefix or key.startswith(prefix_slash):
                self._extras.pop(key, None)
        _run_sync(self._store.delete_dir(prefix.strip("/")))

    def __repr__(self) -> str:
        return self.root_uri


def _head_under(key: str, prefix: str) -> str | None:
    if not prefix:
        return key.split("/", 1)[0] or None
    if key == prefix or not key.startswith(prefix + "/"):
        return None
    rest = key[len(prefix) + 1 :]
    return rest.split("/", 1)[0] or None


def _run_sync(coro):
    return asyncio.run(coro)


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
