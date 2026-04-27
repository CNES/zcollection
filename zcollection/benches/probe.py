# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""I/O counters for benchmarks.

A :class:`CountingProbe` wraps a Zarr v3 store and increments per-operation
counters whenever the underlying store is exercised. It uses
:class:`zarr.storage.WrapperStore` so it stays oblivious to the backend
(local, obstore, memory) and never changes data semantics.
"""

from typing import Any
from collections import Counter

import zarr.storage


class CountingProbe(zarr.storage.WrapperStore):
    """Wrap a Zarr Store and count per-method invocations.

    Counters are stored in :attr:`counts`; reset between phases via
    :meth:`reset`. Counters tracked: ``get``, ``set``, ``delete``,
    ``list_dir``, ``exists``.
    """

    def __init__(
        self, store: Any, *, counts: Counter[str] | None = None
    ) -> None:
        """Initialize the probe wrapping ``store``."""
        super().__init__(store)
        self.counts: Counter[str] = counts if counts is not None else Counter()

    def _with_store(self, store: Any) -> CountingProbe:
        # Zarr clones the wrapper for read-only views; share counters across clones.
        return CountingProbe(store, counts=self.counts)

    def reset(self) -> None:
        """Clear all counters."""
        self.counts.clear()

    async def get(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """Forward ``get`` to the wrapped store and increment the counter."""
        self.counts["get"] += 1
        return await self._store.get(key, *args, **kwargs)

    async def get_partial_values(self, *args: Any, **kwargs: Any) -> Any:
        """Forward ``get_partial_values`` and increment the ``get`` counter."""
        self.counts["get"] += 1
        return await self._store.get_partial_values(*args, **kwargs)

    async def set(self, key: str, value: Any, **kwargs: Any) -> Any:
        """Forward ``set`` to the wrapped store and increment the counter."""
        self.counts["set"] += 1
        return await self._store.set(key, value, **kwargs)

    async def set_if_not_exists(self, key: str, value: Any) -> Any:
        """Forward ``set_if_not_exists`` and increment the ``set`` counter."""
        self.counts["set"] += 1
        return await self._store.set_if_not_exists(key, value)

    async def delete(self, key: str) -> Any:
        """Forward ``delete`` and increment the counter."""
        self.counts["delete"] += 1
        return await self._store.delete(key)

    async def delete_dir(self, prefix: str) -> Any:
        """Forward ``delete_dir`` and increment the ``delete`` counter."""
        self.counts["delete"] += 1
        return await self._store.delete_dir(prefix)

    async def exists(self, key: str) -> bool:
        """Forward ``exists`` and increment the counter."""
        self.counts["exists"] += 1
        return await self._store.exists(key)

    def list_dir(self, prefix: str) -> Any:
        """Forward ``list_dir`` and increment the counter."""
        self.counts["list_dir"] += 1
        return self._store.list_dir(prefix)

    def list_prefix(self, prefix: str) -> Any:
        """Forward ``list_prefix`` and increment the ``list_dir`` counter."""
        self.counts["list_dir"] += 1
        return self._store.list_prefix(prefix)
