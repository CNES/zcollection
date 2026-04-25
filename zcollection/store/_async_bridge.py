"""Run a coroutine to completion from sync code, even when an event loop is already running."""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run ``coro`` and return its value.

    If no loop is running on the calling thread, ``asyncio.run`` is used.
    Otherwise the coroutine is offloaded to a worker thread that owns its
    own loop — this is what lets the sync ``Store`` methods stay callable
    from inside an async context (e.g. an in-flight ``insert_async``).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


def to_list_async(aiter: Any) -> list[str]:
    """Drain an async iterator into a list (run synchronously)."""
    async def _drain() -> list[str]:
        return [item async for item in aiter]
    return run_sync(_drain())
