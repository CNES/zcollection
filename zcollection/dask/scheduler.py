# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Process-global :class:`AsyncRunner`: a long-lived event loop in a thread.

The sync facade dispatches every async coroutine through this runner so
callers never juggle ``asyncio.run``. Inside a Dask worker we reuse the
worker's own loop instead of spinning a new one.
"""

from typing import Any, TypeVar
import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
import threading


T = TypeVar("T")


class AsyncRunner:
    """Owns one event loop driven by a single background thread."""

    def __init__(self) -> None:
        """Start the background event loop thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="zcollection-runner",
            daemon=True,
        )
        self._thread.start()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the underlying event loop."""
        return self._loop

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Schedule ``coro`` on the runner's loop and block until it finishes."""
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        """Stop the background loop and join its thread."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()


_RUNNER: AsyncRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_runner() -> AsyncRunner:
    """Return the process-global runner, creating it on first use."""
    global _RUNNER  # noqa: PLW0603 — process-global lazy singleton
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = AsyncRunner()
    return _RUNNER


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Block on ``coro`` using the global runner, or the current loop on a worker.

    On a Dask worker we already sit inside a running loop; in that rare case
    the caller is expected to pass an awaitable directly to Dask. We still
    accept the call for symmetry but use the global runner.

    Args:
        coro: An async coroutine to run to completion.

    Returns:
        The result of the coroutine.

    """
    return get_runner().run(coro)


def in_event_loop() -> bool:
    """Check if the current thread has a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def shutdown_runner() -> None:
    """Tear down the global runner (mainly for tests)."""
    global _RUNNER  # noqa: PLW0603
    if _RUNNER is not None:
        _RUNNER.close()
        _RUNNER = None


__all__ = (
    "Any",
    "AsyncRunner",
    "get_runner",
    "in_event_loop",
    "run_sync",
    "shutdown_runner",
)
