"""Optional Dask bridge — submit per-partition coroutines as Dask futures.

If a Dask Client is reachable, :func:`dask_map_async` submits each coroutine
to the cluster and the worker's own event loop drives it. Otherwise it falls
back to :func:`asyncio.gather` on the local runner with bounded concurrency.
"""

from typing import Any, TypeVar
import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import Future

from .scheduler import get_runner

T = TypeVar("T")


def _try_get_client() -> Any | None:
    """Return the active distributed Client, or None if Dask isn't usable."""
    try:
        from distributed import get_client
    except ImportError:
        return None
    try:
        return get_client()
    except ValueError, RuntimeError:
        return None


def _await_in_worker(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run ``coro_factory()`` to completion on the worker's event loop.

    Each Dask worker has its own asyncio loop available through
    :func:`distributed.get_worker`; we schedule the coroutine on it and block.

    Args:
        coro_factory: A 0-arg callable that returns the coroutine to run.

    Returns:
        The result of the coroutine.

    """
    from distributed import get_worker

    worker = get_worker()
    loop = worker.loop.asyncio_loop  # type: ignore[attr-defined]
    coro = coro_factory()
    if not asyncio.iscoroutine(coro):

        async def _wrap() -> T:
            return await coro

        coro = _wrap()
    fut: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


def dask_map_async(
    coro_factories: list[Callable[[], Awaitable[T]]],
    *,
    client: Any | None = None,
    concurrency: int = 8,
) -> list[T]:
    """Run a list of coroutine factories in parallel.

    Each entry is a 0-arg callable that *returns* a coroutine — coroutines
    aren't picklable, so we ship the factories instead. If a Dask client is
    available, work is fanned out to it; otherwise the local runner gathers
    them with bounded concurrency.

    Args:
        coro_factories: A list of 0-arg callables that return coroutines to run.
        client: An optional Dask Client to use; if ``None``, we'll try to find
            one, and if that fails we'll run locally.
        concurrency: The maximum number of concurrent coroutines when running
            locally; ignored if a Dask client is used.

    Returns:
        A list of results from the coroutines, in the same order as the input
        factories.

    """
    if not coro_factories:
        return []

    if client is None:
        client = _try_get_client()

    if client is not None:
        futures = client.map(_await_in_worker, coro_factories, pure=False)
        return list(client.gather(futures))

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(factory: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await factory()

    async def _gather() -> list[T]:
        return await asyncio.gather(*[_bounded(f) for f in coro_factories])

    return get_runner().run(_gather())
