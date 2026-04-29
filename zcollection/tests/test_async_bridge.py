# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Direct tests for ``zcollection.store._async_bridge``."""

import asyncio

import pytest

from zcollection.store._async_bridge import run_sync, to_list_async


# --- run_sync ------------------------------------------------------


def test_run_sync_no_loop_running():
    """``run_sync`` works on the main thread with no running loop."""

    async def _coro() -> int:
        return 42

    assert run_sync(_coro()) == 42


def test_run_sync_inside_running_loop():
    """``run_sync`` is callable from inside an already-running loop."""

    async def _inner() -> int:
        return 7

    async def _outer() -> int:
        # Sync call from within a coroutine — the production hazard.
        return run_sync(_inner())

    assert asyncio.run(_outer()) == 7


def test_run_sync_propagates_exceptions():
    """Exceptions inside the coroutine surface to the caller."""

    async def _boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        run_sync(_boom())


def test_run_sync_propagates_exceptions_from_running_loop():
    """Exception propagation also works when offloaded to the worker thread."""

    async def _boom() -> None:
        raise ValueError("boom")

    async def _outer() -> None:
        run_sync(_boom())

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(_outer())


# --- to_list_async -------------------------------------------------


def test_to_list_async_drains_iterator():
    """``to_list_async`` collects an async iterator into a list."""

    async def _gen():
        for x in ["a", "b", "c"]:
            yield x

    assert to_list_async(_gen()) == ["a", "b", "c"]


def test_to_list_async_empty_iterator():
    """An empty async iterator yields an empty list."""

    async def _gen():
        if False:  # pragma: no cover - empty generator
            yield ""

    assert to_list_async(_gen()) == []


def test_to_list_async_inside_running_loop():
    """``to_list_async`` is also safe to call from inside a running loop."""

    async def _gen():
        for x in ["x", "y"]:
            yield x

    async def _outer():
        return to_list_async(_gen())

    assert asyncio.run(_outer()) == ["x", "y"]
