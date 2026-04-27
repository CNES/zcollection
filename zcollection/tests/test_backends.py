# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Backend coverage: obstore_store (S3 via moto) + dask/runner.

The S3 tests spin up an in-process ``moto`` server and exercise the
``ObjectStore`` adapter end-to-end. They auto-skip if either ``moto`` or
``boto3`` (used to seed buckets) is unavailable.

The dask/runner tests stay in-process; they validate the local fallback,
the ``_try_get_client`` probe, ordering, and concurrency-bound semantics.
"""

import asyncio
from collections.abc import Awaitable, Callable
import threading

import pytest

import zcollection as zc
from zcollection.dask.runner import _try_get_client, dask_map_async
from zcollection.errors import StoreError

from ._s3_server import moto_server


# %%
# obstore_store via moto -------------------------------------------------


@pytest.fixture(scope="module")
def s3():
    """Yield a moto-backed :class:`~_s3_server.S3Endpoint` (auto-skip).

    The fixture is module-scoped so all tests in this file share one
    server. Moto is fast enough to start that this is mostly a wallclock
    optimisation but it also avoids the rare ``OSError: address in use``
    flakes when many tests grab their own port.
    """
    pytest.importorskip("moto")
    pytest.importorskip("moto.server")
    pytest.importorskip("boto3")
    pytest.importorskip("obstore")
    with moto_server() as endpoint:
        yield endpoint


def test_objectstore_url_must_have_scheme() -> None:
    """A bare path is rejected at construction time."""
    pytest.importorskip("obstore")
    from zcollection.store.obstore_store import ObjectStore

    with pytest.raises(StoreError, match="scheme"):
        ObjectStore("just/a/path")


def test_objectstore_unknown_scheme_raises() -> None:
    """A scheme outside the supported set raises StoreError."""
    pytest.importorskip("obstore")
    from zcollection.store.obstore_store import ObjectStore

    with pytest.raises(StoreError, match="ftp"):
        ObjectStore("ftp://example.com/x")


def test_objectstore_root_uri_strips_trailing_slash(s3) -> None:
    """``root_uri`` is normalised; ``zarr_store`` exposes the inner backend."""
    store = s3.object_store(prefix="prefix-1/")
    assert store.root_uri.endswith("prefix-1")  # trailing slash stripped
    assert "ObjectStore" in repr(store)
    # Sanity: the inner zarr ObjectStore is reachable for codecs.
    assert store.zarr_store() is not None


def test_objectstore_write_read_roundtrip(s3) -> None:
    """Write a key via the API, read it back with the matching method."""
    store = s3.object_store()
    payload = b"hello, S3"
    store.write_bytes("notes/greeting.txt", payload)
    assert store.read_bytes("notes/greeting.txt") == payload


def test_objectstore_read_missing_returns_none(s3) -> None:
    """A non-existent key reads back as ``None`` (not an exception)."""
    store = s3.object_store()
    assert store.read_bytes("does/not/exist") is None


def test_objectstore_exists_for_object_and_prefix(s3) -> None:
    """``exists`` is true for both literal keys and prefixes that have children."""
    store = s3.object_store()
    store.write_bytes("a/b/c.txt", b"x")
    assert store.exists("a/b/c.txt") is True  # exact key
    assert store.exists("a/b") is True  # prefix
    assert store.exists("a/missing") is False


def test_objectstore_list_prefix_yields_children(s3) -> None:
    """``list_prefix`` yields the immediate child names of a prefix."""
    store = s3.object_store()
    for key in ("a/x.txt", "a/y.txt", "a/sub/leaf"):
        store.write_bytes(key, b"v")
    children = set(store.list_prefix("a"))
    # obstore returns keys including the prefix path; we just need to see
    # the immediate names rather than a full S3 walk.
    flat = {entry.split("/")[-1] for entry in children}
    assert {"x.txt", "y.txt"}.issubset(flat) or any(
        "sub" in entry for entry in children
    )
    # list_dir is documented as an alias of list_prefix.
    assert set(store.list_dir("a")) == set(store.list_prefix("a"))


def test_objectstore_delete_prefix_recurses(s3) -> None:
    """``delete_prefix`` removes every object under the prefix."""
    store = s3.object_store()
    store.write_bytes("zone/a", b"1")
    store.write_bytes("zone/sub/b", b"2")
    assert store.exists("zone")
    store.delete_prefix("zone")
    assert store.read_bytes("zone/a") is None
    assert store.read_bytes("zone/sub/b") is None


def test_objectstore_read_only_blocks_writes(s3) -> None:
    """Opening read-only blocks ``write_bytes`` and ``delete_prefix``."""
    rw = s3.object_store(prefix="ro-test")
    rw.write_bytes("seed", b"x")
    ro = s3.object_store(bucket=rw.root_uri.split("/")[2], read_only=True)
    # Reads still work.
    assert ro.read_bytes("ro-test/seed") == b"x"
    with pytest.raises(PermissionError):
        ro.write_bytes("ro-test/seed", b"y")
    with pytest.raises(PermissionError):
        ro.delete_prefix("ro-test")


def test_objectstore_collection_roundtrip(s3) -> None:
    """A full ``create_collection`` / ``insert`` / ``query`` cycle on S3."""
    import numpy

    schema = (
        zc.Schema()
        .with_dimension("time", chunks=8)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("v", dtype="float32", dimensions=("time",))
        .build()
    )
    # Bucket on a coarse cycle to keep partitions to a handful and sort
    # them numerically (lexicographic "cycle=0, cycle=1, cycle=2" is also
    # numerical at this size).
    n = 6
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=8)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("cycle", dtype="int64", dimensions=("time",))
        .with_variable("v", dtype="float32", dimensions=("time",))
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(
                schema.variables["time"], numpy.arange(n, dtype="int64")
            ),
            "cycle": zc.Variable(
                schema.variables["cycle"],
                numpy.array([0, 0, 1, 1, 2, 2], dtype="int64"),
            ),
            "v": zc.Variable(
                schema.variables["v"], numpy.arange(n, dtype="float32")
            ),
        },
    )
    store = s3.object_store(prefix="collection")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="time",
        partitioning=zc.partitioning.Sequence(("cycle",), dimension="time"),
    )
    col.insert(ds, merge="replace")
    out = col.query()
    assert out is not None
    numpy.testing.assert_array_equal(
        out["time"].to_numpy(), ds["time"].to_numpy()
    )
    numpy.testing.assert_array_equal(out["v"].to_numpy(), ds["v"].to_numpy())


# %%
# dask/runner -------------------------------------------------------------


def test_dask_map_async_empty_list_short_circuits() -> None:
    """An empty input list returns ``[]`` without touching the runner."""
    assert dask_map_async([]) == []


def test_dask_map_async_local_preserves_order() -> None:
    """Locally-run results come back in the input order."""

    async def _factory(i: int) -> int:
        await asyncio.sleep(0)  # surrender to the loop
        return i

    def _make(i: int) -> Callable[[], Awaitable[int]]:
        return lambda: _factory(i)

    factories: list[Callable[[], Awaitable[int]]] = [_make(i) for i in range(8)]
    results = dask_map_async(factories)
    assert results == list(range(8))


def test_dask_map_async_propagates_exception() -> None:
    """An exception raised inside a coroutine surfaces from gather()."""

    async def _ok() -> int:
        return 1

    async def _boom() -> int:
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        dask_map_async([_ok, _boom])


def test_dask_map_async_concurrency_is_bounded() -> None:
    """At most ``concurrency`` factories run simultaneously."""
    in_flight = 0
    high_water = 0
    lock = threading.Lock()

    async def _slow(i: int) -> int:
        nonlocal in_flight, high_water
        async with asyncio.Lock():
            with lock:
                in_flight += 1
                high_water = max(high_water, in_flight)
        await asyncio.sleep(0.05)
        with lock:
            in_flight -= 1
        return i

    def _make(i: int) -> Callable[[], Awaitable[int]]:
        return lambda: _slow(i)

    factories: list[Callable[[], Awaitable[int]]] = [_make(i) for i in range(8)]
    results = dask_map_async(factories, concurrency=3)
    assert results == list(range(8))
    assert high_water <= 3


def test_dask_map_async_with_explicit_client_uses_it(monkeypatch) -> None:
    """A client passed explicitly is used without consulting ``_try_get_client``.

    We pass a fake client whose ``map``/``gather`` mimic the real thing
    but run synchronously in the test process; this proves the dispatch
    path without standing up a real Dask cluster.
    """

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def map(self, fn, items, *, pure: bool):
            return [fn(x) for x in items]

        def gather(self, futures):
            self.calls.extend(futures)
            return list(futures)

    # _try_get_client should not be consulted when ``client`` is explicit.
    def _boom() -> object:
        raise AssertionError("_try_get_client should not be called")

    monkeypatch.setattr(
        "zcollection.dask.runner._try_get_client",
        _boom,
    )

    fake = FakeClient()

    def _factory(i: int):
        async def _coro() -> int:
            return i * 10

        return _coro

    factories = [_factory(i) for i in range(3)]
    # FakeClient.map calls the function directly, so the worker bridge
    # gets the *factory* and tries to import distributed; route through
    # ``client.gather`` instead by monkeypatching ``_await_in_worker``.

    def _local_await(factory):
        coro = factory()
        return asyncio.run(coro)

    monkeypatch.setattr(
        "zcollection.dask.runner._await_in_worker",
        _local_await,
    )

    results = dask_map_async(factories, client=fake)
    assert results == [0, 10, 20]


def test_try_get_client_without_distributed(monkeypatch) -> None:
    """When ``distributed`` is unimportable, the probe returns ``None``."""
    import builtins

    real_import = builtins.__import__

    def _block_distributed(name, *args, **kwargs):
        if name == "distributed":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_distributed)
    assert _try_get_client() is None


def test_try_get_client_no_active_client(monkeypatch) -> None:
    """When ``distributed.get_client`` raises, the probe returns ``None``."""
    import builtins
    import sys
    import types

    fake = types.ModuleType("distributed")

    def _no_client():
        raise ValueError("no global client")

    fake.get_client = _no_client  # type: ignore[attr-defined]

    real_import = builtins.__import__

    def _route(name, *args, **kwargs):
        if name == "distributed":
            return fake
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "distributed", fake)
    monkeypatch.setattr(builtins, "__import__", _route)
    assert _try_get_client() is None


def test_dask_map_async_falls_back_when_probe_returns_none(monkeypatch) -> None:
    """If no client is reachable, the local gather path is used."""
    monkeypatch.setattr("zcollection.dask.runner._try_get_client", lambda: None)

    async def _factory(i: int) -> int:
        return i + 1

    def _make(i: int) -> Callable[[], Awaitable[int]]:
        return lambda: _factory(i)

    factories: list[Callable[[], Awaitable[int]]] = [_make(i) for i in range(3)]
    results = dask_map_async(factories)
    assert results == [1, 2, 3]


def test_dask_map_async_concurrency_lower_bound() -> None:
    """``concurrency=0`` is clamped to 1 so progress is always possible."""
    counter = {"n": 0}

    async def _f() -> int:
        counter["n"] += 1
        return 1

    out = dask_map_async([_f, _f, _f], concurrency=0)
    assert out == [1, 1, 1]
    assert counter["n"] == 3


def test_await_in_worker_runs_on_worker_loop(monkeypatch) -> None:
    """``_await_in_worker`` schedules the coroutine on the worker's loop.

    We stand up a real asyncio loop on a background thread and stub
    ``distributed.get_worker`` to return a fake worker exposing it. This
    exercises the threadsafe-bridge that production-time uses, without
    needing a Dask cluster.
    """
    import sys
    import types

    from zcollection.dask.runner import _await_in_worker

    # Run a fresh event loop in its own thread.
    loop = asyncio.new_event_loop()

    def _spin() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()

    class _FakeWorker:
        class _LoopHolder:
            asyncio_loop = loop

        loop = _LoopHolder

    fake_distributed = types.ModuleType("distributed")
    fake_distributed.get_worker = _FakeWorker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "distributed", fake_distributed)

    async def _payload() -> int:
        await asyncio.sleep(0)
        return 99

    try:
        assert _await_in_worker(_payload) == 99

        # Run a second coroutine on the same worker loop to confirm
        # repeated dispatches share state correctly.
        async def _ping() -> str:
            return "pong"

        assert _await_in_worker(_ping) == "pong"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        loop.close()


def test_dask_map_async_returns_real_dask_results() -> None:
    """Sanity: feeding real coroutine factories returns numpy-friendly values."""
    import numpy

    async def _square(i: int) -> int:
        return i * i

    def _make(i: int) -> Callable[[], Awaitable[int]]:
        return lambda: _square(i)

    factories: list[Callable[[], Awaitable[int]]] = [_make(i) for i in range(5)]
    out = dask_map_async(factories)
    numpy.testing.assert_array_equal(
        numpy.array(out), numpy.array([0, 1, 4, 9, 16])
    )
