# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Targeted tests for previously-uncovered branches.

This file is organised by module — each section pokes at the specific
edge cases that were missing line coverage, not happy paths (those
already exist in the rest of the suite).
"""

import asyncio
import threading

import numpy
import pytest

import zcollection as zc
from zcollection.dask import scheduler as dsched
from zcollection.errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    ReadOnlyError,
    ZCollectionError,
)
from zcollection.io.immutable import (
    immutable_group_exists,
    open_immutable_dataset_async,
    write_immutable_dataset,
)
from zcollection.partitioning import Sequence
from zcollection.store.layout import IMMUTABLE_DIR
from zcollection.view import View, ViewReference


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------
# dask.scheduler — AsyncRunner lifecycle
# ---------------------------------------------------------------------


def test_async_runner_run_executes_coroutine():
    """``AsyncRunner.run`` schedules a coroutine and returns its result."""
    runner = dsched.AsyncRunner()
    try:

        async def _coro() -> int:
            return 11

        assert runner.run(_coro()) == 11
        assert runner.loop.is_running()
    finally:
        runner.close()
        # ``close`` stops the loop and joins its thread.
        assert runner.loop.is_closed()


def test_async_runner_close_is_idempotent():
    """Calling ``close`` twice does not raise."""
    runner = dsched.AsyncRunner()
    runner.close()
    runner.close()


def test_get_runner_singleton_and_shutdown():
    """``get_runner`` returns the same instance until shutdown."""
    dsched.shutdown_runner()
    try:
        a = dsched.get_runner()
        b = dsched.get_runner()
        assert a is b
    finally:
        dsched.shutdown_runner()
    # After shutdown the global is reset; a fresh call creates a new one.
    c = dsched.get_runner()
    assert c is not a
    dsched.shutdown_runner()


def test_in_event_loop_false_on_main_thread():
    """``in_event_loop`` returns False when no loop is running here."""
    assert dsched.in_event_loop() is False


def test_in_event_loop_true_inside_running_loop():
    """``in_event_loop`` returns True when called from inside a coroutine."""

    async def _scenario() -> bool:
        return dsched.in_event_loop()

    assert asyncio.run(_scenario()) is True


def test_run_sync_uses_global_runner():
    """``dask.scheduler.run_sync`` dispatches through the shared runner."""

    async def _coro() -> str:
        return "ok"

    try:
        assert dsched.run_sync(_coro()) == "ok"
    finally:
        dsched.shutdown_runner()


def test_get_runner_double_checked_locking_under_threads():
    """Concurrent first-use callers all converge on a single runner instance."""
    dsched.shutdown_runner()
    try:
        seen: list[dsched.AsyncRunner] = []
        barrier = threading.Barrier(8)

        def _grab() -> None:
            barrier.wait()
            seen.append(dsched.get_runner())

        threads = [threading.Thread(target=_grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All threads observe the same singleton.
        assert all(r is seen[0] for r in seen)
    finally:
        dsched.shutdown_runner()


# ---------------------------------------------------------------------
# io.immutable — write/read tree, edge cases
# ---------------------------------------------------------------------


def test_write_immutable_dataset_returns_empty_when_no_immutables():
    """Datasets with no immutable variables are a no-op."""
    schema = (
        zc.Schema()
        .with_dimension("num", size=None, chunks=4)
        .with_variable("num", dtype="int64", dimensions=("num",))
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "num": zc.Variable(
                schema.variables["num"],
                numpy.array([0, 1, 2], dtype="int64"),
            ),
        },
    )
    store = zc.MemoryStore()
    written = write_immutable_dataset(store, ds)
    assert written == []
    assert immutable_group_exists(store) is False


def _tagged(
    schema: zc.DatasetSchema, dataset: zc.Dataset, axis: str
) -> tuple[zc.DatasetSchema, zc.Dataset]:
    """Bind ``axis`` so non-axis-spanning vars get the immutable flag."""
    tagged_schema = schema.with_partition_axis(axis)
    tagged_dataset = zc.Dataset(
        schema=tagged_schema,
        variables={
            name: zc.Variable(tagged_schema.variables[name], v.to_numpy())
            for name, v in dataset.variables.items()
        },
        attrs=dict(dataset.attrs),
    )
    return tagged_schema, tagged_dataset


def test_immutable_dataset_roundtrips_and_lists_paths(schema, dataset):
    """Writing a dataset with axis-bound schema emits ``static`` to ``_immutable``."""
    tagged_schema, tagged_ds = _tagged(schema, dataset, "num")
    store = zc.MemoryStore()
    written = write_immutable_dataset(store, tagged_ds)
    assert "static" in written
    assert immutable_group_exists(store) is True

    out = _run(open_immutable_dataset_async(store, tagged_schema))
    assert "static" in out
    assert numpy.array_equal(
        out["static"].to_numpy(), dataset["static"].to_numpy()
    )


def test_open_immutable_dataset_async_returns_empty_when_missing(schema):
    """Reading from a store with no immutable group returns ``{}``."""
    store = zc.MemoryStore()
    tagged_schema = schema.with_partition_axis("num")
    out = _run(open_immutable_dataset_async(store, tagged_schema))
    assert out == {}


def test_open_immutable_dataset_async_filter_by_name(schema, dataset):
    """The ``variables`` filter selects by short name."""
    tagged_schema, tagged_ds = _tagged(schema, dataset, "num")
    store = zc.MemoryStore()
    write_immutable_dataset(store, tagged_ds)
    # Filter to a non-immutable name -> nothing returned.
    out = _run(
        open_immutable_dataset_async(store, tagged_schema, variables=["value"])
    )
    assert out == {}
    # Filter to the actual immutable name.
    out = _run(
        open_immutable_dataset_async(store, tagged_schema, variables=["static"])
    )
    assert set(out.keys()) == {"static"}


def test_immutable_group_exists_marker_path():
    """``immutable_group_exists`` checks the zarr.json marker."""
    store = zc.MemoryStore()
    assert immutable_group_exists(store) is False
    # Hand-create just the marker — emulating a partial state.
    store.write_bytes(f"{IMMUTABLE_DIR}/zarr.json", b"{}")
    assert immutable_group_exists(store) is True


# ---------------------------------------------------------------------
# view.base — error paths and empty-overlay short-circuits
# ---------------------------------------------------------------------


def test_view_create_existing_without_overwrite_raises(
    tmp_path, schema, dataset, partitioning
):
    """A second ``View.create`` on the same store raises ``CollectionExistsError``."""
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    base.insert(dataset)

    view_store = zc.LocalStore(tmp_path / "view")
    derived = zc.VariableSchema(
        name="derived",
        dtype=numpy.dtype("float32"),
        dimensions=("num", "x"),
    )
    View.create(
        view_store,
        base=base,
        variables=[derived],
        reference=ViewReference(uri=f"file://{tmp_path / 'col'}"),
    )
    with pytest.raises(CollectionExistsError):
        View.create(
            view_store,
            base=base,
            variables=[derived],
            reference=ViewReference(uri=f"file://{tmp_path / 'col'}"),
        )


def test_view_open_missing_raises_not_found(
    tmp_path, schema, dataset, partitioning
):
    """``View.open`` on a store with no view config raises ``CollectionNotFoundError``."""
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    base.insert(dataset)
    empty_store = zc.LocalStore(tmp_path / "view")
    with pytest.raises(CollectionNotFoundError):
        View.open(empty_store, base=base)


def test_view_create_rejects_unknown_dimension(tmp_path, schema, partitioning):
    """A view variable referencing an unknown dimension is rejected."""
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    bad = zc.VariableSchema(
        name="bogus",
        dtype=numpy.dtype("float32"),
        dimensions=("num", "missing"),
    )
    with pytest.raises(ZCollectionError, match="unknown dimension"):
        View.create(
            zc.LocalStore(tmp_path / "view"),
            base=base,
            variables=[bad],
            reference=ViewReference(uri=f"file://{tmp_path / 'col'}"),
        )


def test_view_query_async_with_only_base_variables(
    tmp_path, schema, dataset, partitioning
):
    """Asking only for base variables short-circuits the overlay load."""
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    base.insert(dataset)
    view_store = zc.LocalStore(tmp_path / "view")
    derived = zc.VariableSchema(
        name="value_squared",
        dtype=numpy.dtype("float32"),
        dimensions=("num", "x"),
    )
    view = View.create(
        view_store,
        base=base,
        variables=[derived],
        reference=ViewReference(uri=f"file://{tmp_path / 'col'}"),
    )
    out = _run(view.query_async(variables=["num", "value"]))
    assert out is not None
    # Overlay variable was not requested → not loaded.
    assert "value_squared" not in out.variables


def test_view_update_async_returning_unknown_var_is_ignored(
    tmp_path, schema, dataset, partitioning
):
    """Returning a name outside the view schema is silently dropped."""
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    base.insert(dataset)
    view_store = zc.LocalStore(tmp_path / "view")
    derived = zc.VariableSchema(
        name="derived",
        dtype=numpy.dtype("float32"),
        dimensions=("num", "x"),
    )
    view = View.create(
        view_store,
        base=base,
        variables=[derived],
        reference=ViewReference(uri=f"file://{tmp_path / 'col'}"),
    )
    written = _run(view.update_async(lambda ds: {"unknown": numpy.zeros(1)}))
    # Step still ran for each partition but no overlay variable was produced.
    assert sorted(written) == ["num=0", "num=1", "num=2"]
    # Overlay group was never populated → query returns base only.
    out = _run(view.query_async())
    assert "derived" not in out.variables


# ---------------------------------------------------------------------
# store.memory + factory — edge paths
# ---------------------------------------------------------------------


def test_memory_store_delete_prefix_removes_descendants():
    """``delete_prefix`` clears every key under the given prefix."""
    store = zc.MemoryStore()
    store.write_bytes("a/x", b"1")
    store.write_bytes("a/sub/y", b"2")
    store.write_bytes("b/z", b"3")
    store.delete_prefix("a")
    assert store.read_bytes("a/x") is None
    assert store.read_bytes("a/sub/y") is None
    # Sibling untouched.
    assert store.read_bytes("b/z") == b"3"


def test_open_store_rejects_unknown_scheme():
    """``open_store`` raises ``StoreError`` for unsupported URL schemes."""
    from zcollection.errors import StoreError

    with pytest.raises(StoreError, match="unrecognised"):
        zc.open_store("ftp://example.com/bucket")


def test_open_store_memory_returns_memory_store():
    """``memory://`` URLs construct a ``MemoryStore``."""
    store = zc.open_store("memory://anything")
    assert isinstance(store, zc.MemoryStore)


# ---------------------------------------------------------------------
# partitioning.base — small misc
# ---------------------------------------------------------------------


def test_sequence_encode_decode_roundtrip():
    """``Sequence`` encode/decode are inverses for a single-axis key."""
    part = Sequence(("num",), dimension="num")
    encoded = part.encode((("num", 3),))
    assert encoded == "num=3"
    decoded = part.decode(encoded)
    assert decoded == (("num", 3),)


def test_sequence_to_json_roundtrip():
    """``Sequence`` survives a ``to_json``/``from_json`` round-trip."""
    part = Sequence(("num",), dimension="num")
    rebuilt = Sequence.from_json(part.to_json())
    assert rebuilt.dimension == "num"


# ---------------------------------------------------------------------
# api.create_collection — overwrite path through a URL
# ---------------------------------------------------------------------


def test_create_collection_overwrite_via_url(tmp_path, schema, partitioning):
    """``create_collection(url, overwrite=True)`` replaces an existing root."""
    url = f"file://{tmp_path / 'col'}"
    zc.create_collection(
        url, schema=schema, axis="num", partitioning=partitioning
    )
    # Without overwrite the URL form must still raise.
    with pytest.raises(CollectionExistsError):
        zc.create_collection(
            url, schema=schema, axis="num", partitioning=partitioning
        )
    col = zc.create_collection(
        url,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    assert col.read_only is False


def test_open_collection_read_only_on_open_store(
    tmp_path, schema, partitioning, dataset
):
    """A ``Store`` instance passed to ``open_collection`` honours ``read_only``."""
    url = f"file://{tmp_path / 'col'}"
    zc.create_collection(
        url, schema=schema, axis="num", partitioning=partitioning
    ).insert(dataset)
    col = zc.open_collection(url, mode="r")
    with pytest.raises(ReadOnlyError):
        col.insert(dataset)
