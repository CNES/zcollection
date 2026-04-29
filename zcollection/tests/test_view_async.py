# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Async-path tests for ``zcollection.view.base.View``."""

import asyncio

import numpy
import pytest

import zcollection as zc
from zcollection.errors import ReadOnlyError
from zcollection.view import View, ViewReference


def _run(coro):
    return asyncio.run(coro)


def _make_view(tmp_path, schema, dataset, partitioning, *, read_only=False):
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
    if read_only:
        return View.open(view_store, base=base, read_only=True)
    return view


def test_update_async_writes_overlay(tmp_path, schema, dataset, partitioning):
    """``update_async`` produces and persists overlay variables."""
    view = _make_view(tmp_path, schema, dataset, partitioning)

    def _square(ds: zc.Dataset) -> dict[str, numpy.ndarray]:
        return {"value_squared": ds["value"].to_numpy() ** 2}

    written = _run(view.update_async(_square))
    assert sorted(written) == ["num=0", "num=1", "num=2"]

    out = _run(view.query_async())
    assert numpy.allclose(
        out["value_squared"].to_numpy(),
        out["value"].to_numpy() ** 2,
    )


def test_update_async_on_read_only_view_raises(
    tmp_path, schema, dataset, partitioning
):
    """A read-only view rejects ``update_async`` with ``ReadOnlyError``."""
    view = _make_view(tmp_path, schema, dataset, partitioning, read_only=True)

    async def _scenario() -> None:
        with pytest.raises(ReadOnlyError):
            await view.update_async(lambda ds: {})

    _run(_scenario())


def test_query_async_returns_none_when_no_partitions_match(
    tmp_path, schema, dataset, partitioning
):
    """Filtering to no partitions returns ``None`` from ``query_async``."""
    view = _make_view(tmp_path, schema, dataset, partitioning)

    out = _run(view.query_async(filters="num == 999"))
    assert out is None


def test_query_async_before_update_returns_base_only(
    tmp_path, schema, dataset, partitioning
):
    """Without an overlay write, ``query_async`` still returns the base data."""
    view = _make_view(tmp_path, schema, dataset, partitioning)

    out = _run(view.query_async())
    assert out is not None
    # Overlay variable not yet written.
    assert "value_squared" not in out.variables
    assert "value" in out.variables


def test_update_then_query_async_in_sequence(
    tmp_path, schema, dataset, partitioning
):
    """Sequential ``update_async`` then ``query_async`` from one event loop."""
    view = _make_view(tmp_path, schema, dataset, partitioning)

    def _double(ds: zc.Dataset) -> dict[str, numpy.ndarray]:
        return {"value_squared": ds["value"].to_numpy() * 2.0}

    async def _scenario():
        written = await view.update_async(_double)
        result = await view.query_async()
        return written, result

    written, result = _run(_scenario())
    assert sorted(written) == ["num=0", "num=1", "num=2"]
    assert numpy.allclose(
        result["value_squared"].to_numpy(),
        result["value"].to_numpy() * 2.0,
    )
