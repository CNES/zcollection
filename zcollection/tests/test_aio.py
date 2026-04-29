# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Direct tests for the public async facade in ``zcollection.aio``."""

import asyncio

import pytest

import zcollection as zc
from zcollection import aio
from zcollection.collection import Collection
from zcollection.errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    ReadOnlyError,
)


def _run(coro):
    return asyncio.run(coro)


def test_async_create_with_store_instance(schema, partitioning):
    """``aio.create_collection`` accepts a Store instance directly."""

    async def _scenario() -> Collection:
        store = zc.MemoryStore()
        return await aio.create_collection(
            store, schema=schema, axis="num", partitioning=partitioning
        )

    col = _run(_scenario())
    assert isinstance(col, Collection)


def test_async_create_with_url(tmp_path, schema, partitioning):
    """``aio.create_collection`` accepts a ``file://`` URL string."""

    async def _scenario() -> Collection:
        return await aio.create_collection(
            f"file://{tmp_path / 'col'}",
            schema=schema,
            axis="num",
            partitioning=partitioning,
        )

    col = _run(_scenario())
    assert isinstance(col, Collection)


def test_async_create_existing_raises(schema, partitioning):
    """A second create on the same store raises ``CollectionExistsError``."""
    store = zc.MemoryStore()

    async def _scenario() -> None:
        await aio.create_collection(
            store, schema=schema, axis="num", partitioning=partitioning
        )
        with pytest.raises(CollectionExistsError):
            await aio.create_collection(
                store, schema=schema, axis="num", partitioning=partitioning
            )

    _run(_scenario())


def test_async_open_missing_raises(tmp_path):
    """``aio.open_collection`` on a missing path raises ``CollectionNotFoundError``."""

    async def _scenario() -> None:
        with pytest.raises(CollectionNotFoundError):
            await aio.open_collection(f"file://{tmp_path / 'nope'}")

    _run(_scenario())


def test_async_open_read_only_blocks_writes(schema, partitioning, dataset):
    """Default mode ``r`` returns a read-only collection."""
    store = zc.MemoryStore()

    async def _scenario() -> None:
        col = await aio.create_collection(
            store, schema=schema, axis="num", partitioning=partitioning
        )
        await col.insert_async(dataset)
        ro = await aio.open_collection(store)
        assert ro.read_only is True
        with pytest.raises(ReadOnlyError):
            await ro.insert_async(dataset)

    _run(_scenario())


def test_async_open_rw_allows_writes(schema, partitioning, dataset):
    """``mode='rw'`` returns a writable collection."""
    store = zc.MemoryStore()

    async def _scenario() -> None:
        await aio.create_collection(
            store, schema=schema, axis="num", partitioning=partitioning
        )
        rw = await aio.open_collection(store, mode="rw")
        assert rw.read_only is False
        written = await rw.insert_async(dataset)
        assert sorted(written) == ["num=0", "num=1", "num=2"]

    _run(_scenario())


@pytest.mark.parametrize("mode", ["bogus", "", "RW", "read"])
def test_async_open_invalid_mode_raises(mode):
    """Unknown modes raise ``ValueError`` before touching the store."""
    store = zc.MemoryStore()

    async def _scenario() -> None:
        with pytest.raises(ValueError, match="mode must be"):
            await aio.open_collection(store, mode=mode)

    _run(_scenario())
