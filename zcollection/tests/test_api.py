# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Direct tests for the public sync facade in ``zcollection.api``."""

import pytest

import zcollection as zc
from zcollection import api
from zcollection.collection import Collection
from zcollection.errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    ReadOnlyError,
)


def test_create_collection_with_store_instance(schema, partitioning):
    """``create_collection`` accepts a Store instance directly."""
    store = zc.MemoryStore()
    col = api.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
    )
    assert isinstance(col, Collection)
    assert col.read_only is False


def test_create_collection_with_url(tmp_path, schema, partitioning):
    """``create_collection`` accepts a ``file://`` URL string."""
    col = api.create_collection(
        f"file://{tmp_path / 'col'}",
        schema=schema,
        axis="num",
        partitioning=partitioning,
    )
    assert isinstance(col, Collection)


def test_create_collection_existing_without_overwrite_raises(
    schema, partitioning
):
    """A second create on the same store raises ``CollectionExistsError``."""
    store = zc.MemoryStore()
    api.create_collection(
        store, schema=schema, axis="num", partitioning=partitioning
    )
    with pytest.raises(CollectionExistsError):
        api.create_collection(
            store, schema=schema, axis="num", partitioning=partitioning
        )


def test_create_collection_overwrite_true_replaces(schema, partitioning):
    """``overwrite=True`` replaces the existing root."""
    store = zc.MemoryStore()
    api.create_collection(
        store, schema=schema, axis="num", partitioning=partitioning
    )
    col = api.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    assert isinstance(col, Collection)


def test_open_collection_missing_raises_not_found(tmp_path):
    """``open_collection`` on a non-existent path raises ``CollectionNotFoundError``."""
    with pytest.raises(CollectionNotFoundError):
        api.open_collection(f"file://{tmp_path / 'nope'}")


def test_open_collection_read_only_default(schema, partitioning, dataset):
    """Default mode is read-only, mutating ops raise ``ReadOnlyError``."""
    store = zc.MemoryStore()
    api.create_collection(
        store, schema=schema, axis="num", partitioning=partitioning
    ).insert(dataset)

    col = api.open_collection(store)
    assert col.read_only is True
    with pytest.raises(ReadOnlyError):
        col.insert(dataset)


def test_open_collection_rw_allows_writes(schema, partitioning, dataset):
    """``mode='rw'`` returns a writable collection."""
    store = zc.MemoryStore()
    api.create_collection(
        store, schema=schema, axis="num", partitioning=partitioning
    )
    col = api.open_collection(store, mode="rw")
    assert col.read_only is False
    written = col.insert(dataset)
    assert sorted(written) == ["num=0", "num=1", "num=2"]


@pytest.mark.parametrize("mode", ["bogus", "", "RW", "read", None])
def test_open_collection_invalid_mode_raises(mode):
    """Unknown modes raise ``ValueError`` before touching the store."""
    store = zc.MemoryStore()
    with pytest.raises(ValueError, match="mode must be"):
        api.open_collection(store, mode=mode)
