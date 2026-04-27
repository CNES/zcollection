# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""End-to-end round-trip: create → insert → reopen → query."""

import numpy
import pytest

import zcollection as zc
from zcollection.errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    ExpressionError,
    ReadOnlyError,
)
from zcollection.partitioning import compile_filter


def test_create_insert_query(store, schema, dataset, partitioning):
    """Create/insert/query round-trips bit-exactly."""
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    written = col.insert(dataset)
    assert written == ["num=0", "num=1", "num=2"]
    assert list(col.partitions()) == written

    out = col.query()
    assert numpy.array_equal(out["num"].to_numpy(), dataset["num"].to_numpy())
    assert numpy.array_equal(
        out["value"].to_numpy(), dataset["value"].to_numpy()
    )


def test_reopen_after_close(store, schema, dataset, partitioning):
    """Reopening a closed collection in read-only mode returns the same data."""
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    reopened = zc.open_collection(store, mode="r")
    assert reopened.read_only
    assert reopened.axis == "num"
    out = reopened.query()
    assert numpy.array_equal(
        out["value"].to_numpy(), dataset["value"].to_numpy()
    )


def test_filter_pushdown(store, schema, dataset, partitioning):
    """Filter expressions push down to partition selection."""
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    out = col.query(filters="num == 1")
    assert numpy.array_equal(out["num"].to_numpy(), numpy.array([1, 1]))

    out = col.query(filters="num >= 1 and num <= 2")
    assert set(out["num"].to_numpy().tolist()) == {1, 2}

    none = col.query(filters="num == 99")
    assert none is None


def test_select_variables(store, schema, dataset, partitioning):
    """Querying with ``variables=`` returns only the requested variables."""
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    out = col.query(variables=("num",))
    assert list(out.variables) == ["num"]


def test_drop_partitions(store, schema, dataset, partitioning):
    """``drop_partitions`` removes selected partitions from the listing."""
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    dropped = col.drop_partitions(filters="num == 0")
    assert dropped == ["num=0"]
    assert list(col.partitions()) == ["num=1", "num=2"]


def test_read_only_blocks_writes(store, schema, dataset, partitioning):
    """Read-only collections raise ReadOnlyError on insert and drop."""
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    ro = zc.open_collection(store, mode="r")
    with pytest.raises(ReadOnlyError):
        ro.insert(dataset)
    with pytest.raises(ReadOnlyError):
        ro.drop_partitions()


def test_create_existing_raises(store, schema, partitioning):
    """Creating over an existing collection without overwrite raises."""
    zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    with pytest.raises(CollectionExistsError):
        zc.create_collection(
            store,
            schema=schema,
            axis="num",
            partitioning=partitioning,
        )


def test_open_missing_raises(tmp_path):
    """Opening a missing collection raises CollectionNotFoundError."""
    with pytest.raises(CollectionNotFoundError):
        zc.open_collection(f"file://{tmp_path}/nope")


def test_invalid_filter_syntax():
    """``compile_filter`` rejects unsafe or unsupported expressions."""
    with pytest.raises(ExpressionError):
        compile_filter("__import__('os')")
    with pytest.raises(ExpressionError):
        compile_filter("num + 1 == 2")  # binary op disallowed
