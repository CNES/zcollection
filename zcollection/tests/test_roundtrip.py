"""End-to-end round-trip: create → insert → reopen → query."""
from __future__ import annotations

import numpy
import pytest

import zcollection3 as zc
from zcollection3.errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    ExpressionError,
    ReadOnlyError,
)
from zcollection3.partitioning import compile_filter


def test_create_insert_query(store, schema, dataset, partitioning):
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    written = col.insert(dataset)
    assert written == ["num=0", "num=1", "num=2"]
    assert list(col.partitions()) == written

    out = col.query()
    assert numpy.array_equal(out["num"].to_numpy(), dataset["num"].to_numpy())
    assert numpy.array_equal(out["value"].to_numpy(), dataset["value"].to_numpy())


def test_reopen_after_close(store, schema, dataset, partitioning):
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    reopened = zc.open_collection(store, mode="r")
    assert reopened.read_only
    assert reopened.axis == "num"
    out = reopened.query()
    assert numpy.array_equal(out["value"].to_numpy(), dataset["value"].to_numpy())


def test_filter_pushdown(store, schema, dataset, partitioning):
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    out = col.query(filters="num == 1")
    assert numpy.array_equal(out["num"].to_numpy(), numpy.array([1, 1]))

    out = col.query(filters="num >= 1 and num <= 2")
    assert set(out["num"].to_numpy().tolist()) == {1, 2}

    none = col.query(filters="num == 99")
    assert none is None


def test_select_variables(store, schema, dataset, partitioning):
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    out = col.query(variables=("num",))
    assert list(out.variables) == ["num"]


def test_drop_partitions(store, schema, dataset, partitioning):
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    dropped = col.drop_partitions(filters="num == 0")
    assert dropped == ["num=0"]
    assert list(col.partitions()) == ["num=1", "num=2"]


def test_read_only_blocks_writes(store, schema, dataset, partitioning):
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    ro = zc.open_collection(store, mode="r")
    with pytest.raises(ReadOnlyError):
        ro.insert(dataset)
    with pytest.raises(ReadOnlyError):
        ro.drop_partitions()


def test_create_existing_raises(store, schema, partitioning):
    zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    with pytest.raises(CollectionExistsError):
        zc.create_collection(
            store, schema=schema, axis="num",
            partitioning=partitioning,
        )


def test_open_missing_raises(tmp_path):
    with pytest.raises(CollectionNotFoundError):
        zc.open_collection(f"file://{tmp_path}/nope")


def test_invalid_filter_syntax():
    with pytest.raises(ExpressionError):
        compile_filter("__import__('os')")
    with pytest.raises(ExpressionError):
        compile_filter("num + 1 == 2")  # binary op disallowed
