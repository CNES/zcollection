# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Direct tests for ``zcollection.io.async_partition``."""

import asyncio

import numpy
import pytest

import zcollection as zc
from zcollection.io.async_partition import (
    open_partition_dataset_async,
    partition_exists_async,
    write_partition_dataset_async,
)
from zcollection.io.partition import open_partition_dataset, partition_exists


def _run(coro):
    return asyncio.run(coro)


def test_async_write_then_sync_read_roundtrips(schema, dataset):
    """An async-written partition is readable via the sync reader."""
    store = zc.MemoryStore()

    _run(write_partition_dataset_async(store, "p0", dataset))

    out = open_partition_dataset(store, "p0", schema)
    assert numpy.array_equal(out["num"].to_numpy(), dataset["num"].to_numpy())
    assert numpy.array_equal(
        out["value"].to_numpy(), dataset["value"].to_numpy()
    )


def test_async_read_matches_sync_read(schema, dataset):
    """``open_partition_dataset_async`` returns the same data as the sync reader."""
    store = zc.MemoryStore()
    _run(write_partition_dataset_async(store, "p0", dataset))

    sync_ds = open_partition_dataset(store, "p0", schema)
    async_ds = _run(open_partition_dataset_async(store, "p0", schema))

    for name in sync_ds.variables:
        assert numpy.array_equal(
            sync_ds[name].to_numpy(), async_ds[name].to_numpy()
        ), f"variable {name!r} mismatch"


def test_partition_exists_async_true_and_false(schema, dataset):
    """``partition_exists_async`` reflects presence of the partition."""
    store = zc.MemoryStore()
    assert _run(partition_exists_async(store, "p0")) is False
    _run(write_partition_dataset_async(store, "p0", dataset))
    assert _run(partition_exists_async(store, "p0")) is True
    # Mirrors the sync helper.
    assert partition_exists(store, "p0") is True


def test_partition_exists_async_distinguishes_partitions(schema, dataset):
    """Existence is per-partition, not collection-wide."""
    store = zc.MemoryStore()
    _run(write_partition_dataset_async(store, "p0", dataset))
    assert _run(partition_exists_async(store, "p0")) is True
    assert _run(partition_exists_async(store, "p1")) is False


def test_open_partition_dataset_async_missing_raises(schema):
    """Opening a missing partition raises (no silent empty result)."""
    store = zc.MemoryStore()
    with pytest.raises(FileNotFoundError):
        _run(open_partition_dataset_async(store, "nope", schema))


def test_async_concurrent_writes_to_distinct_partitions(schema, dataset):
    """Two writers to disjoint partitions both succeed under ``asyncio.gather``."""
    store = zc.MemoryStore()

    async def _scenario():
        await asyncio.gather(
            write_partition_dataset_async(store, "p0", dataset),
            write_partition_dataset_async(store, "p1", dataset),
        )

    _run(_scenario())
    assert _run(partition_exists_async(store, "p0")) is True
    assert _run(partition_exists_async(store, "p1")) is True
    out0 = open_partition_dataset(store, "p0", schema)
    out1 = open_partition_dataset(store, "p1", schema)
    assert numpy.array_equal(out0["value"].to_numpy(), out1["value"].to_numpy())


def test_async_write_overwrite_replaces_partition(schema, dataset):
    """``overwrite=True`` (default) replaces an existing partition's contents."""
    store = zc.MemoryStore()
    _run(write_partition_dataset_async(store, "p0", dataset))

    # Build a dataset with different values along the same shape.
    new_value = numpy.full_like(dataset["value"].to_numpy(), 42.0)
    replacement = zc.Dataset(
        schema=schema,
        variables={
            "num": dataset["num"],
            "value": zc.Variable(schema.variables["value"], new_value),
            "static": dataset["static"],
        },
    )
    _run(write_partition_dataset_async(store, "p0", replacement))

    out = open_partition_dataset(store, "p0", schema)
    assert numpy.array_equal(out["value"].to_numpy(), new_value)
