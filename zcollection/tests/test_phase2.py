"""Phase 2 — async API, Date / GroupedSequence partitioning, merge, map/update."""
from __future__ import annotations

import asyncio

import numpy
import pytest

import zcollection3 as zc
from zcollection3 import aio
from zcollection3.collection import merge as merge_mod
from zcollection3.dask import dask_map_async
from zcollection3.partitioning import Date, GroupedSequence


# --- async facade round-trip ----------------------------------------


def test_async_create_insert_query(tmp_path, schema, dataset, partitioning):
    store = zc.LocalStore(tmp_path / "col")

    async def _scenario():
        col = await aio.create_collection(
            store, schema=schema, axis="num",
            partitioning=partitioning, overwrite=True,
        )
        written = await col.insert_async(dataset)
        assert sorted(written) == ["num=0", "num=1", "num=2"]

        reopened = await aio.open_collection(store, mode="r")
        out = await reopened.query_async()
        return out

    out = asyncio.run(_scenario())
    assert numpy.array_equal(out["num"].to_numpy(), dataset["num"].to_numpy())


def test_async_open_invalid_mode(tmp_path):
    async def _scenario():
        with pytest.raises(ValueError, match="mode must be"):
            await aio.open_collection(f"file://{tmp_path}/x", mode="bogus")

    asyncio.run(_scenario())


# --- Date partitioning ---------------------------------------------


def _date_dataset() -> tuple[zc.DatasetSchema, zc.Dataset]:
    schema = (
        zc.Schema()
        .with_dimension("time", size=None, chunks=4)
        .with_variable("time", dtype="datetime64[s]", dimensions=("time",))
        .with_variable("v", dtype="float32", dimensions=("time",))
        .build()
    )
    times = numpy.array(
        [
            "2024-01-05", "2024-01-20",
            "2024-02-03", "2024-02-15",
            "2024-03-01",
        ],
        dtype="datetime64[s]",
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(schema.variables["time"], times),
            "v": zc.Variable(
                schema.variables["v"],
                numpy.arange(times.size, dtype="float32"),
            ),
        },
    )
    return schema, ds


def test_date_partitioning_monthly(tmp_path):
    schema, ds = _date_dataset()
    store = zc.LocalStore(tmp_path / "col")
    part = Date("time", resolution="M")
    col = zc.create_collection(
        store, schema=schema, axis="time",
        partitioning=part, overwrite=True,
    )
    written = col.insert(ds)
    assert sorted(written) == [
        "year=2024/month=01",
        "year=2024/month=02",
        "year=2024/month=03",
    ]

    out = col.query(filters="year == 2024 and month == 2")
    assert out["v"].to_numpy().tolist() == [2.0, 3.0]


def test_date_partitioning_roundtrip_serde(tmp_path):
    schema, ds = _date_dataset()
    store = zc.LocalStore(tmp_path / "col")
    part = Date("time", resolution="D")
    zc.create_collection(
        store, schema=schema, axis="time",
        partitioning=part, overwrite=True,
    ).insert(ds)

    reopened = zc.open_collection(store, mode="r")
    assert isinstance(reopened.partitioning, Date)
    assert reopened.partitioning.resolution == "D"


# --- GroupedSequence -----------------------------------------------


def test_grouped_sequence_buckets_last_axis(tmp_path):
    schema = (
        zc.Schema()
        .with_dimension("num", size=None, chunks=4)
        .with_variable("cycle", dtype="int64", dimensions=("num",))
        .with_variable("pass_id", dtype="int64", dimensions=("num",))
        .with_variable("v", dtype="float32", dimensions=("num",))
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "cycle": zc.Variable(
                schema.variables["cycle"],
                numpy.array([1, 1, 1, 2, 2], dtype="int64"),
            ),
            "pass_id": zc.Variable(
                schema.variables["pass_id"],
                numpy.array([1, 7, 12, 3, 25], dtype="int64"),
            ),
            "v": zc.Variable(
                schema.variables["v"],
                numpy.arange(5, dtype="float32"),
            ),
        },
    )
    store = zc.LocalStore(tmp_path / "col")
    part = GroupedSequence(("cycle", "pass_id"), dimension="num", size=10)
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=part, overwrite=True,
    )
    written = col.insert(ds)
    # cycle=1: pass_id 1,7 → bucket 0; pass_id 12 → bucket 10
    # cycle=2: pass_id 3 → bucket 0; pass_id 25 → bucket 20
    assert sorted(written) == [
        "cycle=1/pass_id=0",
        "cycle=1/pass_id=10",
        "cycle=2/pass_id=0",
        "cycle=2/pass_id=20",
    ]


def test_grouped_sequence_size_validation():
    with pytest.raises(zc.errors.PartitionError):
        GroupedSequence(("a",), dimension="num", size=1)


# --- merge strategies ----------------------------------------------


def test_merge_replace_default(tmp_path, schema, dataset, partitioning):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    # Re-insert the same partition with new values; default = replace.
    new_value = numpy.full((2, 3), 99.0, dtype="float32")
    overwrite_ds = zc.Dataset(
        schema=schema,
        variables={
            "num": zc.Variable(
                schema.variables["num"],
                numpy.array([1, 1], dtype="int64"),
            ),
            "value": zc.Variable(schema.variables["value"], new_value),
            "static": zc.Variable(
                schema.variables["static"],
                numpy.array([10.0, 20.0, 30.0], dtype="float32"),
            ),
        },
    )
    col.insert(overwrite_ds)

    out = col.query(filters="num == 1")
    assert numpy.array_equal(out["value"].to_numpy(), new_value)


def test_merge_concat_appends(tmp_path, schema, dataset, partitioning):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    extra = zc.Dataset(
        schema=schema,
        variables={
            "num": zc.Variable(
                schema.variables["num"],
                numpy.array([1], dtype="int64"),
            ),
            "value": zc.Variable(
                schema.variables["value"],
                numpy.full((1, 3), 7.0, dtype="float32"),
            ),
            "static": zc.Variable(
                schema.variables["static"],
                numpy.array([10.0, 20.0, 30.0], dtype="float32"),
            ),
        },
    )
    col.insert(extra, merge="concat")

    out = col.query(filters="num == 1")
    # Original had 2 rows for num==1; concat adds 1 → total 3.
    assert out["value"].to_numpy().shape[0] == 3


def test_merge_time_series_drops_overlap_and_sorts(tmp_path):
    schema, ds = _date_dataset()
    store = zc.LocalStore(tmp_path / "col")
    part = Date("time", resolution="Y")
    col = zc.create_collection(
        store, schema=schema, axis="time",
        partitioning=part, overwrite=True,
    )
    col.insert(ds)

    # Insert a slice that overlaps Feb and adds an out-of-order Apr.
    new_times = numpy.array(
        ["2024-04-10", "2024-02-10"],
        dtype="datetime64[s]",
    )
    update = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(schema.variables["time"], new_times),
            "v": zc.Variable(
                schema.variables["v"],
                numpy.array([99.0, 42.0], dtype="float32"),
            ),
        },
    )
    col.insert(update, merge="time_series")

    out = col.query()
    times = out["time"].to_numpy()
    values = out["v"].to_numpy()
    # Sorted by time after merge.
    assert (numpy.diff(times) >= numpy.timedelta64(0, "s")).all()
    # Feb-15 was inside [2024-02-10, 2024-04-10] → dropped.
    # Feb-3 is below the inserted min → kept; new Feb-10 added.
    assert numpy.datetime64("2024-02-15") not in times
    assert numpy.datetime64("2024-02-03") in times
    assert numpy.datetime64("2024-02-10") in times
    # April row carries the new value.
    apr_idx = numpy.where(times == numpy.datetime64("2024-04-10"))[0]
    assert apr_idx.size == 1
    assert values[apr_idx[0]] == 99.0


def test_merge_resolve_unknown_strategy():
    with pytest.raises(KeyError):
        merge_mod.resolve("nope")


def test_merge_resolve_callable_passes_through(schema, dataset):
    def custom(existing, inserted, *, axis, partitioning_dim):
        return inserted

    assert merge_mod.resolve(custom) is custom


# --- map / update ---------------------------------------------------


def test_map_returns_per_partition_results(tmp_path, schema, dataset, partitioning):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    sizes = col.map(lambda ds: int(ds["num"].to_numpy().size))
    assert sizes == {"num=0": 2, "num=1": 2, "num=2": 3}


def test_update_writes_back(tmp_path, schema, dataset, partitioning):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )
    col.insert(dataset)

    def double_value(ds: zc.Dataset) -> zc.Dataset:
        new_vars = dict(ds.variables)
        new_vars["value"] = zc.Variable(
            ds["value"].schema, ds["value"].to_numpy() * 2.0,
        )
        return zc.Dataset(schema=ds.schema, variables=new_vars, attrs=ds.attrs)

    touched = col.update(double_value)
    assert sorted(touched) == ["num=0", "num=1", "num=2"]

    out = col.query()
    assert numpy.array_equal(
        out["value"].to_numpy(),
        dataset["value"].to_numpy() * 2.0,
    )


# --- single-writer-per-partition concurrency -----------------------


def test_concurrent_inserts_distinct_partitions(tmp_path, schema, partitioning):
    """Disjoint partitions written concurrently — no corruption, all visible."""
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store, schema=schema, axis="num",
        partitioning=partitioning, overwrite=True,
    )

    def _slice_for(num: int) -> zc.Dataset:
        return zc.Dataset(
            schema=schema,
            variables={
                "num": zc.Variable(
                    schema.variables["num"],
                    numpy.array([num, num], dtype="int64"),
                ),
                "value": zc.Variable(
                    schema.variables["value"],
                    numpy.full((2, 3), float(num), dtype="float32"),
                ),
                "static": zc.Variable(
                    schema.variables["static"],
                    numpy.array([10.0, 20.0, 30.0], dtype="float32"),
                ),
            },
        )

    async def _scenario():
        await asyncio.gather(*[
            col.insert_async(_slice_for(n)) for n in range(5)
        ])

    asyncio.run(_scenario())

    parts = list(col.partitions())
    assert sorted(parts) == [f"num={n}" for n in range(5)]
    out = col.query()
    # Each partition's value column equals its num.
    nums = out["num"].to_numpy()
    vals = out["value"].to_numpy()
    for i, n in enumerate(nums):
        assert (vals[i] == float(n)).all()


def test_dask_map_async_without_cluster():
    """``dask_map_async`` falls back to the local runner when no Client exists."""
    async def _coro_factory(i):
        return i * i

    factories = [(lambda i=i: _coro_factory(i)) for i in range(4)]
    results = dask_map_async(factories, concurrency=2)
    assert results == [0, 1, 4, 9]
