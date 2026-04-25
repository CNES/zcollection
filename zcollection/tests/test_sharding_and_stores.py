"""Phase 3 — sharding policy, sharded round-trip, store factory, bench harness."""

import json

import numpy
import pytest

import zcollection as zc
from zcollection.benches import BenchSpec, run_suite
from zcollection.benches.harness import dump_json
from zcollection.benches.probe import CountingProbe
from zcollection.codecs import shard_decision, shard_target_bytes
from zcollection.codecs.sharding import compute_shard_shape
from zcollection.errors import StoreError

# --- sharding shape policy ----------------------------------------


def test_compute_shard_shape_doubles_largest_dim():
    shape = compute_shard_shape(
        inner_chunks=(4096, 240),
        shape=(None, 240),
        dtype=numpy.dtype("float32"),
        target_shard_bytes=128 << 20,  # 128 MiB
    )
    # inner = 4096*240*4 = ~3.75 MiB. Target 128 MiB -> ~34x growth.
    # x_ac is already at full width (cap=1) so all growth goes into time.
    assert shape[1] == 240
    assert shape[0] >= 4096 * 16
    raw_bytes = shape[0] * shape[1] * 4
    assert raw_bytes <= 128 << 20


def test_compute_shard_shape_clips_to_dim_size():
    shape = compute_shard_shape(
        inner_chunks=(64, 64),
        shape=(64, 64),
        dtype=numpy.dtype("float64"),
        target_shard_bytes=64 << 20,
    )
    assert shape == (64, 64)


def test_shard_decision_returns_none_when_disabled():
    assert (
        shard_decision(
            inner_chunks=(1024,),
            shape=(None,),
            dtype=numpy.dtype("float32"),
            target_shard_bytes=None,
        )
        is None
    )


def test_shard_decision_returns_none_when_no_growth():
    # Inner already exceeds target → no benefit to wrapping in a shard.
    assert (
        shard_decision(
            inner_chunks=(1024 * 1024,),
            shape=(1024 * 1024,),
            dtype=numpy.dtype("float64"),
            target_shard_bytes=1 << 20,
        )
        is None
    )


def test_profile_target_shard_bytes():
    assert shard_target_bytes("local-fast") is None
    assert shard_target_bytes("cloud-balanced") == 128 << 20
    assert shard_target_bytes("cloud-cold") == 512 << 20


# --- sharded end-to-end round-trip --------------------------------


def _sharded_schema_and_dataset() -> tuple[zc.DatasetSchema, zc.Dataset]:
    schema = (
        zc.Schema()
        .with_dimension("time", size=None, chunks=128)
        .with_dimension("x_ac", size=8, chunks=8)
        .with_variable(
            "time",
            dtype="int64",
            dimensions=("time",),
            codecs=zc.codecs.profile("cloud-balanced"),
        )
        .with_variable(
            "ssh",
            dtype="float32",
            dimensions=("time", "x_ac"),
            codecs=zc.codecs.profile("cloud-balanced"),
        )
        .build()
    )
    n = 4096
    times = numpy.arange(n, dtype="int64")
    ssh = numpy.arange(n * 8, dtype="float32").reshape(n, 8)
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(schema.variables["time"], times),
            "ssh": zc.Variable(schema.variables["ssh"], ssh),
        },
    )
    return schema, ds


def test_sharded_round_trip_local(tmp_path):
    schema, ds = _sharded_schema_and_dataset()
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="time",
        partitioning=zc.partitioning.Sequence(("time",), dimension="time"),
        overwrite=True,
    )
    # Bucket all rows into one partition by collapsing the key.
    all_zero = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(
                schema.variables["time"],
                numpy.zeros(ds["time"].to_numpy().size, dtype="int64"),
            ),
            "ssh": ds["ssh"],
        },
    )
    written = col.insert(all_zero)
    assert written == ["time=0"]

    out = col.query()
    assert out["ssh"].to_numpy().shape == ds["ssh"].to_numpy().shape
    assert numpy.array_equal(out["ssh"].to_numpy(), ds["ssh"].to_numpy())


def test_sharded_array_metadata_uses_sharding_codec(tmp_path):
    """Confirm the on-disk array carries a ShardingCodec at the serializer slot."""
    import zarr

    schema, ds = _sharded_schema_and_dataset()
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="time",
        partitioning=zc.partitioning.Sequence(("time",), dimension="time"),
        overwrite=True,
    )
    col.insert(
        zc.Dataset(
            schema=schema,
            variables={
                "time": zc.Variable(
                    schema.variables["time"],
                    numpy.zeros(ds["time"].to_numpy().size, dtype="int64"),
                ),
                "ssh": ds["ssh"],
            },
        )
    )

    arr = zarr.open_array(store=store.zarr_store(), path="time=0/ssh", mode="r")
    meta = arr.metadata.to_dict()
    codec_names = [c.get("name", "") for c in meta.get("codecs", [])]
    assert "sharding_indexed" in codec_names


# --- store factory dispatch ---------------------------------------


def test_open_store_local_path(tmp_path):
    s = zc.open_store(str(tmp_path / "x"))
    assert isinstance(s, zc.LocalStore)


def test_open_store_memory():
    s = zc.open_store("memory://")
    assert isinstance(s, zc.MemoryStore)


def test_open_store_icechunk_resolves(tmp_path):
    pytest.importorskip("icechunk")
    from zcollection.store.icechunk_store import IcechunkStore

    s = zc.open_store(f"icechunk://{tmp_path / 'repo'}")
    assert isinstance(s, IcechunkStore)


def test_open_store_unknown_scheme():
    with pytest.raises(StoreError, match="unrecognised"):
        zc.open_store("ftp://example.com/data")


def test_open_store_s3_requires_obstore(monkeypatch):
    """If obstore isn't installed, the cloud path raises a clear StoreError."""
    import importlib
    import sys

    real = sys.modules.pop("obstore", None)
    monkeypatch.setitem(sys.modules, "obstore", None)
    sys.modules.pop("zcollection.store.obstore_store", None)
    try:
        with pytest.raises((StoreError, ImportError)):
            zc.open_store("s3://bucket/prefix")
    finally:
        if real is not None:
            sys.modules["obstore"] = real
        importlib.invalidate_caches()


# --- counting probe -----------------------------------------------


def test_counting_probe_counts_writes_and_reads(tmp_path):
    """Wrap the local Zarr store in a CountingProbe and verify counters move."""
    schema, ds = _sharded_schema_and_dataset()
    store = zc.LocalStore(tmp_path / "col")
    probe = CountingProbe(store.zarr_store())

    # Swap the inner store on the wrapper (small hack — only for benches).
    store._store = probe

    col = zc.create_collection(
        store,
        schema=schema,
        axis="time",
        partitioning=zc.partitioning.Sequence(("time",), dimension="time"),
        overwrite=True,
    )
    one_part = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(
                schema.variables["time"],
                numpy.zeros(ds["time"].to_numpy().size, dtype="int64"),
            ),
            "ssh": ds["ssh"],
        },
    )
    probe.reset()
    col.insert(one_part)
    assert probe.counts["set"] > 0
    write_count = probe.counts["set"]

    probe.reset()
    out = col.query()
    assert probe.counts["get"] > 0
    assert out is not None
    # Sharded reads should issue *fewer* GETs than the per-chunk count of writes.
    # (At minimum: not catastrophically more.)
    assert probe.counts["get"] <= write_count + 32


# --- bench harness end-to-end on local store ---------------------


def test_bench_suite_runs_locally(tmp_path):
    spec = BenchSpec(
        n_partitions=2,
        rows_per_partition=128,
        width=8,
        profile="local-fast",
    )
    results = run_suite(f"file://{tmp_path}/bench", spec)
    names = {r.name for r in results}
    assert {
        "insert_full_dataset",
        "open_collection_cold",
        "query_one_partition_full",
        "query_full",
    } <= names

    out_path = tmp_path / "results.json"
    dump_json(results, out_path)
    payload = json.loads(out_path.read_text())
    assert isinstance(payload, list)
    assert payload
    assert all("seconds" in item for item in payload)
