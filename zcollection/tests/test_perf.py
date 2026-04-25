"""Performance smoke tests on a realistic workload.

Runs the bench harness (insert, cold open, partition query, full query)
against two backends:

- a local filesystem (``LocalStore``);
- a local S3 server (MinIO subprocess), which mimics the cloud layout
  without any network egress.

These tests are opt-in — they only run with ``pytest --perf``. Each test
prints a small timing/probe summary to stdout so a developer can spot
regressions at a glance, but does **not** assert hard wall-clock numbers
(those are too host-dependent for CI). Capture a baseline JSON with
``python -m zcollection.benches`` and compare locally.
"""

import pytest

from zcollection.benches.harness import BenchSpec, dump_json, run_suite

from ._s3_server import minio_server


pytestmark = pytest.mark.perf


# Workload sized to feel like one altimetry month (~1.2M rows, 240-wide
# swath). Trim if your test box is small.
PERF_SPEC = BenchSpec(
    n_partitions=4,
    rows_per_partition=300_000,
    width=240,
    profile="cloud-balanced",
)


def _summarise(results: list, label: str) -> None:
    """Print a one-line-per-phase summary."""
    print(f"\n=== {label} ===")
    for r in results:
        counts = " ".join(f"{k}={v}" for k, v in sorted(r.counts.items()))
        print(f"  {r.name:<28} {r.seconds:7.3f}s   {counts}")


def test_perf_local_filesystem(tmp_path):
    """Run the bench suite against a LocalStore on tmp_path."""
    store_url = f"file://{tmp_path / 'col'}"
    results = run_suite(store_url, PERF_SPEC)

    _summarise(results, f"local FS  {store_url}")
    dump_json(results, tmp_path / "perf-local.json")

    by_name = {r.name: r for r in results}
    # Sanity: insert and queries should have done real I/O.
    assert by_name["insert_full_dataset"].seconds > 0
    assert by_name["query_full"].seconds > 0


def test_perf_minio_s3(tmp_path, minio_bin):
    """Run the bench suite against a local MinIO server.

    Skipped unless the ``minio`` binary is on PATH (or ``MINIO_BIN`` is
    set). Requires ``obstore`` and ``boto3`` for bucket creation.
    """
    pytest.importorskip("obstore")
    pytest.importorskip("boto3")

    data_dir = tmp_path / "minio-data"
    data_dir.mkdir()
    out_path = tmp_path / "perf-s3.json"

    with minio_server(str(data_dir), binary=minio_bin) as endpoint:
        store = endpoint.object_store(prefix="bench")
        results = _run_with_store(store, PERF_SPEC)

    _summarise(results, f"MinIO     {endpoint.endpoint}")
    dump_json(results, out_path)

    by_name = {r.name: r for r in results}
    assert by_name["insert_full_dataset"].seconds > 0
    assert by_name["query_full"].seconds > 0


def _run_with_store(store, spec):
    """Bench harness variant that takes a pre-built Store object.

    The public ``run_suite`` works from a URL; for S3 we want to inject
    the MinIO-tuned client_options/credentials, so we duplicate the
    minimum amount of plumbing here.
    """
    import time

    import zcollection as zc
    from zcollection.benches.harness import (
        BenchResult,
        _build_dataset,
        _wrap_with_probe,
    )
    from zcollection.partitioning import Date

    schema, ds = _build_dataset(spec)
    col = zc.create_collection(
        store,
        schema=schema,
        axis="time",
        partitioning=Date("time", resolution="M"),
        overwrite=True,
    )
    results: list[BenchResult] = []

    insert_probe = _wrap_with_probe(col.store)
    insert_probe.reset()
    t0 = time.perf_counter()
    col.insert(ds)
    results.append(
        BenchResult(
            name="insert_full_dataset",
            seconds=time.perf_counter() - t0,
            counts=dict(insert_probe.counts),
        )
    )

    open_probe = _wrap_with_probe(store)
    open_probe.reset()
    t0 = time.perf_counter()
    col_ro = zc.open_collection(store, mode="r")
    results.append(
        BenchResult(
            name="open_collection_cold",
            seconds=time.perf_counter() - t0,
            counts=dict(open_probe.counts),
        )
    )

    query_probe = _wrap_with_probe(store)
    query_probe.reset()
    t0 = time.perf_counter()
    col_ro.query(filters="year == 2024 and month == 1")
    results.append(
        BenchResult(
            name="query_one_partition_full",
            seconds=time.perf_counter() - t0,
            counts=dict(query_probe.counts),
        )
    )

    query_probe.reset()
    t0 = time.perf_counter()
    col_ro.query()
    results.append(
        BenchResult(
            name="query_full",
            seconds=time.perf_counter() - t0,
            counts=dict(query_probe.counts),
        )
    )
    return results
