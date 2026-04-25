"""Bench harness sanity — runs the suite on a tiny local store."""

import json

from zcollection.benches.harness import BenchSpec, compare, dump_json, run_suite


def test_run_suite_against_local_store(tmp_path):
    """``run_suite`` produces the expected named bench phases with probe counts."""
    spec = BenchSpec(n_partitions=2, rows_per_partition=100, width=4)
    store_url = f"file://{tmp_path / 'col'}"
    results = run_suite(store_url, spec)

    names = [r.name for r in results]
    assert names == [
        "insert_full_dataset",
        "open_collection_cold",
        "query_one_partition_full",
        "query_full",
    ]
    by_name = {r.name: r for r in results}
    # Probe wired in: insert and the two query phases must hit the zarr layer.
    # ``open_collection_cold`` deliberately reads only the config sidecar, so
    # zero zarr ops there is the correct outcome.
    for hot in (
        "insert_full_dataset",
        "query_one_partition_full",
        "query_full",
    ):
        assert sum(by_name[hot].counts.values()) > 0, (
            f"{hot} had no probe activity"
        )


def test_dump_and_compare_roundtrip(tmp_path):
    """``dump_json``/``compare`` round-trip yields ratios near 1.0."""
    spec = BenchSpec(n_partitions=2, rows_per_partition=100, width=4)
    store_url = f"file://{tmp_path / 'col'}"
    results = run_suite(store_url, spec)

    out_path = tmp_path / "current.json"
    dump_json(results, out_path)
    payload = json.loads(out_path.read_text())
    assert {item["name"] for item in payload} == {r.name for r in results}

    # Self-compare → ratios should all be 1.0 (within float).
    ratios = compare(results, out_path)
    for name, ratio in ratios.items():
        assert 0.9 <= ratio <= 1.1, (name, ratio)
