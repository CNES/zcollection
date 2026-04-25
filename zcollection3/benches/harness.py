"""Bench harness — define a workload, run scenarios, dump JSON results."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy

import zcollection3 as zc
from zcollection3.partitioning import Date


@dataclass(frozen=True, slots=True)
class BenchSpec:
    """Workload definition. Sized small by default; tune for cloud runs."""

    n_partitions: int = 12
    rows_per_partition: int = 50_000
    width: int = 240  # x_ac dimension, typical altimetry swath
    profile: str = "cloud-balanced"
    seed: int = 0


@dataclass(slots=True)
class BenchResult:
    name: str
    seconds: float
    counts: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _build_dataset(spec: BenchSpec) -> tuple[zc.DatasetSchema, zc.Dataset]:
    schema = (
        zc.Schema()
        .with_dimension("time", size=None, chunks=4096)
        .with_dimension("x_ac", size=spec.width, chunks=spec.width)
        .with_variable(
            "time", dtype="datetime64[s]", dimensions=("time",),
            codecs=zc.codecs.profile(spec.profile),
        )
        .with_variable(
            "ssh", dtype="float32", dimensions=("time", "x_ac"),
            codecs=zc.codecs.profile(spec.profile),
        )
        .build()
    )
    rng = numpy.random.default_rng(spec.seed)
    total = spec.n_partitions * spec.rows_per_partition
    # Span n_partitions distinct months in 2024.
    start = numpy.datetime64("2024-01-01T00:00:00", "s")
    step = numpy.timedelta64(60, "s")
    times = start + numpy.arange(total) * step
    ssh = rng.standard_normal((total, spec.width), dtype="float32")
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(schema.variables["time"], times),
            "ssh": zc.Variable(schema.variables["ssh"], ssh),
        },
    )
    return schema, ds


def _make_collection(store_url: str, spec: BenchSpec) -> tuple[Any, zc.Dataset]:
    schema, ds = _build_dataset(spec)
    store = zc.open_store(store_url)
    col = zc.create_collection(
        store,
        schema=schema,
        axis="time",
        partitioning=Date("time", resolution="M"),
        overwrite=True,
    )
    return col, ds


def _timed(name: str, fn: Callable[[], Any], probe: Any | None = None) -> BenchResult:
    if probe is not None:
        probe.reset()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    counts = dict(probe.counts) if probe is not None else {}
    return BenchResult(name=name, seconds=elapsed, counts=counts)


def run_suite(store_url: str, spec: BenchSpec | None = None) -> list[BenchResult]:
    """Run the full Phase 3 acceptance suite against ``store_url``."""
    spec = spec or BenchSpec()
    results: list[BenchResult] = []

    # 1. insert_full_dataset — drives N partitions in one call
    col, ds = _make_collection(store_url, spec)
    results.append(_timed("insert_full_dataset", lambda: col.insert(ds)))

    # 2. open_collection_cold — fresh process / fresh store handle
    fresh_store = zc.open_store(store_url, read_only=True)
    results.append(_timed(
        "open_collection_cold",
        lambda: zc.open_collection(fresh_store, mode="r"),
    ))

    # 3. query_one_partition_full
    col_ro = zc.open_collection(zc.open_store(store_url, read_only=True), mode="r")
    results.append(_timed(
        "query_one_partition_full",
        lambda: col_ro.query(filters="year == 2024 and month == 1"),
    ))

    # 4. query_full
    results.append(_timed("query_full", col_ro.query))

    return results


def dump_json(results: list[BenchResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in results], indent=2))


def compare(current: list[BenchResult], baseline_path: str | Path) -> dict[str, float]:
    """Return ``{name: ratio}`` where ratio = baseline_seconds / current_seconds.

    A ratio ≥ 1 means the current run is at-least-as-fast as the baseline.
    """
    baseline = json.loads(Path(baseline_path).read_text())
    base_by_name = {item["name"]: item["seconds"] for item in baseline}
    out: dict[str, float] = {}
    for r in current:
        if r.name not in base_by_name:
            continue
        if r.seconds == 0:
            out[r.name] = float("inf")
        else:
            out[r.name] = base_by_name[r.name] / r.seconds
    return out
