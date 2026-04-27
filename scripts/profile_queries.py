# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Profile the query paths.

Builds a synthetic collection and runs every supported ``query`` flavour
under :mod:`pyinstrument` to spot hot functions on the read path. Each
scenario also reports the underlying Zarr-store call counts via the
existing :class:`zcollection.benches.probe.CountingProbe`, so you can
correlate "where the time is spent" with "what was actually fetched".

The output is dumped as text in the terminal and as standalone HTML
files under ``--out`` (default: ``./profiles/``) — open them in a
browser for an interactive call tree.

Run with::

    python scripts/profile_queries.py --out /tmp/zc-profiles --partitions 12

This script doubles as a regression-detection tool: re-run after a
change, diff the HTML trees, and check the shifted hot spots match the
intent of the change.
"""

from typing import Any
import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
import shutil
import sys
import tempfile

import numpy
from pyinstrument import Profiler

import zcollection as zc
from zcollection.benches.probe import CountingProbe
from zcollection.partitioning import Date


_LOGGER = logging.getLogger("profile_queries")


# %%
# Building a representative collection
# ------------------------------------


@dataclass
class Spec:
    """Synthetic-workload knobs."""

    n_partitions: int = 12
    rows_per_partition: int = 50_000
    width: int = 240  # x_ac swath, typical altimetry geometry
    profile: str = "cloud-balanced"
    seed: int = 0


def build_collection(url: str, spec: Spec) -> zc.Collection:
    """Build a fresh ``Date``-partitioned collection at ``url``.

    The schema mirrors a realistic altimetry product: a 1-D ``time``
    axis at 1 Hz plus a 2-D ``ssh(time, x_ac)``. The data is random but
    the layout matches what a query path would actually hit.
    """
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=4096)
        .with_dimension("x_ac", size=spec.width, chunks=spec.width)
        .with_variable(
            "time",
            dtype="datetime64[s]",
            dimensions=("time",),
            codecs=zc.codecs.profile(spec.profile),
        )
        .with_variable(
            "ssh",
            dtype="float32",
            dimensions=("time", "x_ac"),
            codecs=zc.codecs.profile(spec.profile),
        )
        .build()
    )
    rng = numpy.random.default_rng(spec.seed)
    total = spec.n_partitions * spec.rows_per_partition
    start = numpy.datetime64("2024-01-01T00:00:00", "s")
    times = start + numpy.arange(total) * numpy.timedelta64(60, "s")
    ssh = rng.standard_normal((total, spec.width), dtype="float32")
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(schema.variables["time"], times),
            "ssh": zc.Variable(schema.variables["ssh"], ssh),
        },
    )
    col = zc.create_collection(
        url,
        schema=schema,
        axis="time",
        partitioning=Date("time", resolution="M"),
        catalog_enabled=True,
        overwrite=True,
    )
    col.insert(ds)
    return col


def _wrap_with_probe(store: Any) -> CountingProbe:
    """Replace ``store``'s underlying Zarr store with a counting probe.

    The probe tracks ``get``/``set``/``list_dir``/``exists`` calls. We
    install it after the collection is created so the probe only sees
    query-path traffic.
    """
    probe = CountingProbe(store.zarr_store())
    store._store = probe
    return probe


# %%
# Profiling helpers
# -----------------
#
# Each scenario re-opens the collection from a fresh store handle, so
# the profiler captures cold-cache behaviour. Re-running the same
# scenario inside the profiler context would otherwise hide the
# config-load / catalog-walk overhead behind warm memory caches.


@dataclass
class Scenario:
    """One profiling scenario.

    The factory returns a *coroutine* — the profiler runs it in-process
    so :mod:`pyinstrument` can sample the actual asyncio call tree
    instead of the threadsafe wait barrier of the sync facade.
    """

    #: Short human-readable identifier used as the output filename.
    name: str
    #: One-line summary printed alongside the profile output.
    description: str
    #: Build a 0-arg coroutine factory from a fresh read-only collection.
    factory: Callable[[zc.Collection], Callable[[], Awaitable[Any]]]


@dataclass
class ScenarioResult:
    """Captured metrics for one profiling run."""

    name: str
    seconds: float
    counts: dict[str, int]
    text_report: str
    html_path: Path = field(default=Path())


def profile_scenario(
    url: str, scenario: Scenario, *, out_dir: Path, interval: float
) -> ScenarioResult:
    """Run ``scenario`` once under :mod:`pyinstrument` and record results.

    The work runs on the calling thread's event loop so the profiler's
    ``async_mode="enabled"`` can attribute each sample to the right
    coroutine instead of the runner's threadsafe wait barrier.
    """
    # Fresh read-only handle so the profiler sees cold-cache traffic.
    store = zc.open_store(url, read_only=True)
    probe = _wrap_with_probe(store)
    col = zc.open_collection(store, mode="r")
    coro_factory = scenario.factory(col)

    profiler = Profiler(async_mode="enabled", interval=interval)
    probe.reset()

    async def _run() -> None:
        await coro_factory()

    profiler.start()
    try:
        asyncio.run(_run())
    finally:
        profiler.stop()

    text_report = profiler.output_text(
        unicode=True, color=False, show_all=False
    )
    html_path = out_dir / f"{scenario.name}.html"
    html_path.write_text(profiler.output_html())

    elapsed = profiler.last_session.duration if profiler.last_session else 0.0
    return ScenarioResult(
        name=scenario.name,
        seconds=elapsed,
        counts=dict(probe.counts),
        text_report=text_report,
        html_path=html_path,
    )


# %%
# Scenarios — every public ``query`` flavour
# ------------------------------------------


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="query_full",
        description="full-collection query (every partition, every variable)",
        factory=lambda col: col.query_async,
    ),
    Scenario(
        name="query_one_partition",
        description="filter that selects exactly one partition",
        factory=lambda col: (
            lambda: col.query_async(filters="year == 2024 and month == 1")
        ),
    ),
    Scenario(
        name="query_variable_subset",
        description="full query but only the ``time`` variable",
        factory=lambda col: lambda: col.query_async(variables=("time",)),
    ),
    Scenario(
        name="query_first_three_partitions",
        description="filter selecting three months in 2024",
        factory=lambda col: (
            lambda: col.query_async(
                filters="year == 2024 and month in (1, 2, 3)"
            )
        ),
    ),
)


# %%
# CLI driver
# ----------


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point — build a collection, profile every scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "zc-profiles",
        help="Directory for HTML profile reports.",
    )
    parser.add_argument(
        "--store",
        default=None,
        help=(
            "Target store URL. Defaults to a temporary file:// store "
            "that is rebuilt on every invocation."
        ),
    )
    parser.add_argument(
        "--partitions", type=int, default=12, help="Number of partitions."
    )
    parser.add_argument("--rows-per-partition", type=int, default=50_000)
    parser.add_argument(
        "--width", type=int, default=240, help="Size of the x_ac axis."
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the synthetic collection."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0005,
        help="Pyinstrument sampling interval in seconds (default: 0.5 ms).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show INFO logs."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    # Build the collection. Use a tmp file:// store unless the user
    # pointed us at one (e.g. icechunk:// or s3://).
    cleanup: Path | None = None
    if args.store is None:
        target = Path(tempfile.gettempdir()) / "zc-profile-target"
        if target.exists():
            shutil.rmtree(target)
        url = f"file://{target}"
        cleanup = target if not args.keep else None
    else:
        url = args.store

    spec = Spec(
        n_partitions=args.partitions,
        rows_per_partition=args.rows_per_partition,
        width=args.width,
    )
    _LOGGER.info("building collection at %s (%s)", url, spec)
    build_collection(url, spec)

    try:
        results = [
            profile_scenario(
                url, scenario, out_dir=args.out, interval=args.interval
            )
            for scenario in SCENARIOS
        ]
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)

    # Per-scenario summary header + the pyinstrument tree.
    width = max(len(r.name) for r in results)
    sys.stdout.write("\n=== summary ===\n")
    for r in results:
        counts = " ".join(f"{k}={v}" for k, v in sorted(r.counts.items()))
        sys.stdout.write(
            f"{r.name:<{width}}  {r.seconds * 1000:7.1f} ms   {counts}\n"
        )
    sys.stdout.write(f"\nHTML reports under: {args.out}\n\n")

    for r in results:
        sys.stdout.write(f"\n=== {r.name}: {r.seconds * 1000:.1f} ms ===\n")
        sys.stdout.write(r.text_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
