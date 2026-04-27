"""
Convert NetCDF granules to a ZCollection
========================================

A pedagogic, end-to-end script that ingests one or more NetCDF granules
(here: SWOT nadir altimetry products) into a partitioned
:py:class:`~zcollection.Collection` and maintains a half-orbit Parquet
:py:class:`~zcollection.indexing.Indexer` on top of it.

What a ZCollection partitions — and what it doesn't
---------------------------------------------------

A :class:`~zcollection.Collection` partitions a dataset along **exactly
one** unbounded axis. Every variable in the schema must be one of:

* **Partitioned** — its dimensions include the partition axis. Its rows
  are split across partitions according to the partitioning rule.
* **Immutable** — every dimension has a fixed declared size. The
  variable is then identical in every partition; ZCollection writes it
  once at the collection root (``_immutable/``) and merges it back into
  the dataset returned by every partition open.

Anything else is rejected at schema bind time. A SWOT nadir granule
ships **two independent unbounded series** — a 1 Hz block in
``/data_01`` and a 20 Hz block in ``/data_20``. Putting both in the
same collection would mean storing per-partition data on an axis the
collection doesn't know how to slice or merge: a soundness bug. The
right answer is two collections (one per resampling rate) or, if you
don't need partitioning, a single Zarr group.

This example builds the **1 Hz collection** from each granule's
``/data_01`` group, lifted to the root of the Dataset (so its ``time``
*is* the partition axis). Build a second collection for ``/data_20`` by
re-running the script against the same granules with a different
``--output`` URL and the same flags.

What the script demonstrates
----------------------------

* **Three partitioning options** selectable from the CLI:

  - ``--key date`` (default): :py:class:`~zcollection.partitioning.Date`
    on ``time``, with a ``--resolution`` (default ``D`` for daily).
  - ``--key cycle``: :py:class:`~zcollection.partitioning.Sequence` on
    ``cycle_number``.
  - ``--key cycle_pass``:
    :py:class:`~zcollection.partitioning.Sequence` on
    ``(cycle_number, pass_number)``.

* **Open-or-create**: the target ZCollection is created from the first
  granule's schema if it does not exist yet, otherwise opened in
  read-write mode.
* **Any storage backend**: the output URL is dispatched by scheme
  (``file://``, ``memory://``, ``s3://``, ``icechunk://``).
* **Half-orbit indexing**: after every insert run a Parquet index keyed
  on ``(cycle_number, pass_number)`` is rebuilt, so ``Indexer.lookup``
  returns row ranges without scanning the data files.

Merge strategies
----------------

When a new granule lands in a partition that **already** has data on
disk, ZCollection needs a rule to combine the existing rows with the
incoming ones. The strategy is picked through the ``--merge`` flag:

* ``replace`` (default) — the existing partition is overwritten by the
  inserted dataset. Use when the new granule is the source of truth and
  you don't care about the previous content.
* ``concat`` — the inserted rows are simply appended after the existing
  rows along the partition axis. Cheapest, no deduplication, no sorting.
* ``time_series`` — drops every existing row whose ``time`` falls inside
  ``[inserted.time.min(), inserted.time.max()]``, then concatenates and
  sorts along ``time``. Right when a re-acquisition fully covers a
  previous one in time.
* ``upsert`` — row-wise replace-or-add by exact ``time`` equality.
  Existing rows whose time appears in the new batch are dropped;
  everything else is kept; the result is sorted by time.
* ``upsert_within`` — same as ``upsert`` but matches existing rows
  against the **nearest** inserted timestamp within ``--tolerance``.
  Useful when re-acquired timestamps are jittered by clock drift. The
  tolerance is parsed by :func:`numpy.timedelta64` (e.g. ``500ms``,
  ``1s``, ``2us``).

Run with::

    python examples/ex_netcdf_to_zcollection.py \\
        --output file:///tmp/swot-nadir-1hz \\
        --key date --resolution D \\
        --merge upsert_within --tolerance 500ms \\
        /path/to/SWOT_GPS_2PsP*.nc

A SWOT nadir granule looks like (one cycle/pass per file)::

    SWOT_GPS_2PsP031_100_20250410_165925_20250410_175052.nc
                  ^cycle ^pass ^start (UTC)

This example uses the ``netCDF4`` library to access nested groups,
which xarray cannot do natively without an explicit ``group=`` argument.
"""

from typing import TYPE_CHECKING, Any
import argparse
from datetime import datetime
import itertools
import logging
from pathlib import Path

import netCDF4
import numpy

import zcollection as zc
from zcollection.collection import merge as merge_strategies
from zcollection.errors import CollectionNotFoundError
from zcollection.indexing import Indexer


if TYPE_CHECKING:
    from collections.abc import Iterator

_LOGGER = logging.getLogger("ex_netcdf_to_zcollection")


# %%
# Reading a granule
# -----------------
#
# We open each SWOT NetCDF file with :mod:`netCDF4` (xarray cannot read
# named subgroups without an explicit ``group=``) and extract the 1 Hz
# variables from ``/data_01``. Cycle/pass numbers come from the global
# attributes and are broadcast over the time axis as filler columns.

#: The partition axis. SWOT 1 Hz time stamps live here, so ``time`` is
#: both the variable name and the partitioning dim.
ROOT_TIME_DIM: str = "time"
#: Reference epoch for SWOT time variables (``seconds since ...``).
EPOCH: numpy.datetime64 = numpy.datetime64("2000-01-01T00:00:00", "ns")

#: 1 Hz variables we lift from ``/data_01`` to the collection root.
#: Extending the list is a one-line edit; the schema is inferred from
#: whatever the first granule exposes.
DATA_01_VARS: tuple[str, ...] = (
    "time",
    "latitude",
    "longitude",
    "altitude",
    "distance_to_coast",
    "surface_classification_flag",
)


def _decode_time(seconds: numpy.ndarray) -> numpy.ndarray:
    """Convert SWOT ``seconds since 2000-01-01`` to ``datetime64[ns]``."""
    nanos = (seconds * 1e9).astype("int64")
    return EPOCH + nanos.astype("timedelta64[ns]")


def _parse_meas_time(text: str) -> numpy.datetime64:
    """Parse a SWOT ``YYYY-MM-DD HH:MM:SS.ffffff`` global attr."""
    return numpy.datetime64(datetime.fromisoformat(text), "ns")


def read_granule(
    path: Path,
) -> tuple[dict[str, Any], dict[str, numpy.ndarray]]:
    """Open one SWOT NetCDF file and return ``(meta, data_01)``.

    ``meta`` holds the global attributes (cycle/pass numbers, granule
    name); ``data_01`` is a per-variable dict of 1 Hz arrays.
    """
    with netCDF4.Dataset(path) as nc:
        meta = {
            "cycle_number": int(nc.cycle_number),
            "pass_number": int(nc.pass_number),
            "first_meas_time": _parse_meas_time(nc.first_meas_time),
            "granule": path.name,
        }
        data_01 = {
            name: numpy.asarray(nc["data_01"][name][:])
            for name in DATA_01_VARS
            if name in nc["data_01"].variables
        }

    # Decode SWOT 'seconds since 2000-01-01' into datetime64[ns] so the
    # Date partitioner can bucket on it directly.
    data_01["time"] = _decode_time(data_01["time"])
    return meta, data_01


# %%
# Building the schema
# -------------------
#
# A ZCollection schema is *immutable* and *declarative*. We discover it
# once, from the first granule, then reuse it for every subsequent
# insert. ``cycle_number`` and ``pass_number`` are 1 Hz filler columns
# (broadcast from the per-file scalars) so the Sequence partitioners
# and the half-orbit indexer can key on them.


def build_schema(sample_data_01: dict[str, numpy.ndarray]) -> zc.DatasetSchema:
    """Return a :class:`~zcollection.DatasetSchema` matching the granule layout."""
    builder = (
        zc.Schema()
        # 86400 ≈ one day at 1 Hz. Tune to match your typical partition
        # footprint; compression and read efficiency both depend on it.
        .with_dimension(ROOT_TIME_DIM, chunks=86400)
        .with_variable(
            "time", dtype="datetime64[ns]", dimensions=(ROOT_TIME_DIM,)
        )
        .with_variable(
            "cycle_number", dtype="uint16", dimensions=(ROOT_TIME_DIM,)
        )
        .with_variable(
            "pass_number", dtype="uint16", dimensions=(ROOT_TIME_DIM,)
        )
    )
    for name, arr in sample_data_01.items():
        if name == "time":
            continue  # already declared
        builder = builder.with_variable(
            name, dtype=arr.dtype, dimensions=(ROOT_TIME_DIM,)
        )
    return builder.build()


# %%
# Building the in-memory Dataset for one granule
# ----------------------------------------------
#
# For each input granule we instantiate one :class:`~zcollection.Dataset`
# matching the schema. ``cycle_number`` and ``pass_number`` are
# broadcast over the granule's 1 Hz time axis.


def make_dataset(
    schema: zc.DatasetSchema,
    meta: dict[str, Any],
    data_01: dict[str, numpy.ndarray],
) -> zc.Dataset:
    """Bind raw arrays to the schema, returning one :class:`~zcollection.Dataset`."""
    n = data_01["time"].size
    cycle = numpy.full(n, meta["cycle_number"], dtype="uint16")
    pass_ = numpy.full(n, meta["pass_number"], dtype="uint16")

    variables: dict[str, zc.Variable] = {
        "time": zc.Variable(schema.variables["time"], data_01["time"]),
        "cycle_number": zc.Variable(schema.variables["cycle_number"], cycle),
        "pass_number": zc.Variable(schema.variables["pass_number"], pass_),
    }
    for name, arr in data_01.items():
        if name == "time":
            continue
        variables[name] = zc.Variable(schema.variables[name], arr)

    return zc.Dataset(schema=schema, variables=variables)


# %%
# Picking the merge strategy
# --------------------------
#
# A merge strategy decides what happens when an insert lands in a
# partition that already has data. The functions live in
# :mod:`zcollection.collection.merge` and are also exposed as string
# aliases (``"replace"``, ``"concat"``, ``"time_series"``, ``"upsert"``).
# For ``upsert_within`` we build a closure with the tolerance baked in
# via :func:`~zcollection.collection.merge.upsert_within`.

_TIMEDELTA_UNITS: dict[str, str] = {
    "ns": "ns",
    "us": "us",
    "µs": "us",
    "ms": "ms",
    "s": "s",
    "m": "m",
    "h": "h",
    "D": "D",
}


def _parse_tolerance(text: str) -> numpy.timedelta64:
    """Parse a string like ``"500ms"`` or ``"1s"`` into a ``timedelta64``."""
    text = text.strip()
    for suffix, unit in _TIMEDELTA_UNITS.items():
        if text.endswith(suffix):
            value = text[: -len(suffix)].strip() or "1"
            try:
                return numpy.timedelta64(int(value), unit)
            except ValueError as exc:  # non-integer → reject early
                raise argparse.ArgumentTypeError(
                    f"tolerance value must be an integer; got {value!r}"
                ) from exc
    raise argparse.ArgumentTypeError(
        f"tolerance {text!r} must end with one of "
        f"{tuple(_TIMEDELTA_UNITS)} (e.g. '500ms', '1s', '20us')."
    )


def make_merge(
    name: str,
    tolerance: numpy.timedelta64 | None,
) -> str | merge_strategies.MergeCallable:
    """Resolve the CLI ``--merge`` flag into something ``Collection.insert`` accepts."""
    if name == "upsert_within":
        if tolerance is None:
            raise ValueError(
                "--merge upsert_within requires --tolerance "
                "(e.g. --tolerance 500ms)"
            )
        return merge_strategies.upsert_within(tolerance)
    if name in {"replace", "concat", "time_series", "upsert"}:
        return name
    raise ValueError(
        f"unknown --merge {name!r}; choose from replace, concat, "
        "time_series, upsert, upsert_within"
    )


# %%
# Picking the partitioning
# ------------------------


def make_partitioning(
    key: str, resolution: str
) -> zc.partitioning.Partitioning:
    """Resolve the CLI ``--key`` flag into a Partitioning instance."""
    if key == "date":
        return zc.partitioning.Date(
            ("time",), resolution=resolution, dimension=ROOT_TIME_DIM
        )
    if key == "cycle":
        return zc.partitioning.Sequence(
            ("cycle_number",), dimension=ROOT_TIME_DIM
        )
    if key == "cycle_pass":
        return zc.partitioning.Sequence(
            ("cycle_number", "pass_number"), dimension=ROOT_TIME_DIM
        )
    raise ValueError(
        f"unknown --key {key!r}; choose from 'date', 'cycle', 'cycle_pass'"
    )


# %%
# Open existing collection or create it from the first granule
# ------------------------------------------------------------


def open_or_create(
    output_url: str,
    sample_data_01: dict[str, numpy.ndarray],
    *,
    key: str,
    resolution: str,
) -> zc.Collection:
    """Open the target :class:`~zcollection.Collection` or create it.

    On first call the schema is inferred from the first granule and the
    partitioning is materialised on disk; subsequent calls open the
    existing root.
    """
    try:
        col = zc.open_collection(output_url, mode="rw")
        _LOGGER.info("opened existing collection at %s", output_url)
        return col
    except CollectionNotFoundError:
        schema = build_schema(sample_data_01)
        partitioning = make_partitioning(key, resolution)
        _LOGGER.info(
            "creating new collection at %s (key=%s, resolution=%s)",
            output_url,
            key,
            resolution,
        )
        return zc.create_collection(
            output_url,
            schema=schema,
            axis=ROOT_TIME_DIM,
            partitioning=partitioning,
            catalog_enabled=True,
        )


# %%
# Half-orbit indexer
# ------------------
#
# A SWOT half-orbit is a contiguous run of 1 Hz rows sharing the same
# ``(cycle_number, pass_number)`` pair. The Indexer keeps one Parquet
# row per (partition, run), letting ``Indexer.lookup(...)`` return slice
# ranges without touching the data files.


def _split_runs(values: numpy.ndarray) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` for every contiguous run of identical values."""
    if values.size == 0:
        return
    edges = numpy.concatenate(
        [[0], numpy.where(numpy.diff(values) != 0)[0] + 1, [values.size]]
    )
    yield from itertools.pairwise(edges.tolist())


def half_orbit_rows(ds: zc.Dataset) -> numpy.ndarray:
    """Build one structured row per (cycle, pass) run for the indexer."""
    cycle = ds["cycle_number"].to_numpy()
    pass_ = ds["pass_number"].to_numpy()
    composite = (cycle.astype("int64") << 16) | pass_.astype("int64")
    rows = [
        (start, stop, int(cycle[start]), int(pass_[start]))
        for start, stop in _split_runs(composite)
    ]
    return numpy.array(
        rows,
        dtype=[
            ("_start", "int64"),
            ("_stop", "int64"),
            ("cycle_number", "uint16"),
            ("pass_number", "uint16"),
        ],
    )


def rebuild_index(col: zc.Collection, index_path: str) -> Indexer:
    """Rebuild the half-orbit index over the whole collection."""
    indexer = Indexer.build(col, builder=half_orbit_rows)
    indexer.write(index_path)
    _LOGGER.info("wrote %d index rows to %s", len(indexer), index_path)
    return indexer


# %%
# CLI
# ---


def main(argv: list[str] | None = None) -> int:
    """Entry point — wires everything together.

    When invoked without arguments — the case sphinx-gallery hits
    when it executes this file as part of the documentation build —
    this function prints a usage hint and returns ``0`` instead of
    calling ``argparse``. That keeps the example renderable in the
    gallery without needing real NetCDF granules in the build
    environment.
    """
    import sys

    if argv is None and len(sys.argv) <= 1:
        sys.stdout.write(
            "ex_netcdf_to_zcollection: this example needs NetCDF granules\n"
            "and a target URL. Run it from the command line, e.g.\n"
            "    python examples/ex_netcdf_to_zcollection.py \\\n"
            "        --output file:///tmp/swot-collection \\\n"
            "        path/to/SWOT_GPS_2PsP*.nc\n"
            "See the module docstring at the top of the file for the\n"
            "full set of options (--key / --resolution / --merge / "
            "--tolerance / --index).\n"
        )
        return 0

    parser = argparse.ArgumentParser(
        description="Ingest SWOT-style NetCDF granules into a ZCollection.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more input NetCDF granules.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Target ZCollection URL "
            "(file://, memory://, s3://, icechunk://)."
        ),
    )
    parser.add_argument(
        "--key",
        choices=("date", "cycle", "cycle_pass"),
        default="date",
        help="Partitioning key. Default: date.",
    )
    parser.add_argument(
        "--resolution",
        default="D",
        help=(
            "Date resolution when --key=date "
            "(Y, M, D, h, m, s). Default: D."
        ),
    )
    parser.add_argument(
        "--merge",
        choices=(
            "replace",
            "concat",
            "time_series",
            "upsert",
            "upsert_within",
        ),
        default="replace",
        help=(
            "Merge strategy when an insert lands in an existing "
            "partition. Default: replace. See module docstring for the "
            "semantics of each option."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=_parse_tolerance,
        default=None,
        help=(
            "Time tolerance for --merge upsert_within "
            "(e.g. '500ms', '1s', '20us'). Required by that strategy "
            "and ignored otherwise."
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help=(
            "Path to the half-orbit Parquet index. "
            "Defaults to '<output>.index.parquet' for file:// outputs."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show INFO logs."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    # First granule seeds the schema (if the collection doesn't exist yet).
    first = args.inputs[0]
    _LOGGER.info("reading sample granule %s", first)
    _, sample_01 = read_granule(first)

    col = open_or_create(
        args.output,
        sample_01,
        key=args.key,
        resolution=args.resolution,
    )

    merge = make_merge(args.merge, args.tolerance)
    _LOGGER.info(
        "merge strategy: %s%s",
        args.merge,
        f" (tolerance={args.tolerance})" if args.tolerance is not None else "",
    )

    for path in args.inputs:
        _LOGGER.info("ingesting %s", path)
        meta, d01 = read_granule(path)
        ds = make_dataset(col.schema, meta, d01)
        col.insert(ds, merge=merge)
    _LOGGER.info(
        "collection now has %d partitions",
        len(list(col.partitions())),
    )

    # Default index path next to the collection on local disk.
    index_path = args.index
    if index_path is None and args.output.startswith("file://"):
        index_path = (
            Path(args.output.removeprefix("file://"))
            .resolve()
            .with_suffix(".index.parquet")
        )
    if index_path is not None:
        rebuild_index(col, str(index_path))
    else:
        _LOGGER.info(
            "skipping index build (no --index path given, "
            "and --output is not file://)"
        )

    return 0


if __name__ == "__main__":
    # Don't ``raise SystemExit(main())`` here: when sphinx-gallery
    # executes this file as part of the docs build it treats any
    # SystemExit (even ``SystemExit(0)``) as an example failure and
    # aborts the build. ``main()`` already raises on every error
    # path, so a plain call is enough — Python's default exit code
    # of 0 covers the success case.
    main()
