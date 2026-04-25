"""
Indexing a Collection
=====================

A secondary index lets you find which partitions and row-slices satisfy a
key-based query without scanning the whole collection. v3 indices are a
single Parquet table with ``(<key cols...>, _partition, _start, _stop)``
rows, built by walking the collection with
:py:meth:`Indexer.build<zcollection.indexing.Indexer.build>`.

Run with::

    python examples/ex_indexing.py
"""

from collections.abc import Iterator
import itertools
from pathlib import Path
import shutil
import tempfile

import numpy

import zcollection as zc
from zcollection.indexing import Indexer

# %%
# Build a half-orbit dataset
# --------------------------
#
# Each row carries ``cycle_number`` and ``pass_number``. A "half-orbit" is a
# contiguous run of rows sharing the same (cycle, pass) pair.
root = Path(tempfile.gettempdir()) / "zc-ex-indexing"
if root.exists():
    shutil.rmtree(root)
base_path = root / "collection"
index_path = root / "index.parquet"

n_cycles, n_passes, rows_per_pass = 5, 20, 10
total = n_cycles * n_passes * rows_per_pass

cycles = numpy.repeat(
    numpy.arange(n_cycles, dtype="uint16"), n_passes * rows_per_pass
)
passes = numpy.tile(
    numpy.repeat(numpy.arange(n_passes, dtype="uint16"), rows_per_pass),
    n_cycles,
)
times = numpy.arange(total, dtype="int64")

schema = (
    zc.Schema()
    .with_dimension("time", chunks=rows_per_pass * n_passes)
    .with_variable("time", dtype="int64", dimensions=("time",))
    .with_variable("cycle_number", dtype="uint16", dimensions=("time",))
    .with_variable("pass_number", dtype="uint16", dimensions=("time",))
    .build()
)
ds = zc.Dataset(
    schema=schema,
    variables={
        "time": zc.Variable(schema.variables["time"], times),
        "cycle_number": zc.Variable(schema.variables["cycle_number"], cycles),
        "pass_number": zc.Variable(schema.variables["pass_number"], passes),
    },
)

# Partition the data by cycle so each cycle is one partition.
collection = zc.create_collection(
    f"file://{base_path}",
    schema=schema,
    axis="time",
    partitioning=zc.partitioning.Sequence(("cycle_number",), dimension="time"),
)
collection.insert(ds)
print(f"collection: {len(list(collection.partitions()))} partitions")


# %%
# Build the index
# ---------------
#
# The builder takes one partition's :class:`~zcollection.Dataset` and
# returns a structured numpy array with the key columns plus integer
# ``_start`` / ``_stop`` columns delineating each contiguous run. The
# Indexer concatenates those rows over every partition.
def split_runs(values: numpy.ndarray) -> Iterator[tuple[int, int]]:
    """Yield (start, stop) for each contiguous run of identical values."""
    if values.size == 0:
        return
    edges = numpy.concatenate(
        [[0], numpy.where(numpy.diff(values) != 0)[0] + 1, [values.size]]
    )
    yield from itertools.pairwise(edges.tolist())


def half_orbit_rows(ds: zc.Dataset) -> numpy.ndarray:
    """Return one row per (cycle, pass, half-orbit) group."""
    cycle = ds["cycle_number"].to_numpy()
    pass_ = ds["pass_number"].to_numpy()
    # Combine cycle+pass into one composite key to find run boundaries.
    composite = (cycle.astype(numpy.int64) << 16) | pass_.astype(numpy.int64)
    rows = [
        (start, stop, int(cycle[start]), int(pass_[start]))
        for start, stop in split_runs(composite)
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


indexer = Indexer.build(collection, builder=half_orbit_rows)
print(f"index rows: {len(indexer)}, columns: {indexer.key_columns}")
indexer.write(str(index_path))


# %%
# Query the index
# ---------------
#
# :py:meth:`Indexer.lookup<zcollection.indexing.Indexer.lookup>` accepts a
# scalar (equality) or a list (set membership). It returns
# ``{partition_path: [(start, stop), ...]}``, ready for slicing reads.
ranges = indexer.lookup(pass_number=[1, 2])
for path, slices in ranges.items():
    print(f" * {path}: {len(slices)} matching ranges")

# %%
# Round-trip the index from disk
# ------------------------------
reloaded = Indexer.read(str(index_path))
assert len(reloaded) == len(indexer)
print("indexer round-trip: OK")
