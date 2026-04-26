"""
End-to-end Walkthrough
======================

Builds a float32 dataset on a ``LocalStore``, partitions it, reopens the
collection from disk, queries with a filter, and asserts bit-exact equality.

Run with::

    python examples/ex_walkthrough.py
"""

from pathlib import Path
import shutil
import tempfile

import numpy

import zcollection as zc


# %%
# Initialization
# --------------
#
# Set up a temporary directory for the collection.
target = Path(tempfile.gettempdir()) / "zc-walkthrough"
if target.exists():
    shutil.rmtree(target)

# %%
# Build a schema
# --------------
#
# Declare dimensions and variables with their dtypes and chunk sizes.
schema = (
    zc.Schema()
    .with_dimension("time", chunks=4096)
    .with_dimension("x_ac", size=240, chunks=240)
    .with_variable("time", dtype="int64", dimensions=("time",))
    .with_variable("partition", dtype="int64", dimensions=("time",))
    .with_variable(
        "ssh",
        dtype="float32",
        dimensions=("time", "x_ac"),
        fill_value=numpy.float32("nan"),
    )
    .build()
)

# %%
# Build a sample dataset
# ----------------------
#
# Create a :py:class:`~zcollection.Dataset` with synthetic data split across
# 4 partitions.
N_PARTITIONS = 4
ROWS_PER_PARTITION = 25_000
rng = numpy.random.default_rng(42)
n = N_PARTITIONS * ROWS_PER_PARTITION
time = numpy.arange(n, dtype="int64")
partition = numpy.repeat(
    numpy.arange(N_PARTITIONS, dtype="int64"), ROWS_PER_PARTITION
)
ssh = rng.standard_normal(size=(n, 240), dtype="float32")

ds = zc.Dataset(
    schema=schema,
    variables={
        "time": zc.Variable(schema.variables["time"], time),
        "partition": zc.Variable(schema.variables["partition"], partition),
        "ssh": zc.Variable(schema.variables["ssh"], ssh),
    },
)
print(f"dataset: {ds}  ({ds['ssh'].to_numpy().nbytes / 1e6:.1f} MB ssh)")

# %%
# Create the collection
# ---------------------
#
# :func:`~zcollection.create_collection` writes the schema to disk and returns
# a writable :py:class:`~zcollection.Collection`. The
# :py:class:`~zcollection.partitioning.Sequence` partitioner splits rows by
# the ``partition`` variable.
collection = zc.create_collection(
    f"file://{target}",
    schema=schema,
    axis="time",
    partitioning=zc.partitioning.Sequence(("partition",), dimension="time"),
)

# %%
# Insert data
# -----------
#
# Rows are automatically routed to the correct partition on disk.
written = collection.insert(ds)
print(f"wrote {len(written)} partitions: {written}")

# %%
# Reopen and query
# ----------------
#
# Reopen read-only and load the full dataset back; assert bit-exact equality.
reopened = zc.open_collection(f"file://{target}", mode="r")
print(f"reopened: axis={reopened.axis} parts={list(reopened.partitions())}")

full = reopened.query()
assert numpy.array_equal(full["time"].to_numpy(), ds["time"].to_numpy())
assert numpy.array_equal(full["ssh"].to_numpy(), ds["ssh"].to_numpy())
print("bit-exact round-trip: OK")

# %%
# Filter pushdown
# ---------------
#
# Filters are evaluated against partition keys; only matching partitions are
# read from disk.
sub = reopened.query(filters="partition == 2")
assert sub["partition"].to_numpy().tolist() == [2] * ROWS_PER_PARTITION
print("filter pushdown: OK")
