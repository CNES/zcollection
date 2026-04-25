"""Overview of a Collection.

==========================

This section walks through the main features of a v3
:py:class:`~zcollection.Collection`: building a schema, creating a partitioned
collection on a store, inserting data, querying with filters, and updating
variables in place.

Run with::

    python examples/ex_collection.py
"""

from pathlib import Path
import pprint
import shutil
import tempfile

import numpy

import zcollection as zc

# %%
# Initialization
# --------------
#
# v3 stores are URL-driven. ``LocalStore`` (POSIX), ``MemoryStore``
# (in-process, useful for tests), and ``IcechunkStore`` (transactional) are
# built in. Pass a string URL to :func:`~zcollection.create_collection` and
# the right backend is chosen for you.
target = Path(tempfile.gettempdir()) / "zc-ex-collection"
if target.exists():
    shutil.rmtree(target)


# %%
# Build a schema
# --------------
#
# A v3 collection is created from an explicit
# :py:class:`~zcollection.DatasetSchema`. The fluent
# :py:class:`~zcollection.SchemaBuilder` (aliased as ``zc.Schema()``) declares
# dimensions and variables up front, including their codecs.
def build_schema() -> zc.DatasetSchema:
    """Build the dataset schema for the example."""
    return (
        zc.Schema()
        .with_dimension("time", chunks=4096)
        .with_dimension("x_ac", size=240, chunks=240)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("partition", dtype="int64", dimensions=("time",))
        .with_variable("var1", dtype="float32", dimensions=("time", "x_ac"))
        .with_variable(
            "var2",
            dtype="float32",
            dimensions=("time", "x_ac"),
            fill_value=numpy.float32("nan"),
        )
        .build()
    )


schema = build_schema()


# %%
# Build a sample dataset
# ----------------------
#
# A :py:class:`~zcollection.Dataset` is the in-memory pairing of a schema
# with concrete numpy (or dask) arrays. There is only one ``Variable`` class
# in v3 — the ``Array`` / ``DelayedArray`` split from v2 is gone.
def build_dataset(
    schema: zc.DatasetSchema, n_partitions: int = 3
) -> zc.Dataset:
    """Build a synthetic dataset matching the given schema."""
    rng = numpy.random.default_rng(42)
    rows_per_part = 10_000
    n = n_partitions * rows_per_part
    time = numpy.arange(n, dtype="int64")
    partition = numpy.repeat(
        numpy.arange(n_partitions, dtype="int64"), rows_per_part
    )
    var1 = rng.standard_normal(size=(n, 240), dtype="float32")
    var2 = numpy.full((n, 240), numpy.float32("nan"))
    return zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(schema.variables["time"], time),
            "partition": zc.Variable(schema.variables["partition"], partition),
            "var1": zc.Variable(schema.variables["var1"], var1),
            "var2": zc.Variable(schema.variables["var2"], var2),
        },
    )


zds = build_dataset(schema)
print(zds)

# %%
# Create the collection
# ---------------------
#
# Every keyword is named — there is no positional ``(axis, ds, partitioner,
# path, fs)`` form anymore. The collection inherits its schema from the call;
# the schema is then frozen on disk and enforced by every later insert.
#
# Partitionings are pure-numpy splitters that map a dataset's rows to
# partition keys. Here we use a synthetic
# :py:class:`~zcollection.partitioning.Sequence` bucket variable. For real
# time series, prefer :py:class:`~zcollection.partitioning.Date`.
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
# Insertion is partitioned automatically. The default merge strategy is
# ``replace`` (last write wins for a given partition); other strategies live
# under :py:mod:`zcollection.merge`.
written = collection.insert(zds)
print(f"wrote {len(written)} partitions: {written}")

# %%
# .. note::
#
#    Use :py:func:`zcollection.merge.time_series` to incrementally append
#    along a sorted time axis without overwriting prior data::
#
#        from zcollection import merge
#        collection.insert(zds, merge=merge.time_series)
#
# Reopen and list partitions
# --------------------------
collection = zc.open_collection(f"file://{target}", mode="rw")
pprint.pprint(list(collection.partitions()))

# %%
# Query
# -----
#
# :py:meth:`~zcollection.Collection.query` returns a
# :py:class:`~zcollection.Dataset` (or ``None`` if no partition matches).
# Filters use a typed expression language over the partition keys; no
# ``eval`` is involved.
loaded = collection.query()
assert loaded is not None
print(f"full query: {loaded}")

filtered = collection.query(filters="partition == 1")
assert filtered is not None
assert filtered["partition"].to_numpy().tolist() == [1] * 10_000
print("filter pushdown: OK")

# %%
# Subset variables
subset = collection.query(variables=("time", "var1"))
assert subset is not None
print(f"variable subset: {tuple(subset.variables)}")


# %%
# Update variables in place
# -------------------------
#
# :py:meth:`~zcollection.Collection.update` rewrites the partitions matching
# ``filters`` after applying ``fn`` to each one. ``fn`` receives a Dataset
# and must return a new Dataset with the same dimensions.
def square_var1(ds: zc.Dataset) -> zc.Dataset:
    """Return a dataset with ``var1`` squared."""
    arr = ds["var1"].to_numpy() ** 2
    return zc.Dataset(
        schema=ds.schema,
        variables={
            **{name: ds[name] for name in ds},
            "var1": zc.Variable(ds.schema.variables["var1"], arr),
        },
    )


collection.update(square_var1, filters="partition == 0")
after = collection.query(filters="partition == 0")
assert after is not None
print(f"updated max(var1)= {float(after['var1'].to_numpy().max()):.3f}")

# %%
# Map a function over partitions
# ------------------------------
#
# :py:meth:`~zcollection.Collection.map` applies ``fn`` to every partition
# (no write-back) and returns ``{partition_path: result}``. Use it for
# reductions, statistics, or building secondary indices.
means = collection.map(lambda ds: float(ds["var1"].to_numpy().mean()))
for path, mu in means.items():
    print(f" * {path}: mean(var1) = {mu:+.4f}")

# %%
# Drop partitions
# ---------------
#
# Surgical deletes use the same filter language. Reserved layout files
# (``zarr.json``, ``_immutable``, ``_catalog``) are never touched.
deleted = collection.drop_partitions(filters="partition == 2")
print(f"dropped: {deleted}")
print(f"remaining: {list(collection.partitions())}")
