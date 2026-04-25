"""
Hierarchical Groups
===================

Builds a dataset with nested Zarr groups (``/data_01/ku/...``), persists
it through a partitioned :py:class:`~zcollection.Collection`, and
demonstrates the path-based access, dimension inheritance, and the
size-aware multi-line ``__repr__``.

Run with::

    python examples/ex_groups.py
"""

from pathlib import Path
import shutil
import tempfile

import numpy
import zarr

import zcollection as zc

# %%
# Initialization
# --------------
#
# Set up a temporary directory for the collection.
target = Path(tempfile.gettempdir()) / "zc-groups"
if target.exists():
    shutil.rmtree(target)

# %%
# Build a hierarchical schema
# ---------------------------
#
# Groups are declared via the ``group=`` keyword on
# :meth:`~zcollection.SchemaBuilder.with_dimension` /
# :meth:`~zcollection.SchemaBuilder.with_variable` /
# :meth:`~zcollection.SchemaBuilder.with_attribute`, or by calling
# :meth:`~zcollection.SchemaBuilder.with_group` to attach group-level
# attributes ahead of time. Intermediate groups along the path are
# created on demand. Variables in ``/data_01/ku`` reference the root's
# ``time`` dimension via dimension inheritance.
schema = (
    zc.Schema()
    .with_dimension("time", chunks=4096)
    .with_variable("time", dtype="int64", dimensions=("time",))
    .with_group("/data_01", attrs={"product": "L2"})
    .with_group("/data_01/ku", attrs={"band": "Ku"})
    .with_dimension("range", size=240, chunks=240, group="/data_01/ku")
    .with_variable(
        "power",
        dtype="float32",
        dimensions=("time", "range"),  # ``time`` inherited from root
        group="/data_01/ku",
    )
    .build()
)
print(schema.all_variables())

# %%
# Build a sample dataset
# ----------------------
#
# Variables placed inside nested groups are addressed by their absolute
# path in the constructor mapping. Short names (without ``/``) populate
# the root group.
N = 5_000
ku = schema.groups["data_01"].groups["ku"]
ds = zc.Dataset(
    schema=schema,
    variables={
        "time": zc.Variable(
            schema.variables["time"], numpy.arange(N, dtype="int64")
        ),
        "data_01/ku/power": zc.Variable(
            ku.variables["power"],
            numpy.random.default_rng(0)
            .standard_normal(size=(N, 240), dtype="float32"),
        ),
    },
)

# %%
# The size-aware multi-line repr
# ------------------------------
#
# :class:`~zcollection.Dataset` and :class:`~zcollection.Variable` print
# as multi-line, xarray-like blocks. The byte size is computed
# recursively for the dataset and each child group so you can gauge
# memory/disk footprint at a glance.
print(ds)

# %%
# Path-based access
# -----------------
#
# Use absolute paths (``"/data_01/ku/power"``) or short-name search
# (:meth:`~zcollection.Group.find_variable`) to navigate the hierarchy.
# :meth:`~zcollection.Group.find_dimension` walks up the tree so child
# groups inherit dimensions declared on ancestors.
power = ds.get_variable("/data_01/ku/power")
ku_group = ds.get_group("/data_01/ku")
print(f"power shape: {power.shape}, long_name: {ku_group.long_name()!r}")
print(f"inherited dim: {ku_group.find_dimension('time')!r}")

# %%
# Persist the dataset
# -------------------
#
# A partitioned collection writes nested Zarr groups *inside* every
# partition. ``open_collection().query()`` reconstructs the same
# in-memory hierarchy.
collection = zc.create_collection(
    f"file://{target}",
    schema=schema,
    axis="time",
    partitioning=zc.partitioning.Sequence(("time",), dimension="time"),
)
collection.insert(ds)

# %%
# Inspect the on-disk layout
# --------------------------
#
# Each partition is a real Zarr v3 group containing a real ``data_01``
# subgroup, which itself contains a real ``ku`` subgroup with the
# ``power`` array. No flattening, no name mangling.
zstore = collection.store.zarr_store()
partition_paths = list(collection.partitions())
first = zarr.open_group(store=zstore, path=partition_paths[0], mode="r")
print(f"top-level keys: {list(first)}")
print(f"  /data_01 keys: {list(first['data_01'])}")
print(f"  /data_01/ku keys: {list(first['data_01/ku'])}")

# %%
# Reopen and verify
# -----------------
#
# The query result is a :class:`~zcollection.Dataset` with the same
# hierarchy as the original.
reopened = zc.open_collection(f"file://{target}", mode="r").query()
assert reopened.get_variable("/data_01/ku/power").shape == (N, 240)
print(reopened)
