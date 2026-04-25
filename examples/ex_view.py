"""
Overview of a View
==================

A :py:class:`~zcollection.view.View` overlays *extra* variables on top of a
read-only base :py:class:`~zcollection.Collection`. The base data is queried
through the view as if it were a single dataset, but the view's variables
live under its own store and only the view-owned variables are writable.

Run with::

    python examples/ex_view.py
"""

from pathlib import Path
import shutil
import tempfile

import numpy

import zcollection as zc
from zcollection.view import View, ViewReference


# %%
# Build a base collection
# -----------------------
def build_schema() -> zc.DatasetSchema:
    """Build the dataset schema for the example."""
    return (
        zc.Schema()
        .with_dimension("time", chunks=4096)
        .with_dimension("x_ac", size=240, chunks=240)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("partition", dtype="int64", dimensions=("time",))
        .with_variable("var1", dtype="float32", dimensions=("time", "x_ac"))
        .build()
    )


root = Path(tempfile.gettempdir()) / "zc-ex-view"
if root.exists():
    shutil.rmtree(root)
base_path = root / "base"
view_path = root / "view"

schema = build_schema()
rng = numpy.random.default_rng(42)
n_partitions = 3
rows_per_part = 10_000
n = n_partitions * rows_per_part
ds = zc.Dataset(
    schema=schema,
    variables={
        "time": zc.Variable(
            schema.variables["time"], numpy.arange(n, dtype="int64")
        ),
        "partition": zc.Variable(
            schema.variables["partition"],
            numpy.repeat(
                numpy.arange(n_partitions, dtype="int64"), rows_per_part
            ),
        ),
        "var1": zc.Variable(
            schema.variables["var1"],
            rng.standard_normal(size=(n, 240), dtype="float32"),
        ),
    },
)

base = zc.create_collection(
    f"file://{base_path}",
    schema=schema,
    axis="time",
    partitioning=zc.partitioning.Sequence(("partition",), dimension="time"),
)
base.insert(ds)
print(f"base: {len(list(base.partitions()))} partitions")


# %%
# Create the view
# ---------------
#
# A view is created from a base collection, a list of :class:`VariableSchema`
# describing the *new* variables it adds, and a :class:`ViewReference`
# pointer to the base. The view owns its own store so the base remains
# read-only from the view's perspective.
view_var = zc.VariableSchema(
    name="var2",
    dtype=numpy.dtype("float32"),
    dimensions=("time", "x_ac"),
    fill_value=numpy.float32("nan"),
)

view_store = zc.open_store(f"file://{view_path}")
view = View.create(
    view_store,
    base=base,
    variables=[view_var],
    reference=ViewReference(uri=f"file://{base_path}"),
)
print(f"view variables: {view.variables}")


# %%
# Populate the view
# -----------------
#
# :py:meth:`~zcollection.view.View.update` runs ``fn`` on every base
# partition. ``fn`` receives the merged base+view :class:`Dataset` and must
# return a ``{view_var_name: numpy_array}`` mapping sized along the
# partitioning dimension.
def derive_var2(base_ds: zc.Dataset) -> dict[str, numpy.ndarray]:
    """Return ``var2`` derived from the base dataset's ``var1``."""
    return {"var2": base_ds["var1"].to_numpy() * 2.0}


view.update(derive_var2)


# %%
# Query through the view
# ----------------------
#
# A view's :py:meth:`~zcollection.view.View.query` returns a Dataset that
# concatenates base and view variables. Filters are pushed down to the base
# collection's partitioning.
out = view.query(filters="partition == 1")
assert out is not None
print(f"merged variables: {tuple(out.variables)}")
assert numpy.array_equal(
    out["var2"].to_numpy(),
    out["var1"].to_numpy() * 2.0,
)
print("view derivation: OK")

# %%
# Read-only opens
# ---------------
ro = View.open(view_store, base=base, read_only=True)
print(f"read-only={ro.read_only}, vars={ro.variables}")
