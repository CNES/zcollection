"""
Working with a view
===================

This example shows how to create a view from an existing collection,
add and update its own variables, synchronize it with its reference
collection, and perform computations on its partitions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import pprint

import dask.distributed
import fsspec
import numpy

import zcollection
import zcollection.tests.data

if TYPE_CHECKING:
    from collections.abc import Iterator


# %%
# Setting up the environment
# --------------------------
#
# Start by creating a collection using the same approach as in
# :ref:`Working with a collection <sphx_glr_auto_examples_ex_collection.py>`.
#
# The example uses an in-memory file system and a local Dask cluster
# with threads, allowing workers to share the same storage.
def create_dataset() -> zcollection.Dataset:
    """Create a dataset to record."""
    generator: Iterator[zcollection.Dataset] = \
        zcollection.tests.data.create_test_dataset_with_fillvalue()
    return next(generator)


cluster = dask.distributed.LocalCluster(processes=False)
client = dask.distributed.Client(cluster)

zds: zcollection.Dataset | None = create_dataset()
assert zds is not None
fs: fsspec.AbstractFileSystem = fsspec.filesystem('memory')
collection: zcollection.Collection = zcollection.create_collection(
    'time',
    zds,
    zcollection.partitioning.Date(('time', ), resolution='M'),
    '/view_reference',
    filesystem=fs)
collection.insert(zds, merge_callable=zcollection.merging.merge_time_series)

# %%
# Creating a view
# ---------------
#
# A :py:class:`view <zcollection.view.View>` lets you store additional
# variables derived from a collection without modifying that collection.
#
# The view references its source collection through a
# :py:class:`ViewReference <zcollection.view.ViewReference>` and accesses
# its variables in read-only mode.
#
# Create a view in a separate directory:
view: zcollection.View = zcollection.create_view(
    '/my_view',
    zcollection.view.ViewReference('/view_reference', fs),
    filesystem=fs)

# %%
# By default, a view includes all partitions of its reference collection.
# You can select a subset using the ``filters`` argument when creating it.
#
# The new view has no variables of its own yet. Inspect its directory:
pprint.pprint(fs.listdir('/my_view'))

# %%
# Until variables are added to the view, loading it returns the variables
# of the reference collection.
#
# .. note::
#
#    You can reopen an existing view using:
#
#    .. code-block:: python
#
#        view = zcollection.open_view("/my_view", filesystem=fs)
#
zds = view.load()
assert zds is not None
print(sorted(zds.variables))

# %%
# Adding and updating variables
# -----------------------------
#
# Use :py:meth:`~zcollection.view.View.add_variable` to add variables
# to the view without changing its reference collection.
#
# Here, create ``var3`` using the schema of ``var2``, with a different name:
var3_template: zcollection.meta.VariableSchema = zds.metadata(
).variables['var2']
var3_template.name = 'var3'
view.add_variable(var3_template)
del var3_template

# %%
# Adding a variable creates its array in each partition of the view.
# Inspect the first partition to see the stored files:
pprint.pprint(fs.listdir('/my_view/year=2000/month=01'))

# %%
# The new variable is initialized with its fill value, which is masked
# when the data is read.
zds = view.load()
assert zds is not None
zds.variables['var3'].values

# %%
# Use :py:meth:`~zcollection.view.View.update` to compute values for
# the view's variables, following the same principle as updating a
# collection.
#
# The function can read variables from both the view and its reference
# collection, but writes only to the view.
#
# For example, set ``var3`` to 1 wherever ``var1`` has defined values:
view.update(
    lambda ds: {'var3': ds['var1'].values * 0 + 1})  # type: ignore[arg-type]

# %%
# As with collection updates, ``depth`` can be used to include
# neighboring partitions in the computation. See :doc:`/computing`
# for details on working with overlapping partitions.
#
# Load the view to inspect the updated values:
zds = view.load()
assert zds is not None
var3: numpy.ndarray = zds['var3'].values
print(var3)

# %%
# Variables belonging to the reference collection are read-only.
# Attempting to update ``var2`` through the view raises ``ValueError``:
try:
    view.update(
        lambda ds: {'var2': ds['var2'].values * 0})  # type: ignore[arg-type]
except ValueError as exc:
    print(str(exc))


# %%
# Synchronizing with the reference collection
# -------------------------------------------
#
# When a partition changes in the reference collection, its corresponding
# partition in the view may become inconsistent.
#
# For example, if the number of measurements changes, the arrays in the
# view may no longer have the expected dimensions, preventing the view
# from being loaded.
#
# :py:meth:`~zcollection.view.View.sync` synchronizes the view with its
# reference collection.
#
# When a reference partition is extended with new measurements after
# the existing ones, the view preserves its existing values and extends
# its arrays with their fill values.
#
# Other changes cause the affected partition to be rebuilt, so its
# previously computed values must be calculated again.
#
# The following example changes the last partition of the reference
# collection. Its new timestamps start before the first stored one, so
# they do not extend the existing partition.
#
# As a result, ``sync`` rebuilds the partition, and ``var3`` contains
# only its fill value until it is computed again.
#
# First, define a helper function to resize a dataset.
def resize(ds: zcollection.Dataset, dim: str,
           size: int) -> zcollection.Dataset:
    """Resize a dataset."""

    def new_shape(
        var: zcollection.Variable,
        selected_dim: str,
        new_size: int,
    ) -> tuple[int, ...]:
        """Compute the new shape of a variable."""
        return tuple(new_size if dim == selected_dim else size
                     for dim, size in zip(var.dimensions, var.shape))

    return zcollection.Dataset([
        zcollection.Array(
            name,
            numpy.resize(var.array.compute(), new_shape(var, dim, size)),
            var.dimensions,
            attrs=var.attrs,
            compressor=var.compressor,
            fill_value=var.fill_value,
            filters=var.filters,
        ) for name, var in ds.variables.items()
    ])


# %%
# Open the reference collection in write mode and load its last partition,
# corresponding to June 2000.
collection = zcollection.open_collection('/view_reference',
                                         filesystem=fs,
                                         mode='w')
zds = collection.load(
    filters=lambda keys: keys['month'] == 6 and keys['year'] == 2000)
assert zds is not None

# %%
# Generate hourly timestamps for June, then resize the dataset and replace
# its ``time`` values.
time: numpy.ndarray = numpy.arange(
    numpy.datetime64('2000-06-01T00:00:00', 'ns'),
    numpy.datetime64('2000-06-30T23:59:59', 'ns'),
    numpy.timedelta64(1, 'h'),
)
zds = resize(zds, 'num_lines', len(time))
zds['time'].values = time

# %%
# Insert the modified dataset into the reference collection, replacing
# its existing June partition.
collection.insert(zds)

# %%
# The view still contains arrays corresponding to the previous version
# of that partition. Their dimensions no longer match the reference,
# so attempting to load the view raises ``ValueError``:
try:
    view.load(distributed=False)
except ValueError as err:
    print(err)

# %%
# Synchronize the view to update the affected partition:
filters = view.sync()

# %%
# ``sync`` returns a callable that selects the partitions it synchronized.
# Use it to identify which partitions need their computed values updated.
#
# For example, the following call would recompute ``var3`` only in those
# partitions:
#
# .. code-block:: python
#
#    view.update(
#        lambda ds: dict(var3=ds['var1'].values * 0 + 1),
#        filters=filters)
#
# Inspect the partitions selected by the returned callable:
print(tuple(view.partitions(filters=filters)))

# %%
# The view is now synchronized and can be loaded again.
# ``var3`` retains its computed values in the other partitions, but
# its values in June are masked because that partition was rebuilt.
zds = view.load()
assert zds is not None
zds.variables['var3'].values

# %%
# Mapping over a view
# -------------------
#
# :py:meth:`~zcollection.view.View.map` applies a function to each
# partition without modifying the view.
#
# Like ``Collection.map``, it returns a Dask bag. Call ``compute()``
# to obtain the results.
#
# For example, calculate the mean of ``var1 + var2`` in each partition:
for partition, array in view.map(lambda ds: (  # type: ignore[arg-type]
        ds['var1'].values + ds['var2'].values)).compute():
    print(f' * partition = {partition}: mean = {array.mean()}')

# %%
# .. seealso::
#
#    :py:meth:`~zcollection.view.View.map_overlap` also supports
#    computations using neighboring partitions.
#
# Removing a variable
# -------------------
#
# Use :py:meth:`~zcollection.view.View.drop_variable` to remove a
# variable belonging to the view.
#
# Here, remove ``var3``:
view.drop_variable('var3')

# %%
# The view now exposes only the variables of its reference collection:
zds = view.load()
assert zds is not None
print(sorted(zds.variables))

# %%
# Variables belonging to the reference collection cannot be removed
# through the view. Attempting to remove ``var2`` raises ``ValueError``:
try:
    view.drop_variable('var2')
except ValueError as exc:
    print(str(exc))

# %%
# Close the Dask client and cluster to release their resources.
client.close()
cluster.close()
