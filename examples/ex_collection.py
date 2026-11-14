"""
Working with a collection
=========================

This example walks through the main features of ZCollection: creating a
collection, inserting and reading data, managing variables, and applying
computations to partitions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import datetime
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
# Start by creating a sample dataset that will be used throughout the example.
def create_dataset() -> zcollection.Dataset:
    """Create a dataset to record."""
    generator: Iterator[zcollection.Dataset] = \
        zcollection.tests.data.create_test_dataset_with_fillvalue()
    return next(generator)


zds: zcollection.Dataset | None = create_dataset()
assert zds is not None
zds.to_xarray()

# %%
# Use an in-memory file system to store the collection without writing
# anything to disk.
fs: fsspec.AbstractFileSystem = fsspec.filesystem('memory')

# %%
# Create a local Dask cluster using threads. This allows the workers to
# share the same in-memory file system.
cluster = dask.distributed.LocalCluster(processes=False)
client = dask.distributed.Client(cluster)

# %%
# Creating the collection
# -----------------------
#
# Before creating a collection, define how its data will be partitioned.
# Here, measurements are grouped by month using the ``time`` variable.
partition_handler = zcollection.partitioning.Date(('time', ), resolution='M')

# %%
# Create the collection using the dataset's schema, the partitioning scheme,
# and the in-memory file system.
collection: zcollection.Collection = zcollection.create_collection(
    'time', zds, partition_handler, '/my_collection', filesystem=fs)

# %%
# The collection's configuration records its variables, dimensions, and
# storage settings. ZCollection uses this metadata to ensure that future
# insertions remain compatible with the collection.
#
# .. note::
#
#    You can reopen an existing collection using:
#
#    .. code-block:: python
#
#        collection = zcollection.open_collection(
#            "/my_collection", filesystem=fs)
#
# Inspect the configuration:
pprint.pprint(collection.metadata.get_config())

# %%
# Inserting data
# --------------
#
# The collection is now initialized and ready to receive data.
# ``insert`` splits the dataset according to the partitioning scheme
# and writes each monthly partition.
collection.insert(zds)

# %%
# By default, inserting data into an existing partition replaces its
# contents. See :doc:`/inserting` for details on merging new data with
# existing measurements.
#
# Inspect the directories created for the year 2000:
pprint.pprint(fs.listdir('/my_collection/year=2000'))

# %%
# Loading data
# ------------
#
# Although the data is stored in separate partitions, ``load`` returns
# a single dataset.
#
# By default, :py:meth:`~zcollection.collection.Collection.load` reads
# all partitions. With ``delayed=True``, the returned variables contain
# Dask arrays, whose values are loaded when needed.
collection.load(delayed=True)

# %%
# You can select specific partitions using ``filters``. The expression
# refers to the keys defined by the partitioning scheme, here ``year``
# and ``month``.
#
# For example, load only February 2000:
collection.load(filters='year == 2000 and month == 2')

# %%
# For more complex selections, ``filters`` also accepts a callable
# receiving the partition keys.
#
# This example selects months whose first day falls between the
# specified dates.
collection.load(
    filters=lambda keys: datetime.date(2000, 2, 15) <= datetime.date(
        keys['year'], keys['month'], 1) <= datetime.date(2000, 3, 15))

# %%
# If no partition matches the selection, ``load`` returns ``None``.
assert collection.load(filters='year == 2002 and month == 2') is None

# %%
# Managing variables
# ------------------
#
# Variables can be added to or removed from an existing collection.
# These operations require write access and are not available when
# the collection is opened in read-only mode.
#
# First, remove ``var2`` and load the collection to inspect the result:
collection.drop_variable('var2')
collection.load()

# %%
# Variables used by the partitioning scheme cannot be removed.
# Attempting to remove ``time`` raises ``ValueError``:
try:
    collection.drop_variable('time')
except ValueError as exc:
    print(exc)

# %%
# Use :py:meth:`~zcollection.collection.Collection.add_variable` to
# declare a new variable and create its arrays in the collection.
#
# Here, restore ``var2`` using its original schema:
collection.add_variable(zds.metadata().variables['var2'])

# %%
# The new variable is initialized with its fill value. Values equal
# to this fill value are masked when the collection is read.
zds = collection.load()
assert zds is not None
zds.variables['var2'].values


# %%
# Updating variables
# ------------------
#
# :py:meth:`~zcollection.collection.Collection.update` computes new
# variable values by applying a function to each partition.
#
# The function receives a partition dataset and returns a dictionary
# mapping variable names to their computed arrays.
#
# In this example, ``var2`` is set to 1 wherever ``var1`` has defined
# values.
def ones(zds) -> dict[str, numpy.ndarray]:
    """Returns a variable with ones everywhere."""
    return {'var2': zds.variables['var1'].values * 0 + 1}


collection.update(ones)  # type: ignore[arg-type]

zds = collection.load()
assert zds is not None
zds.variables['var2'].values


# %%
# Using neighboring partitions
# ----------------------------
#
# Some computations require information from neighboring partitions.
# With ``depth=1``, ``update`` loads one neighboring partition on each
# side of the partition being processed.
#
# A function declaring ``partition_info`` receives information about
# the processed partition, including its position within the window.
#
# Here, set ``var2`` to:
#
# - 2 when the partition has neighbors on both sides.
# - -1 when its left neighbor is missing.
# - -2 when its right neighbor is missing.
#
# ``partition_info.indices`` identifies the processed partition within
# the loaded window. A slice starting at 0 indicates that no left
# neighbor was loaded. Similarly, a slice ending at the window's length
# indicates that no right neighbor was loaded.
#
# See :doc:`/computing` for more information about ``PartitionInfo``.
def twos(
        ds,
        partition_info: zcollection.PartitionInfo) -> dict[str, numpy.ndarray]:
    """Returns a variable with twos everywhere if the partition is surrounded
    by partitions on both sides, -1 if the left partition is missing and -2 if
    the right partition is missing."""
    data = numpy.zeros(ds.variables['var1'].shape, dtype='int8')
    dim, indices = partition_info.dimension, partition_info.indices
    assert dim == 'num_lines'
    if indices.start == 0:
        data[:] = -1
    elif indices.stop == data.shape[0]:
        data[:] = -2
    else:
        data[:] = 2
    return {'var2': data}


collection.update(twos, depth=1)  # type: ignore[arg-type]

zds = collection.load()
assert zds is not None
zds.variables['var2'].values

# %%
# Mapping over partitions
# -----------------------
#
# :py:meth:`~zcollection.collection.Collection.map` applies a function
# to each partition without modifying the collection.
#
# It returns a Dask bag containing the partition keys and the results
# of the function. Call ``compute()`` to execute the computation.
#
# For example, calculate the mean of ``var1 + var2`` in each partition:
for partition, array in collection.map(lambda ds: (  # type: ignore[arg-type]
        ds['var1'].values + ds['var2'].values)).compute():
    print(f' * partition = {partition}: mean = {array.mean()}')

# %%
# :py:meth:`~zcollection.collection.Collection.map_overlap` works
# similarly, but loads neighboring partitions along with the one being
# processed.
#
# With ``depth=1``, each function call receives a window containing
# the partition and up to one neighbor on either side.
#
# The example below computes ``var1 + var2`` over each window:
for partition, array in collection.map_overlap(
        lambda ds: (  # type: ignore[arg-type]
            ds['var1'].values + ds['var2'].values),
        depth=1).compute():
    print(f' * partition = {partition}: mean = {array.mean()}')

# %%
# Close the Dask client and cluster to release their resources and
# avoid warnings when running other examples.
client.close()
cluster.close()
