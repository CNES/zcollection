"""
Working without a Dask cluster
==============================

ZCollection can read, write, and update collections without a Dask cluster.
This example shows how to perform these operations in the calling process,
select specific partitions, and choose how to compute mapping results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import itertools
import pathlib
import pprint
import tempfile

import numpy

import zcollection
import zcollection.tests.data

if TYPE_CHECKING:
    from collections.abc import Iterator

# %%
# Setting up the environment
# --------------------------
#
# Start by creating a temporary directory and loading a sample dataset.
# The remaining datasets will be inserted into the same collection.
base_dir = str(pathlib.Path(tempfile.mkdtemp()) / 'my_collection')


datasets: Iterator[zcollection.Dataset] = \
    zcollection.tests.data.create_test_dataset()
first: zcollection.Dataset = next(datasets)

# %%
# Creating a collection
# ---------------------
#
# Create a collection partitioned by month using the ``time`` variable.
#
# Passing ``distributed=False`` to ``insert`` processes the partitions
# in the calling process, without requiring a Dask cluster.
#
# Insert the first dataset, followed by five additional datasets, then
# inspect the partitions that were created.
collection: zcollection.Collection = zcollection.create_collection(
    'time', first, zcollection.partitioning.Date(('time', ), resolution='M'),
    base_dir)
collection.insert(first, distributed=False)
for zds in itertools.islice(datasets, 5):
    collection.insert(zds, distributed=False)

pprint.pprint(sorted(collection.partitions(relative=True)))

# %%
# Reading data
# ------------
#
# The ``distributed`` and ``delayed`` parameters control different aspects
# of loading a collection:
#
# - ``distributed=False`` reads the partitions without a Dask cluster.
# - ``delayed=False`` loads their values immediately into NumPy arrays.
#
# Using both gives a dataset backed by NumPy arrays, loaded in the
# calling process.
loaded = collection.load(delayed=False, distributed=False)
assert loaded is not None
print(type(loaded.variables['var1'].array).__name__)

# %%
# Selecting partitions
# --------------------
#
# You can load only part of a collection using ``filters`` or
# ``selected_partitions``.
#
# ``filters`` selects partitions using the keys defined by the
# partitioning scheme. Here, select January 2000:
loaded = collection.load(filters='year == 2000 and month == 1',
                         distributed=False)
assert loaded is not None
print(loaded.variables['time'].values[[0, -1]])

# %%
# ``selected_partitions`` selects partitions by their paths, relative
# to the collection's directory.
#
# For example, load February 2000:
loaded = collection.load(selected_partitions=['year=2000/month=02'],
                         distributed=False)
assert loaded is not None
print(loaded.variables['time'].values[[0, -1]])


# %%
# Updating selected partitions
# ----------------------------
#
# :py:meth:`~zcollection.collection.Collection.update` can also process
# selected partitions without a Dask cluster.
#
# The following function sets ``var2`` to 1 wherever ``var1`` has
# defined values.
def ones(partition: zcollection.Dataset) -> dict[str, numpy.ndarray]:
    """Return a variable holding ones."""
    return {'var2': partition.variables['var1'].values * 0 + 1}


# Update only February, leaving the other partitions unchanged.
collection.update(
    ones,  # type: ignore[arg-type]
    selected_partitions=['year=2000/month=02'],
    distributed=False)

# Compare January and February to see which partition was updated.
for month in ('01', '02'):
    loaded = collection.load(selected_partitions=[f'year=2000/month={month}'],
                             distributed=False)
    assert loaded is not None
    print(f'month={month}: var2 holds ones everywhere: '
          f'{bool(numpy.all(loaded.variables["var2"].values == 1))}')

# %%
# Mapping over partitions
# -----------------------
#
# :py:meth:`~zcollection.collection.Collection.map` returns a
# :py:class:`dask.bag.Bag` containing the results of a function applied
# to each partition.
#
# The bag remains lazy until ``compute()`` is called. You can choose
# the scheduler independently of whether a Dask cluster is available.
#
# For example, compute the sum of ``var1`` in each partition using
# the threaded scheduler:
bag = collection.map(
    lambda partition: float(partition.variables['var1'].values.sum()))
pprint.pprint(bag.compute(scheduler='threads'))

# %%
# The synchronous scheduler executes the computation in the calling
# process, without creating worker processes or threads.
#
# .. note::
#
#    Without an explicit scheduler or a connected Dask client, a Dask
#    bag uses the multiprocessing scheduler by default.
#
#    When running such computations from a Python script, place the
#    relevant calls under ``if __name__ == '__main__':`` to allow
#    worker processes to start correctly.
#
# Compute the same bag synchronously:
pprint.pprint(bag.compute(scheduler='synchronous'))
