"""
Working without a Dask cluster.
===============================

A collection can be read and written without a Dask cluster. This example
selects partitions, updates their data and computes mapping results.
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
# Initialization of the environment
# ---------------------------------
base_dir = str(pathlib.Path(tempfile.mkdtemp()) / 'my_collection')


datasets: Iterator[zcollection.Dataset] = \
    zcollection.tests.data.create_test_dataset()
first: zcollection.Dataset = next(datasets)

# %%
# Creating a collection
# ---------------------
#
# Pass ``distributed=False`` to ``insert`` to run these insertions in the
# calling process.
collection: zcollection.Collection = zcollection.create_collection(
    'time', first, zcollection.partitioning.Date(('time', ), resolution='M'),
    base_dir)
collection.insert(first, distributed=False)
for zds in itertools.islice(datasets, 5):
    collection.insert(zds, distributed=False)

pprint.pprint(sorted(collection.partitions(relative=True)))

# %%
# Reading
# -------
#
# ``delayed`` names what the variables of the loaded dataset hold: a Dask
# array, or a NumPy one. The two parameters are independent, and reading in
# the calling process into NumPy arrays reaches for Dask at no point.
loaded = collection.load(delayed=False, distributed=False)
assert loaded is not None
print(type(loaded.variables['var1'].array).__name__)

# %%
# Naming the partitions to reach
# ------------------------------
#
# ``filters`` selects the partitions by the keys they are named with.
loaded = collection.load(filters='year == 2000 and month == 1',
                         distributed=False)
assert loaded is not None
print(loaded.variables['time'].values[[0, -1]])

# %%
# ``selected_partitions`` names them by their path, relative to the
# directory the collection is held in.
loaded = collection.load(selected_partitions=['year=2000/month=02'],
                         distributed=False)
assert loaded is not None
print(loaded.variables['time'].values[[0, -1]])


# %%
# An update reaches the partitions it names, and writes to those alone.
def ones(partition: zcollection.Dataset) -> dict[str, numpy.ndarray]:
    """Return a variable holding ones."""
    return {'var2': partition.variables['var1'].values * 0 + 1}


collection.update(
    ones,  # type: ignore[arg-type]
    selected_partitions=['year=2000/month=02'],
    distributed=False)

for month in ('01', '02'):
    loaded = collection.load(selected_partitions=[f'year=2000/month={month}'],
                             distributed=False)
    assert loaded is not None
    print(f'month={month}: var2 holds ones everywhere: '
          f'{bool(numpy.all(loaded.variables["var2"].values == 1))}')

# %%
# Mapping over the partitions
# ---------------------------
#
# ``map`` returns a :py:class:`dask.bag.Bag`. Choose the scheduler when
# computing the results, as in the following two calls.
bag = collection.map(
    lambda partition: float(partition.variables['var1'].values.sum()))
pprint.pprint(bag.compute(scheduler='threads'))

# %%
# .. note::
#
#     A bag computed with no scheduler named, and no cluster to connect to,
#     runs the work in processes of its own. Python starts them by importing
#     the module that holds the call, so a script that computes a bag that
#     way holds its calls under ``if __name__ == '__main__':``.
pprint.pprint(bag.compute(scheduler='synchronous'))
