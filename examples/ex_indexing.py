"""
Indexing a collection
=====================

This example shows how to create a custom index for a collection and use
it to retrieve selected measurements without reading the entire dataset.

The collection contains satellite measurements partitioned by month.
The index provides another way to access the data, using satellite cycle
and pass numbers to locate individual half-orbits.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import pprint

import dask.distributed
import fsspec
import numpy

import zcollection
import zcollection.indexing
import zcollection.partitioning.tests.data

if TYPE_CHECKING:
    from collections.abc import Iterator
    import pathlib

# %%
# Setting up the environment
# --------------------------
#
# Use an in-memory file system and a local Dask cluster with threads,
# allowing workers to share the same storage.
fs = fsspec.filesystem('memory')
cluster = dask.distributed.LocalCluster(processes=False)
client = dask.distributed.Client(cluster)

# %%
# Creating a collection to index
# ------------------------------
#
# Start with a dataset containing measurements from a fictitious
# satellite, organized into several half-orbits.
#
# Each half-orbit is identified by a cycle number and a pass number.
# These values will later be used to build the index.
zds: zcollection.Dataset = zcollection.Dataset.from_xarray(
    zcollection.partitioning.tests.data.create_test_sequence(5, 20, 10))
print(zds)

# %%
# Create a collection partitioned by month using the ``time`` variable,
# then insert the measurements.
collection: zcollection.Collection = zcollection.create_collection(
    'time',
    zds,
    zcollection.partitioning.Date(('time', ), 'M'),
    partition_base_dir='/one_other_collection',
    filesystem=fs)
collection.insert(zds, merge_callable=zcollection.merging.merge_time_series)

# %%
# The collection is organized into monthly partitions. Inspect the
# directories created for the year 2000:
pprint.pprint(fs.listdir('/one_other_collection/year=2000'))


# %%
# Identifying half-orbits
# -----------------------
#
# The index will identify consecutive measurements belonging to the
# same half-orbit, even though the collection itself is partitioned
# by month.
#
# A half-orbit corresponds to a sequence of measurements with constant
# cycle and pass numbers.
#
# First, define a function that finds where either value changes.
# It returns the start and stop indices of each consecutive sequence.
def split_half_orbit(
    cycle_number: numpy.ndarray,
    pass_number: numpy.ndarray,
) -> Iterator[tuple[int, int]]:
    """Calculate the indexes of the start and stop of each half-orbit.

    Args:
        cycle_number: Cycle numbers.
        pass_number: Pass numbers.
    Returns:
        Iterator of start and stop indexes.
    """
    assert pass_number.shape == cycle_number.shape
    changes = numpy.flatnonzero((numpy.diff(pass_number) != 0)
                                | (numpy.diff(cycle_number) != 0)) + 1
    half_orbit = numpy.concatenate(([0], changes, [pass_number.size]))

    yield from zip(half_orbit[:-1], half_orbit[1:])


# %%
# Building index records
# ----------------------
#
# The next function applies this detection to a dataset loaded from
# one partition.
#
# For each half-orbit, it creates an index record containing its start
# and stop positions, cycle number, and pass number.
#
# The records are returned as a structured NumPy array, using the
# data types defined by the indexer.
def _half_orbit(
    zds: zcollection.Dataset,
    *args,
    dtype: numpy.dtype | None = None,
    **kwargs,
) -> numpy.ndarray:
    """Return the indexes of the start and stop of each half-orbit.

    Args:
        zds: Datasets stored in a partition to be indexed.
    Returns:
        Structured array holding the start and stop indexes, the cycle
        number and the pass number of each half-orbit.
    """
    pass_number_varname = kwargs.pop('pass_number', 'pass_number')
    cycle_number_varname = kwargs.pop('cycle_number', 'cycle_number')
    pass_number = zds.variables[pass_number_varname].values
    cycle_number = zds.variables[cycle_number_varname].values

    generator = ((
        i0,
        i1,
        cycle_number[i0],
        pass_number[i0],
    ) for i0, i1 in split_half_orbit(cycle_number, pass_number))

    return numpy.fromiter(generator, dtype)


# %%
# Defining the indexer
# --------------------
#
# Create a custom indexer by extending
# :py:class:`~zcollection.indexing.abc.Indexer`.
#
# The base class provides the mechanisms for storing, updating, and
# querying the index. The subclass defines the additional information
# to store and how to extract it from the collection.
#
# Three methods are implemented:
#
# - ``dtype`` defines the columns of the index, including the cycle
#   and pass numbers.
# - ``create`` initializes the index at a given location.
# - ``update`` processes the collection's partitions using
#   ``_half_orbit`` to generate the index records.
class HalfOrbitIndexer(zcollection.indexing.Indexer):
    """Index collection by half-orbit."""

    #: Column name of the cycle number.
    CYCLE_NUMBER = 'cycle_number'

    #: Column name of the pass number.
    PASS_NUMBER = 'pass_number'

    def dtype(self, /, **kwargs) -> list[tuple[str, str]]:
        """Return the columns of the index.

        Returns:
            A list of (name, type) pairs.
        """
        return [
            *super().dtype(), (self.CYCLE_NUMBER, 'uint16'),
            (self.PASS_NUMBER, 'uint16')
        ]

    @classmethod
    def create(
        cls,
        path: pathlib.Path | str,
        zds: zcollection.Collection,
        filesystem: fsspec.AbstractFileSystem | None = None,
        **kwargs,
    ) -> HalfOrbitIndexer:
        """Create a new index.

        Args:
            path: The path to the index.
            zds: The collection to be indexed.
            filesystem: The filesystem to use.
        Returns:
            The created index.
        """
        return super()._create(  # type: ignore[return-value]
            path,
            zds,
            meta={'attribute': b'value'},
            filesystem=filesystem,
        )

    def update(
        self,
        zds: zcollection.Collection,
        partition_size: int | None = None,
        npartitions: int | None = None,
        **kwargs,
    ) -> None:
        """Update the index.

        Args:
            zds: New data stored in the collection to be indexed.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            cycle_number: The name of the cycle number variable stored in the
                collection. Defaults to "cycle_number".
            pass_number: The name of the pass number variable stored in the
                collection. Defaults to "pass_number".
        """
        super()._update(zds,
                        _half_orbit,
                        partition_size,
                        npartitions,
                        dtype=self.dtype(),
                        **kwargs)


# %%
# Creating and querying the index
# -------------------------------
#
# Create the index in a separate file, then populate it using the
# measurements stored in the collection.
indexer: HalfOrbitIndexer = HalfOrbitIndexer.create('/index.parquet',
                                                    collection,
                                                    filesystem=fs)
indexer.update(collection)

# The resulting table contains the start and stop positions of each
# half-orbit, its cycle and pass numbers, and the partition containing it.
#
# Inspect the index as a pandas DataFrame:
indexer.table.to_pandas()

# %%
# The index can now be queried to locate measurements without scanning
# the entire collection.
#
# For example, select the half-orbits whose pass number is 1 or 2.
# The query returns the information needed by ``load`` to retrieve
# the corresponding measurements.
selection: zcollection.Dataset | None = collection.load(
    indexer=indexer.query({'pass_number': [1, 2]}),
    delayed=False,
)
assert selection is not None
selection.to_xarray()

# %%
# Close the Dask client and cluster to release their resources.
client.close()
cluster.close()
