"""
Indexing a Collection.
======================

In this example, we will see how to index a collection.
"""
from typing import Iterator, List, Optional, Tuple, Union
import pathlib
import pprint

import dask.distributed
import fsspec
import numpy

import zcollection
import zcollection.indexing
import zcollection.tests.data

# %%
# Initialization of the environment
# ---------------------------------
fs = fsspec.filesystem('memory')
cluster = dask.distributed.LocalCluster(processes=False)
client = dask.distributed.Client(cluster)

# %%
# A collection can be indexed. This allows quick access to the data without
# having to browse the entire dataset.
#
# Creating the test collection.
# -----------------------------
#
# For this latest example, we will index another data set. This one contains
# measurements of a fictitious satellite on several half-orbits.
ds = zcollection.Dataset.from_xarray(
    zcollection.tests.data.create_test_sequence(5, 20, 10))
ds

# %%
collection = zcollection.create_collection(
    'time',
    ds,
    zcollection.partitioning.Date(('time', ), 'M'),
    partition_base_dir='/one_other_collection',
    filesystem=fs)
collection.insert(ds, merge_callable=zcollection.merging.merge_time_series)

# %%
# Here we have created a collection partitioned by month.
pprint.pprint(fs.listdir('/one_other_collection/year=2000'))


# %%
# Class to implement
# ------------------
#
# The idea of the implementation is to calculate for each visited partition, the
# slice of data that has a constant quantity. In our example, we will rely on
# the cycle and pass number information. The first method we will implement is
# the detection of these constant parts of two vectors containing the cycle and
# pass number.
def split_half_orbit(
    cycle_number: numpy.ndarray,
    pass_number: numpy.ndarray,
) -> Iterator[Tuple[int, int]]:
    """Calculate the indexes of the start and stop of each half-orbit.

    Args:
        pass_number: Pass numbers.
    Returns:
        Iterator of start and stop indexes.
    """
    assert pass_number.shape == cycle_number.shape
    pass_idx = numpy.where(numpy.roll(pass_number, 1) != pass_number)[0]
    cycle_idx = numpy.where(numpy.roll(cycle_number, 1) != cycle_number)[0]

    half_orbit = numpy.unique(
        numpy.concatenate(
            (pass_idx, cycle_idx, numpy.array([pass_number.size],
                                              dtype='int64'))))
    del pass_idx, cycle_idx

    yield from tuple(zip(half_orbit[:-1], half_orbit[1:]))


# %%
# Now we will compute these constant parts from a dataset contained in a
# partition.
def _half_orbit(
    ds: zcollection.Dataset,
    *args,
    **kwargs,
) -> numpy.ndarray:
    """Return the indexes of the start and stop of each half-orbit.

    Args:
        ds: Datasets stored in a partition to be indexed.
    Returns:
        Dictionary of start and stop indexes for each half-orbit.
    """
    pass_number_varname = kwargs.pop('pass_number', 'pass_number')
    cycle_number_varname = kwargs.pop('cycle_number', 'cycle_number')
    pass_number = ds.variables[pass_number_varname].values
    cycle_number = ds.variables[cycle_number_varname].values

    generator = ((
        i0,
        i1,
        cycle_number[i0],
        pass_number[i0],
    ) for i0, i1 in split_half_orbit(cycle_number, pass_number))

    return numpy.fromiter(generator, numpy.dtype(HalfOrbitIndexer.dtype()))


# %%
# Finally, we implement our indexing class. The base class
# (:py:class:`zcollection.indexing.Indexer<zcollection.indexing.abc.Indexer>`)
# implements the index update and the associated queries.
class HalfOrbitIndexer(zcollection.indexing.Indexer):
    """Index collection by half-orbit."""
    #: Column name of the cycle number.
    CYCLE_NUMBER = 'cycle_number'

    #: Column name of the pass number.
    PASS_NUMBER = 'pass_number'

    @classmethod
    def dtype(cls, /, **kwargs) -> List[Tuple[str, str]]:
        """Return the columns of the index.

        Returns:
            A tuple of (name, type) pairs.
        """
        return super().dtype() + [
            (cls.CYCLE_NUMBER, 'uint16'),
            (cls.PASS_NUMBER, 'uint16'),
        ]

    @classmethod
    def create(
        cls,
        path: Union[pathlib.Path, str],
        ds: zcollection.Collection,
        filesystem: Optional[fsspec.AbstractFileSystem] = None,
        **kwargs,
    ) -> 'HalfOrbitIndexer':
        """Create a new index.

        Args:
            path: The path to the index.
            ds: The collection to be indexed.
            filesystem: The filesystem to use.
        Returns:
            The created index.
        """
        return super()._create(path,
                               ds,
                               meta=dict(attribute=b'value'),
                               filesystem=filesystem)  # type: ignore

    def update(
        self,
        ds: zcollection.Collection,
        partition_size: Optional[int] = None,
        npartitions: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Update the index.

        Args:
            ds: New data stored in the collection to be indexed.
            partition_size: The length of each bag partition.
            npartitions: The number of desired bag partitions.
            cycle_number: The name of the cycle number variable stored in the
                collection. Defaults to "cycle_number".
            pass_number: The name of the pass number variable stored in the
                collection. Defaults to "pass_number".
        """
        super()._update(ds, _half_orbit, partition_size, npartitions, **kwargs)


# %%
# Using the index
# ---------------
#
# Now we can create our index and fill it.
indexer = HalfOrbitIndexer.create('/index.parquet', collection, filesystem=fs)
indexer.update(collection)

# The following command allows us to view the information stored in our index:
# the first and last indexes of the partition associated with the registered
# half-orbit number and the identifier of the indexed partition.
indexer.table.to_pandas()

# %%
# This index can now be used to load a part of a collection.
selection = collection.load(indexer=indexer.query(dict(pass_number=[1, 2])))
assert selection is not None
selection.to_xarray().compute()

# %%
# Close the local cluster to avoid printing warning messages in the other
# examples.
client.close()
cluster.close()
