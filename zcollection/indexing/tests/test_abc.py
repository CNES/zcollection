# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test the base class for indexing."""
from typing import Iterator, List, Optional, Tuple, Union
import pathlib

import fsspec
import numpy
import pyarrow
import pytest

from .. import abc
from ... import collection, convenience, dataset, partitioning
from ...partitioning.tests import data
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ...tests.fs import local_fs
# pylint: enable=unused-import
from ...typing import NDArray


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


# pylint: disable=unused-argument,invalid-name
# The signature of the function must follow the signature of
# zcollection.PartitionCallable
def _half_orbit(
    ds: dataset.Dataset,
    *args,
    **kwargs,
) -> NDArray:
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

    return numpy.fromiter(  # type: ignore
        generator, numpy.dtype(HalfOrbitIndexer.dtype()))


class HalfOrbitIndexer(abc.Indexer):
    """Index SWOT collection by half-orbit."""
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
        ds: collection.Collection,
        *,
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
        ds: collection.Collection,
        *,
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


def test_indexer(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
        local_fs,  # pylint: disable=redefined-outer-name
):
    """Test the base class of the indexer."""
    ds = dataset.Dataset.from_xarray(data.create_test_sequence(5, 20, 10))

    zcollection = convenience.create_collection(
        'time',
        ds,
        partitioning.Date(('time', ), 'M'),
        partition_base_dir=str(local_fs.collection),
        filesystem=local_fs.fs)
    zcollection.insert(ds, merge_callable=collection.merging.merge_time_series)

    indexer = HalfOrbitIndexer.create(str(
        local_fs.collection.joinpath('index.parquet')),
                                      zcollection,
                                      filesystem=local_fs.fs)

    # Index not yet created
    with pytest.raises(ValueError):
        _ = indexer.table

    assert indexer.dtype() == [('start', 'int64'), ('stop', 'int64'),
                               ('cycle_number', 'uint16'),
                               ('pass_number', 'uint16')]
    indexer.update(zcollection)
    assert isinstance(indexer.table, pyarrow.Table)

    selection = zcollection.load(indexer=indexer.query(dict(cycle_number=2)))
    assert selection is not None
    assert set(selection.variables['cycle_number'].values) == {2}

    with pytest.raises(ValueError):
        indexer.query(dict(cycle_number=3), logical_op='X')

    with pytest.raises(ValueError):
        indexer.query(dict(X=3))

    # Updating the index should not change the indexer.
    indexer.update(zcollection)
    other = zcollection.load(indexer=indexer.query(dict(cycle_number=2)))
    assert other is not None
    assert numpy.all(
        other['observation'].values == selection['observation'].values)

    selection = zcollection.load(
        indexer=indexer.query(dict(cycle_number=[2, 4])))
    assert selection is not None
    assert set(selection.variables['cycle_number'].values) == {2, 4}

    selection = zcollection.load(
        indexer=indexer.query(dict(cycle_number=[2, 4], pass_number=1)))
    assert selection is not None
    assert set(selection.variables['cycle_number'].values) == {2, 4}
    assert set(selection.variables['pass_number'].values) == {1}

    selection = zcollection.load(
        indexer=indexer.query(dict(cycle_number=[2, 4], pass_number=[1, 5])))
    assert selection is not None
    assert set(selection.variables['cycle_number'].values) == {2, 4}
    assert set(selection.variables['pass_number'].values) == {1, 5}

    indexer = HalfOrbitIndexer.open(str(
        local_fs.collection.joinpath('index.parquet')),
                                    filesystem=local_fs.fs)
    assert indexer.meta == dict(attribute=b'value')
    selection = zcollection.load(
        indexer=indexer.query(dict(cycle_number=[2, 4], pass_number=[1, 5])))
    assert selection is not None
    assert set(selection.variables['cycle_number'].values) == {2, 4}
    assert set(selection.variables['pass_number'].values) == {1, 5}

    indices = tuple(
        indexer.query(dict(cycle_number=[2, 4]), only_partition_keys=False))
    assert tuple(item[0] for item in indices[0][0]) == (
        'cycle_number',
        'pass_number',
        'year',
        'month',
    )

    indexer = abc.Indexer('', filesystem=fsspec.filesystem('memory'))
    assert indexer.query(dict(cycle_number=[2, 4])) == tuple()
