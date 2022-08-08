# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test partitioning by sequence.
==============================
"""
from typing import Dict, Iterator
import pickle

import dask.array
import numpy
import pytest
import xarray

from . import data
from .. import Sequence, get_codecs
from ... import dataset
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster

# pylint: enable=unused-import # Need to import for fixtures


def test_construction():
    """Test the sequence constructor."""
    assert isinstance(Sequence(('a', 'b')), Sequence)
    assert len(Sequence(('a', 'b'))) == 2
    with pytest.raises(ValueError):
        Sequence(('a', 'b'), (0, ))  # type: ignore
    with pytest.raises(ValueError):
        Sequence((), ())
    with pytest.raises(ValueError):
        Sequence(('a', 'b'), dtype=('c', 'd'))
    with pytest.raises(ValueError):
        Sequence(('a', 'b'), dtype=('float32', 'int32'))
    with pytest.raises(TypeError):
        Sequence(('a', 'b'), dtype=('int32'))
    partitioning = Sequence(('a', 'b'))
    partition_keys = partitioning.parse('a=1/b=2')
    assert partitioning.encode(partition_keys) == (1, 2)
    with pytest.raises(ValueError):
        partitioning.encode((('A', 1), ('b', 2)))
    assert partitioning.decode((1, 2)) == (('a', 1), ('b', 2))
    assert partition_keys == (('a', 1), ('b', 2))
    with pytest.raises(ValueError):
        partitioning.parse('a=1/b=2/c=3')
    with pytest.raises(ValueError):
        partitioning.parse('field=1')


def test_split_dataset(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the split_dataset method."""
    repeatability = 5
    xr_ds = data.create_test_sequence(repeatability, 20, 10)
    partitioning = Sequence(('cycle_number', 'pass_number'))

    cycle_number = 1
    pass_number = 1

    assert partitioning.dtype() == (
        ('cycle_number', 'int64'),
        ('pass_number', 'int64'),
    )

    # Build the test dataset
    ds = dataset.Dataset.from_xarray(xr_ds)

    iterator = partitioning.split_dataset(ds, 'num_lines')
    assert isinstance(iterator, Iterator)

    for partition, indexer in iterator:
        subset = ds.isel(indexer)
        expected = (f'cycle_number={cycle_number}',
                    f'pass_number={pass_number}')
        assert expected == partition
        assert numpy.all(
            xr_ds.where((xr_ds.cycle_number == cycle_number)
                        & (xr_ds.pass_number == pass_number),
                        drop=True).observation ==
            subset.variables['observation'].array)

        partition_keys = partitioning.parse('/'.join(partition))
        assert partition_keys == (('cycle_number', cycle_number),
                                  ('pass_number', pass_number))
        assert partitioning.decode(
            partitioning.encode(partition_keys)) == partition_keys
        assert partitioning.join(partition_keys, '/') == '/'.join(partition)

        pass_number += 1
        if pass_number > repeatability:
            pass_number = 1
            cycle_number += 1

    xr_ds['cycle_number'] = xarray.DataArray(numpy.array(
        [xr_ds['cycle_number'].values] * 2).T,
                                             dims=('num_lines', 'nump_pixels'))
    ds = dataset.Dataset.from_xarray(xr_ds)
    with pytest.raises(ValueError):
        list(partitioning.split_dataset(ds, 'num_lines'))


def test_config():
    """Test the configuration of the Sequence class."""
    partitioning = Sequence(('cycle_number', 'pass_number'))
    config = partitioning.get_config()
    partitioning = get_codecs(config)
    assert isinstance(partitioning, Sequence)


def test_pickle():
    """Test the pickling of the Date class."""
    partitioning = Sequence(('cycle_number', 'pass_number'))
    other = pickle.loads(pickle.dumps(partitioning))
    assert isinstance(other, Sequence)
    assert other.variables == ('cycle_number', 'pass_number')


# pylint: disable=protected-access
def test_multiple_sequence(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the creation of a sequence with multiple variables."""
    arrays = dict(_a=numpy.array([], dtype='i8'),
                  _b=numpy.array([], dtype='i8'),
                  _c=numpy.array([], dtype='i8'))
    for _a in range(5):
        for _b in range(5):
            arrays['_a'] = numpy.concatenate(
                (arrays['_a'], numpy.full((5, ), _a, dtype='i8')))
            arrays['_b'] = numpy.concatenate(
                (arrays['_b'], numpy.full((5, ), _b, dtype='i8')))
            arrays['_c'] = numpy.concatenate(
                (arrays['_c'], numpy.arange(5, dtype='i8')))
    partitioning = Sequence(('_a', '_b', '_c'))
    variables: Dict[str, dask.array.Array] = dict(
        _a=dask.array.from_array(arrays['_a'], chunks=(10, )),  # type: ignore
        _b=dask.array.from_array(arrays['_b'], chunks=(10, )),  # type: ignore
        _c=dask.array.from_array(arrays['_c'], chunks=(10, )))  # type: ignore
    _a = 0
    _b = 0
    _c = 0
    for ix, item in enumerate(partitioning._split(variables)):
        assert item[0] == (('_a', _a), ('_b', _b), ('_c', _c))
        _c += 1
        if _c > 4:
            _c = 0
            _b += 1
        if _b > 4:
            _b = 0
            _a += 1
        assert item[1] == slice(ix, ix + 1)

    numpy.random.shuffle(arrays['_c'])
    variables['_c'] = dask.array.from_array(arrays['_c'],
                                            chunks=(10, ))  # type: ignore

    with pytest.raises(ValueError):
        list(partitioning._split(variables))

    del variables['_c']
    del variables['_b']
    partitioning = Sequence(('_a', '_b', '_c'))

    _a = 0
    for ix, item in enumerate(partitioning._split(variables)):
        assert item[0] == (('_a', _a), )
        _a += 1
        assert item[1] == slice(ix * 25, ix * 25 + 25)
    # pylint: enable=protected-access


def test_values_must_be_integer(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test that the values must be integer."""
    values = numpy.arange(0, 100, dtype='f8')
    partitioning = Sequence(('values', ))
    # pylint: disable=protected-access
    with pytest.raises(TypeError):
        list(partitioning._split({'values': dask.array.from_array(values)}))
    # pylint: enable=protected-access
