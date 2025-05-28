# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test partitioning by grouped sequence.
======================================
"""
from __future__ import annotations

from collections.abc import Iterator
import pickle

import numpy
import numpy as np
import pytest
import xarray

from . import data
from .. import GroupedSequence, get_codecs
from ... import dataset


def test_construction() -> None:
    """Test the GroupedSequence constructor."""
    assert isinstance(GroupedSequence(variables=('a', 'b'), size=2),
                      GroupedSequence)
    assert len(GroupedSequence(variables=('a', 'b'), size=2)) == 2

    with pytest.raises(ValueError, match='Data type must be one of'):
        GroupedSequence(variables=('a', 'b'), size=2,
                        dtype=(0, ))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match='variables must not be empty'):
        GroupedSequence(variables=(), size=2, dtype=())

    with pytest.raises(ValueError, match='must be at least 2'):
        GroupedSequence(variables=('a', 'b'), size=0, dtype=())

    with pytest.raises(ValueError, match='must be at least 2'):
        GroupedSequence(variables=('a', 'b'), size=1, dtype=())

    with pytest.raises(ValueError, match='Data type must be one of'):
        GroupedSequence(variables=('a', 'b'), size=2, dtype=('c', 'd'))

    with pytest.raises(ValueError, match='Data type must be one of'):
        GroupedSequence(variables=('a', 'b'),
                        size=2,
                        dtype=('float32', 'int32'))

    with pytest.raises(TypeError):
        GroupedSequence(variables=('a', 'b'), size=2, dtype='int32')

    partitioning = GroupedSequence(variables=('a', 'b'), size=2)
    partition_keys = partitioning.parse('a=1/b=2')

    assert partitioning.encode(partition_keys) == (1, 2)

    with pytest.raises(ValueError,
                       match='Partition is not driven by this instance'):
        partitioning.encode((('A', 1), ('b', 2)))

    assert partitioning.decode((1, 2)) == (('a', 1), ('b', 2))
    assert partition_keys == (('a', 1), ('b', 2))

    with pytest.raises(ValueError, match='invalid literal for int()'):
        partitioning.parse('a=1/b=2/c=3')

    with pytest.raises(ValueError,
                       match='Partition is not driven by this instance'):
        partitioning.parse('field=1')


@pytest.mark.parametrize('part_size', [2, 3, 4, 5])
def test_split_dataset(part_size: int) -> None:
    """Test the split_dataset method."""
    repeatability = 5
    xds = data.create_test_sequence(
        repeatability=repeatability,
        number_of_measures=20,
        number_of_cycles=10,
    )
    partitioning = GroupedSequence(
        variables=('cycle_number', 'pass_number'),
        size=part_size,
    )

    cycle_number = 1
    pass_number = 0

    assert partitioning.dtype() == (
        ('cycle_number', 'int64'),
        ('pass_number', 'int64'),
    )

    # Build the test dataset
    zds = dataset.Dataset.from_xarray(xds, delayed=False)

    iterator = partitioning.split_dataset(zds=zds, axis='num_lines')
    assert isinstance(iterator, Iterator)

    for partition, indexer in iterator:
        subset = zds.isel(indexer)
        expected = (f'cycle_number={cycle_number}',
                    f'pass_number={pass_number}')
        assert expected == partition
        assert numpy.all(
            xds.where((xds.cycle_number == cycle_number)
                      & ((xds.pass_number // part_size *
                          part_size) == pass_number),
                      drop=True).observation ==
            subset.variables['observation'].array)

        partition_keys = partitioning.parse('/'.join(partition))

        assert partition_keys == (('cycle_number', cycle_number),
                                  ('pass_number', pass_number))
        assert partitioning.decode(
            partitioning.encode(partition_keys)) == partition_keys
        assert partitioning.join(partition_keys, '/') == '/'.join(partition)

        pass_number += part_size
        if pass_number > repeatability:
            pass_number = 0
            cycle_number += 1

    xds['cycle_number'] = xarray.DataArray(numpy.array(
        [xds['cycle_number'].values] * 2).T,
                                           dims=('num_lines', 'nump_pixels'))
    zds = dataset.Dataset.from_xarray(xds, delayed=False)
    with pytest.raises(ValueError, match='must be a one-dimensional array'):
        list(partitioning.split_dataset(zds, 'num_lines'))


VARIABLES_DTYPE_TEST_SET = [(('a', ), None), (('a', ), ('uint8', )),
                            (('a', 'b'), None),
                            (('a', 'b'), ('int8', 'int16'))]


@pytest.mark.parametrize(('variables', 'dtype'), VARIABLES_DTYPE_TEST_SET)
@pytest.mark.parametrize('part_size', [2, 3, 4, 5])
def test_config(variables, dtype, part_size) -> None:
    """Test the configuration of the GroupedSequence class."""
    partitioning = GroupedSequence(variables=variables,
                                   size=part_size,
                                   dtype=dtype)

    config = partitioning.get_config()
    other = get_codecs(config)  # type: ignore[assignment]

    assert isinstance(other, GroupedSequence)
    assert other.dtype() == partitioning.dtype()


@pytest.mark.parametrize(('variables', 'dtype'), VARIABLES_DTYPE_TEST_SET)
@pytest.mark.parametrize('part_size', [2, 3, 4, 5])
def test_pickle(variables, dtype, part_size) -> None:
    """Test the pickling of the GroupedSequence class."""
    partitioning = GroupedSequence(variables=variables,
                                   size=part_size,
                                   dtype=dtype)

    other = pickle.loads(pickle.dumps(partitioning))

    assert isinstance(other, GroupedSequence)
    assert other.dtype() == partitioning.dtype()


@pytest.mark.parametrize('start', [-10, 0, 10, 100])
def test_splits(start):
    """Test the _split method with different sets of parameters."""
    v1 = 'v1'
    v2 = 'v2'

    partitioning = GroupedSequence(variables=[v1, v2], size=5, start=start)

    vdata = {
        v1: np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2]),
        v2: np.array([0, 1, 5, 15, 50, 100, 102, 0, 2, 4]) + start
    }
    expected_groups = [((1, 0 + start), slice(0, 2)),
                       ((1, 5 + start), slice(2, 3)),
                       ((1, 15 + start), slice(3, 4)),
                       ((1, 50 + start), slice(4, 5)),
                       ((1, 100 + start), slice(5, 7)),
                       ((2, 0 + start), slice(7, 10))]

    split_data = partitioning._split(variables=vdata)

    for split, res in zip(split_data, expected_groups):
        s = split[1]
        values = split[0]

        assert s == res[1]
        assert values[0][1] == res[0][0]
        assert values[1][1] == res[0][1]
