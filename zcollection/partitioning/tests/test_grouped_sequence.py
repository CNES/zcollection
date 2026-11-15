# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Grouped sequence partitioning tests
===================================
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

    with pytest.raises(ValueError, match=r'invalid literal for int()'):
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
                                   start=7,
                                   dtype=dtype)

    config = partitioning.get_config()
    other = get_codecs(config)  # type: ignore[assignment]

    assert isinstance(other, GroupedSequence)
    assert other.dtype() == partitioning.dtype()
    assert other.variables == partitioning.variables
    assert other.size == part_size
    assert other.start == 7


def test_pickle() -> None:
    """Test pickling a GroupedSequence."""
    partitioning = GroupedSequence(variables=('a', 'b'),
                                   size=3,
                                   start=7,
                                   dtype=('int8', 'int16'))

    other = pickle.loads(pickle.dumps(partitioning))

    assert isinstance(other, GroupedSequence)
    assert other.get_config() == partitioning.get_config()
    assert other.parse('a=1/b=7') == (('a', 1), ('b', 7))


@pytest.mark.parametrize(('start', 'expected'), [(0, [0, 5, 10, 15]),
                                                 (3, [-2, 3, 8, 13])])
def test_start_aligns_the_groups(start, expected) -> None:
    """The groups are the multiples of the size offset by ``start``."""
    partitioning = GroupedSequence(variables=('v1', 'v2'), size=5, start=start)
    values = {
        'v1': np.ones(4, dtype='int64'),
        'v2': np.array([2, 7, 12, 17]),
    }

    groups = [int(fields[1][1]) for fields, _ in partitioning._split(values)]

    assert groups == expected


@pytest.mark.parametrize('delayed', [False, True])
@pytest.mark.parametrize(('start', 'expected'), [(10, [0, 5, 10, 15]),
                                                 (3, [-2, 3, 8, 13])])
def test_unsigned_input_below_start(delayed, start, expected) -> None:
    """Unsigned input preserves group bounds, other keys and row slices."""
    constructor = dataset.DelayedArray if delayed else dataset.Array
    zds = dataset.Dataset([
        constructor('prefix', numpy.ones(7, dtype='uint8'), ('row', )),
        constructor('key', numpy.array([2, 2, 7, 7, 12, 12, 17],
                                       dtype='uint8'), ('row', )),
    ],
                          delayed=delayed)
    handler = GroupedSequence(('prefix', 'key'), size=5, start=start)

    partitions = list(handler.index_dataset(zds))

    assert [fields for fields, _ in partitions
            ] == [(('prefix', 1), ('key', lower)) for lower in expected]
    assert [indexer for _, indexer in partitions
            ] == [slice(0, 2),
                  slice(2, 4),
                  slice(4, 6),
                  slice(6, 7)]


@pytest.mark.parametrize('delayed', [False, True])
@pytest.mark.parametrize(('input_dtype', 'values', 'key_dtype', 'lower_bound'),
                         [
                             ('uint64', [0, 1, 2], 'uint64', -2),
                             ('int8', [-128, -127, -126], 'int8', -132),
                             ('uint16', [258, 259], 'uint8', 258),
                         ])
def test_group_bound_must_fit_key_dtype(delayed, input_dtype, values,
                                        key_dtype, lower_bound) -> None:
    """Splitting rejects group bounds outside the declared key dtype."""
    constructor = dataset.DelayedArray if delayed else dataset.Array
    zds = dataset.Dataset([
        constructor('key', numpy.array(values, dtype=input_dtype), ('row', )),
    ],
                          delayed=delayed)
    handler = GroupedSequence(('key', ), size=5, start=3, dtype=(key_dtype, ))

    with pytest.raises(
            ValueError,
            match=f"lower bound {lower_bound} of 'key'.*{key_dtype}"):
        list(handler.index_dataset(zds))


@pytest.mark.parametrize('delayed', [False, True])
def test_unsigned_key_with_nonzero_alignment(delayed) -> None:
    """Unsigned keys accept an aligned bound that fits their dtype."""
    constructor = dataset.DelayedArray if delayed else dataset.Array
    zds = dataset.Dataset([
        constructor('key', numpy.array([3, 4], dtype='uint64'), ('row', )),
    ],
                          delayed=delayed)
    handler = GroupedSequence(('key', ), size=5, start=3, dtype=('uint64', ))

    assert list(handler.index_dataset(zds)) == [((('key', 3), ), slice(0, 2))]


@pytest.mark.parametrize('delayed', [False, True])
def test_uint64_group_bounds_keep_their_precision(delayed) -> None:
    """Large unsigned keys retain exact group bounds and their row slices."""
    constructor = dataset.DelayedArray if delayed else dataset.Array
    values = numpy.array([0, 1, 2**63 + 2, 2**63 + 3, 2**64 - 2, 2**64 - 1],
                         dtype='uint64')
    zds = dataset.Dataset([constructor('key', values, ('row', ))],
                          delayed=delayed)
    handler = GroupedSequence(('key', ), size=2, start=10, dtype=('uint64', ))

    partitions = list(handler.index_dataset(zds))

    assert [fields for fields, _ in partitions] == [(('key', 0), ),
                                                    (('key', 2**63 + 2), ),
                                                    (('key', 2**64 - 2), )]
    assert [indexer for _, indexer in partitions
            ] == [slice(0, 2), slice(2, 4),
                  slice(4, 6)]


@pytest.mark.parametrize('start', [-10, 0, 10, 100])
def test_splits(start):
    """Test GroupedSequence._split with different configurations."""
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

    for split, res in zip(split_data, expected_groups, strict=True):
        s = split[1]
        values = split[0]

        assert s == res[1]
        assert values[0][1] == res[0][0]
        assert values[1][1] == res[0][1]
