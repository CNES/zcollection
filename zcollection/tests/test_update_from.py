# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests of `update_from` on collections and views."""
from __future__ import annotations

import numpy
import pytest

from . import reference
from .. import collection, convenience, dataset, meta, partitioning, store, view
from .cluster import dask_client, dask_cluster  # noqa: F401
from .fs import local_fs, s3, s3_base, s3_fs  # noqa: F401


@pytest.fixture
def update_from_target(request):
    """Build a collection of three partitions of four rows, or a view of it
    with the variable `derived`."""
    kind, fs_name = request.param
    fs = request.getfixturevalue(fs_name)
    zds = reference.make_dataset()
    zcol = convenience.create_collection('time',
                                         zds,
                                         reference.make_partitioning(),
                                         str(fs.collection),
                                         filesystem=fs.fs)
    zcol.insert(zds, distributed=False)
    if kind == 'collection':
        return zcol
    zview = convenience.create_view(str(fs.view),
                                    view.ViewReference(str(fs.collection),
                                                       fs.fs),
                                    filesystem=fs.fs,
                                    distributed=False)
    zview.add_variable(meta.VariableSchema('derived',
                                           'float64',
                                           dimensions=(reference.DIMENSION, ),
                                           fill_value=numpy.nan),
                       distributed=False)
    return zview


TARGETS = [(kind, fs) for kind in ('collection', 'view')
           for fs in ('local_fs', 's3_fs')]
LOCAL_TARGETS = [(kind, 'local_fs') for kind in ('collection', 'view')]


def output_name(target):
    """Return the name of the variable the tests update."""
    return 'derived' if isinstance(target, view.View) else 'plain'


def carrying(target):
    """Build a dataset of `time` and new values of the updated variable."""
    zds = target.load(delayed=False, distributed=False)
    name = output_name(target)
    return dataset.Dataset([
        zds['time'],
        dataset.Array(name,
                      numpy.full(zds[name].values.shape, 7, zds[name].dtype),
                      zds[name].dimensions),
    ])


@pytest.mark.parametrize('update_from_target', TARGETS, indirect=True)
def test_other_variables_keep_their_values(update_from_target) -> None:
    """The variables of the dataset take new values; the others keep theirs."""
    name = output_name(update_from_target)
    before = update_from_target.load(delayed=False, distributed=False)
    carried = carrying(update_from_target)

    updated = update_from_target.update_from(carried,
                                             align_on='time',
                                             distributed=False)

    assert len(updated) == 3
    after = update_from_target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(after[name].values, carried[name].values)
    for other in sorted(set(before.variables) - {name}):
        numpy.testing.assert_array_equal(after[other].values,
                                         before[other].values)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_a_dataset_without_time_is_refused(update_from_target) -> None:
    """A dataset without `time` is refused."""
    carried = carrying(update_from_target)

    with pytest.raises(ValueError, match='does not contain the variables'):
        update_from_target.update_from(carried.select_vars(
            [output_name(update_from_target)]),
                                       align_on='time',
                                       distributed=False)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_a_variable_the_schema_does_not_define_is_refused(
        update_from_target) -> None:
    """A variable the schema does not define is refused."""
    carried = carrying(update_from_target)
    carried.rename({output_name(update_from_target): 'absent'})

    with pytest.raises(ValueError,
                       match="does not define the variables 'absent'"):
        update_from_target.update_from(carried,
                                       align_on='time',
                                       distributed=False)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_a_partition_with_an_unmatched_row_keeps_its_values(
        update_from_target) -> None:
    """The partition with an unmatched row keeps its values; the ones before
    it are updated."""
    name = output_name(update_from_target)
    before = update_from_target.load(delayed=False, distributed=False)
    carried = carrying(update_from_target)
    # The last row stays in the last partition, where no such value exists.
    carried['time'].array[-1] = (carried['time'].array[-1] +
                                 reference.DELTA // 2)

    with pytest.raises(ValueError,
                       match="has no match for 1 of the 'time' values"):
        update_from_target.update_from(carried,
                                       align_on='time',
                                       distributed=False)

    # Four rows a day over three days: the last day is the failing partition.
    after = update_from_target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(after[name].values[:8],
                                     carried[name].values[:8])
    numpy.testing.assert_array_equal(after[name].values[8:],
                                     before[name].values[8:])


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_repeated_values_of_align_on_are_refused(update_from_target) -> None:
    """A partition where the values of `align_on` repeat is refused."""
    carried = carrying(update_from_target)
    # A view reads `align_on` from its reference collection.
    reference = update_from_target._reference
    partition = next(iter(reference.partitions()))
    path = reference.fs.sep.join((partition, 'time'))
    values = store.open_ndarray(reference.fs, path)
    values[1] = values[0]
    store.open_array(reference.fs, path, mode='r+')[...] = values

    with pytest.raises(ValueError, match="values of 'time' repeat"):
        update_from_target.update_from(carried,
                                       align_on='time',
                                       distributed=False)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_updating_a_partition_that_does_not_exist_is_refused(
        update_from_target) -> None:
    """A row in a partition that does not exist is refused before any write."""
    name = output_name(update_from_target)
    before = update_from_target.load(delayed=False, distributed=False)
    carried = carrying(update_from_target)
    carried['time'].array[-1] = (carried['time'].array[-1] +
                                 reference.DELTA * 4)

    with pytest.raises(ValueError, match='does not exist'):
        update_from_target.update_from(carried,
                                       align_on='time',
                                       distributed=False)

    after = update_from_target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(after[name].values, before[name].values)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.usefixtures('dask_client')
def test_variables_are_updated_over_a_cluster(update_from_target) -> None:
    """The variables are updated over a Dask cluster."""
    name = output_name(update_from_target)
    carried = carrying(update_from_target)

    update_from_target.update_from(carried, align_on='time')

    after = update_from_target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(after[name].values, carried[name].values)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_aligning_on_a_variable_the_reference_lacks_is_refused(
        update_from_target) -> None:
    """`align_on` must name a variable of the reference collection."""
    with pytest.raises(ValueError, match="has no variable 'absent'"):
        update_from_target.update_from(carrying(update_from_target),
                                       align_on='absent',
                                       distributed=False)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_aligning_on_a_variable_with_other_dimensions_is_refused(
        update_from_target) -> None:
    """`align_on` must have the partitioned dimension only."""
    with pytest.raises(ValueError, match='as its only dimension'):
        update_from_target.update_from(carrying(update_from_target),
                                       align_on='grid',
                                       distributed=False)


@pytest.mark.parametrize('kind', ['collection', 'view'])
def test_rows_sharing_a_key_align_on_another_variable(
        kind,
        local_fs,  # noqa: F811
) -> None:
    """`align_on` identifies rows that share a partitioning key."""
    keys = numpy.repeat([1, 2], 3)
    # The values of `index` are not sorted within a partition.
    index = numpy.array([12, 10, 11, 22, 20, 21])
    zds = dataset.Dataset([
        dataset.Array('key', keys, ('row', )),
        dataset.Array('index', index, ('row', )),
        dataset.Array('value', numpy.zeros(6, 'int64'), ('row', )),
    ],
                          delayed=False)
    zcol = convenience.create_collection('key',
                                         zds,
                                         partitioning.Sequence(('key', )),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs)
    zcol.insert(zds, distributed=False)
    target: collection.Collection | view.View = zcol
    name = 'value'
    if kind == 'view':
        target = convenience.create_view(str(local_fs.view),
                                         view.ViewReference(
                                             str(local_fs.collection),
                                             local_fs.fs),
                                         filesystem=local_fs.fs,
                                         distributed=False)
        name = 'derived'
        target.add_variable(meta.VariableSchema(name,
                                                'int64',
                                                dimensions=('row', ),
                                                fill_value=-1),
                            distributed=False)

    # The dataset lists the rows in another order than the partitions do.
    order = numpy.array([2, 0, 1, 5, 3, 4])
    carried = dataset.Dataset([
        dataset.Array('key', keys[order], ('row', )),
        dataset.Array('index', index[order], ('row', )),
        dataset.Array(name, index[order] * 100, ('row', )),
    ],
                              delayed=False)

    target.update_from(carried, align_on='index', distributed=False)

    after = target.load(delayed=False, distributed=False)
    assert after is not None
    numpy.testing.assert_array_equal(after[name].values,
                                     after['index'].values * 100)


@pytest.mark.parametrize('update_from_target', [('collection', 'local_fs')],
                         indirect=True)
def test_a_variable_without_the_partitioned_dimension_is_refused(
        update_from_target) -> None:
    """An immutable variable is refused before any write."""
    loaded = update_from_target.load(delayed=False, distributed=False)
    carried = dataset.Dataset([loaded['time'], loaded['pixel']])

    with pytest.raises(ValueError, match='does not have the dimension'):
        update_from_target.update_from(carried,
                                       align_on='time',
                                       distributed=False)


@pytest.mark.parametrize('update_from_target', LOCAL_TARGETS, indirect=True)
def test_repeated_values_of_align_on_in_the_dataset_are_refused(
        update_from_target) -> None:
    """Two rows of the dataset with the same value of `align_on` are
    refused."""
    name = output_name(update_from_target)
    before = update_from_target.load(delayed=False, distributed=False)
    carried = carrying(update_from_target)
    carried['time'].array[1] = carried['time'].array[0]

    with pytest.raises(ValueError, match='repeat in the dataset rows'):
        update_from_target.update_from(carried,
                                       align_on='time',
                                       distributed=False)

    after = update_from_target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(after[name].values[:4],
                                     before[name].values[:4])
