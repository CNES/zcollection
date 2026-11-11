# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Overlap windows whose selected variables omit the partitioned dimension."""
from __future__ import annotations

import fsspec
import numpy
import pytest
import zarr

from .. import (
    convenience,
    dataset,
    meta,
    partitioning,
    storage,
    store,
    variable,
    view,
)
from ..variable.abc import read_dimensions
from .cluster import dask_client, dask_cluster  # noqa: F401
from .fs import local_fs  # noqa: F401

TARGETS = [(kind, 'local_fs')
           for kind in ('collection', 'view', 'reference_view')]
UPDATE_TARGETS = [item for item in TARGETS if item[0] != 'reference_view']
SLICES = {1: slice(0, 3), 2: slice(3, 5), 3: slice(2, 6)}
WINDOW_SIZES = {1: 5, 2: 9, 3: 6}


@pytest.fixture
def overlap_target(request):
    """Build partitions of three, two and four rows, with an immutable input."""
    kind, fs_name = request.param
    fs = request.getfixturevalue(fs_name)
    zds = dataset.Dataset([
        variable.Array('key', numpy.repeat([1, 2, 3], [3, 2, 4]), ('row', )),
        variable.Array('values', numpy.arange(9), ('row', )),
        variable.Array('result', numpy.zeros(9, dtype='int64'), ('row', )),
        variable.Array('constant', numpy.array([7, 8]), ('column', )),
    ],
                          delayed=False)
    zcol = convenience.create_collection('key',
                                         zds,
                                         partitioning.Sequence(('key', )),
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
    if kind == 'view':
        zview.add_variable(meta.VariableSchema('derived',
                                               'int64',
                                               dimensions=('row', ),
                                               fill_value=-1),
                           distributed=False)
    return zview


@pytest.mark.parametrize(('overlap_target', 'depth'), [
    pytest.param(('collection', 'local_fs'), 1, id='collection'),
    pytest.param(('view', 'local_fs'), 1, id='view'),
    pytest.param(('reference_view', 'local_fs'), 1, id='reference-only'),
    pytest.param(('collection', 'local_fs'), 2, id='collection-depth-two'),
    pytest.param(('view', 'local_fs'), 2, id='view-depth-two'),
    pytest.param(('collection', 's3_fs'), 1, id='collection-s3'),
    pytest.param(('reference_view', 's3_fs'), 1, id='reference-only-s3'),
],
                         indirect=['overlap_target'])
@pytest.mark.usefixtures('dask_client')
def test_map_overlap_immutable_only(overlap_target, depth):
    """Recover central slices when selected inputs omit the partitioned axis."""

    def callback(zds, *, partition_info):
        assert set(zds.variables) == {'constant'}
        assert 'row' not in zds.dimensions
        assert partition_info.dimension == 'row'
        return partition_info.indices, int(zds['constant'].values.sum())

    result = overlap_target.map_overlap(callback,
                                        depth=depth,
                                        selected_variables=('constant', ),
                                        delayed=False).compute()
    expected = SLICES if depth == 1 else {
        1: slice(0, 3),
        2: slice(3, 5),
        3: slice(5, 9)
    }
    assert len(result) == 3
    for key, (indices, value) in result:
        assert indices == expected[dict(key)['key']]
        assert value == 15


@pytest.mark.parametrize('overlap_target', UPDATE_TARGETS, indirect=True)
@pytest.mark.parametrize(('delayed', 'distributed', 'trim'), [
    pytest.param(False, False, True, id='local-trim'),
    pytest.param(False, False, False, id='local-pretrimmed'),
    pytest.param(True, True, True, id='distributed-delayed'),
])
@pytest.mark.usefixtures('dask_client')
def test_update_overlap_immutable_only(overlap_target, delayed, distributed,
                                       trim):
    """Selected constants can produce partitioned outputs with either trim mode."""
    output = 'derived' if isinstance(overlap_target, view.View) else 'result'

    def callback(zds, *, partition_info):
        assert set(zds.variables) == {'constant'}
        assert 'row' not in zds.dimensions
        key = partition_info.partition['key']
        indices = partition_info.indices
        assert indices == SLICES[key]
        values = numpy.arange(WINDOW_SIZES[key]) + int(
            zds['constant'].values.sum())
        return {output: values if trim else values[indices]}

    overlap_target.update(callback,
                          variables=(output, ),
                          selected_variables=('constant', ),
                          depth=1,
                          delayed=delayed,
                          distributed=distributed,
                          trim=trim)
    result = overlap_target.load(selected_variables=(output, ),
                                 delayed=False,
                                 distributed=False)
    numpy.testing.assert_array_equal(result[output].values,
                                     [15, 16, 17, 18, 19, 17, 18, 19, 20])


@pytest.mark.parametrize('overlap_target', UPDATE_TARGETS, indirect=True)
def test_overlap_size_uses_only_metadata(overlap_target, monkeypatch):
    """Recovering the row count never reads an omitted variable's chunks."""
    getitem = zarr.Array.__getitem__

    def read(array, selection):
        assert 'row' not in read_dimensions(array)
        return getitem(array, selection)

    monkeypatch.setattr(zarr.Array, '__getitem__', read)
    result = overlap_target.map_overlap(
        lambda zds: int(zds['constant'].values.sum()),
        depth=1,
        selected_variables=('constant', ),
        delayed=False).compute(scheduler='synchronous')
    assert [value for _, value in result] == [15, 15, 15]


def test_partition_size_uses_loaded_dimension(monkeypatch):
    """An available dimension needs no axis metadata read."""

    def unexpected(*args, **kwargs):
        pytest.fail('The selected dimension must not require opening an axis')

    monkeypatch.setattr(store, 'open_array', unexpected)
    assert storage.partition_dimension_size(
        dimensions={'row': 3},
        dim='row',
        axis='unused',
        fs=fsspec.filesystem('memory')) == 3


@pytest.mark.parametrize('overlap_target', UPDATE_TARGETS, indirect=True)
def test_overlap_immutable_only_skips_absent_neighbor(overlap_target):
    """An absent neighbor contributes no rows to the central partition's slice."""
    reference = overlap_target.view_ref if isinstance(
        overlap_target, view.View) else overlap_target
    reference.fs.rm(next(path for path in reference.partitions()
                         if path.endswith('key=2')),
                    recursive=True)

    def callback(zds, *, partition_info):
        return partition_info.indices

    result = overlap_target.map_overlap(
        callback, depth=1, selected_variables=('constant', ),
        delayed=False).compute(scheduler='synchronous')
    assert [(dict(key)['key'], indices)
            for key, indices in result] == [(1, slice(0, 3)), (3, slice(0, 4))]
