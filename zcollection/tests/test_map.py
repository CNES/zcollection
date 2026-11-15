# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test callback arguments, selections and errors in partition mapping."""
from __future__ import annotations

import dask.array
import dask.bag
import dask.distributed
import numpy
import pytest
import zarr
import zarr.storage

from .. import (
    PartitionInfo,
    convenience,
    dataset,
    meta,
    partitioning,
    storage,
    variable,
    view,
)
from .cluster import dask_client, dask_cluster  # noqa: F401
from .fixture import empty_target  # noqa: F401
from .fs import local_fs  # noqa: F401

TARGETS = [('collection', 'local_fs'), ('view', 'local_fs')]
STORAGE_TARGETS = [*TARGETS, ('collection', 's3_fs'), ('view', 's3_fs')]
METHODS = ['map', 'map_overlap']


@pytest.fixture
def mapping_target(request):
    """Build three partitions and a view with a variable of its own."""
    kind, fs_name = request.param
    fs = request.getfixturevalue(fs_name)
    values = numpy.array([10, 11, 20, 21, 30, 31])
    masked = numpy.ma.array(values,
                            mask=[False, True, False, False, True, False],
                            fill_value=-99)
    zds = dataset.Dataset([
        variable.Array('key', numpy.repeat([1, 2, 3], 2), ('row', )),
        variable.Array('x', values, ('row', )),
        variable.Array('masked', masked, ('row', ), fill_value=-99),
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
    zview.add_variable(meta.VariableSchema('derived',
                                           'int64',
                                           dimensions=('row', ),
                                           fill_value=None),
                       distributed=False)
    return zview


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('delayed', [False, True])
def test_mapping_arguments(mapping_target, method, delayed):
    """Mappings stay lazy and pass positional and keyword callback arguments."""
    calls = []

    def callback(zds, factor, shift, *, offset, partition_info):
        calls.append(None)
        assert set(zds.variables) == {'x'}
        assert isinstance(zds['x'].array,
                          dask.array.Array if delayed else numpy.ndarray)
        return zds['x'].values[
            partition_info.indices] * factor + shift + offset

    bag = getattr(mapping_target, method)(callback,
                                          2,
                                          3,
                                          offset=4,
                                          delayed=delayed,
                                          selected_variables=('x', ))
    assert not calls
    result = bag.compute(scheduler='synchronous')
    numpy.testing.assert_array_equal(
        numpy.concatenate([values for _, values in result]),
        numpy.array([10, 11, 20, 21, 30, 31]) * 2 + 7)


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize('method', METHODS)
def test_mapping_internal_keyword_names(mapping_target, method):
    """Callback keywords can use names from the mapping's internal wrapper."""
    payload = dict.fromkeys(('partition', '_partition', '_func', '_delayed',
                             '_depth', '_partitions', '_selected_variables',
                             '_wrapped_func', '_args', '_kwargs'), 'argument')

    def callback(zds, *, partition_info, **kwargs):
        assert kwargs == payload
        return zds['x'].values[partition_info.indices]

    result = getattr(mapping_target,
                     method)(callback,
                             delayed=False,
                             selected_variables=('x', ),
                             **payload).compute(scheduler='synchronous')
    numpy.testing.assert_array_equal(
        numpy.concatenate([values for _, values in result]),
        [10, 11, 20, 21, 30, 31])


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize(('method', 'expected'), [
    ('map', [[10, 11], [30, 31]]),
    ('map_overlap', [[10, 11, 20, 21], [20, 21, 30, 31]]),
])
def test_mapping_selected_partitions(mapping_target, method, expected):
    """Only the partitions named by selected_partitions are processed, and
    map_overlap loads their neighbors whether they are selected or not."""

    def callback(zds, *, partition_info):
        return zds['x'].values.tolist()

    result = getattr(mapping_target, method)(
        callback,
        delayed=False,
        selected_variables=('x', ),
        selected_partitions=['key=1',
                             'key=3']).compute(scheduler='synchronous')
    assert sorted(values for _, values in result) == expected


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
def test_a_selected_partition_that_does_not_exist_is_reported(
        mapping_target, caplog):
    """A selected path that names no partition is left out with a warning."""
    result = mapping_target.map(
        lambda zds: zds['x'].values.tolist(),
        delayed=False,
        selected_variables=('x', ),
        selected_partitions=['key=1',
                             'key=9']).compute(scheduler='synchronous')

    assert [values for _, values in result] == [[10, 11]]
    assert "'key=9'" in caplog.text


def test_a_selected_partition_the_view_does_not_hold_is_reported(
        local_fs, caplog):  # noqa: F811
    """A selected path that names a partition of the reference but not of
    the view is left out with a warning."""
    zds = dataset.Dataset([
        variable.Array('key', numpy.repeat([1, 2], 2), ('row', )),
        variable.Array('x', numpy.arange(4), ('row', )),
    ],
                          delayed=False)
    zcol = convenience.create_collection('key',
                                         zds,
                                         partitioning.Sequence(('key', )),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs)
    zcol.insert(zds, distributed=False)
    zview = convenience.create_view(str(local_fs.view),
                                    view.ViewReference(
                                        str(local_fs.collection), local_fs.fs),
                                    filesystem=local_fs.fs,
                                    filters='key == 1',
                                    distributed=False)

    assert [
        zview._relative_path(item)
        for item in zview.partitions(selected_partitions=['key=1', 'key=2'])
    ] == ['key=1']
    assert "'key=2'" in caplog.text

    caplog.clear()
    zview.add_variable(meta.VariableSchema('derived',
                                           'int64',
                                           dimensions=('row', ),
                                           fill_value=None),
                       distributed=False)
    assert zview.update(lambda zds: {'derived': zds['x'].values},
                        variables=['derived'],
                        selected_partitions=['key=1', 'key=2'],
                        delayed=False,
                        distributed=False) == ('key=1', )
    assert "'key=2'" in caplog.text


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.usefixtures('dask_client')
def test_mapping_scattered_argument(mapping_target):
    """A scattered argument is usable inside the callback."""
    coefficients = dask.distributed.get_client().scatter(numpy.array([2, 3]))

    def callback(zds, external):
        return zds['x'].values * external[0] + external[1]

    result = mapping_target.map(callback,
                                coefficients,
                                delayed=False,
                                filters='key == 2',
                                selected_variables=('x', )).compute()
    assert len(result) == 1
    numpy.testing.assert_array_equal(result[0][1], [43, 45])


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.usefixtures('dask_client')
def test_mapping_persisted_array_argument(mapping_target):
    """A persisted array remains a Dask array inside the callback."""
    coefficients = dask.distributed.get_client().persist(
        dask.array.from_array(numpy.array([2, 3]), chunks=1))

    def callback(zds, external):
        assert isinstance(external, dask.array.Array)
        return zds['x'].values * int(external[:1].compute()[0])

    result = mapping_target.map(callback,
                                coefficients,
                                delayed=False,
                                filters='key == 2',
                                selected_variables=('x', )).compute()
    assert len(result) == 1
    numpy.testing.assert_array_equal(result[0][1], [40, 42])


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.usefixtures('dask_client')
def test_mapping_keyword_array_matches_bag(mapping_target):
    """Named Dask arrays follow the caller's ``Bag.map`` semantics."""
    external = dask.array.from_array(numpy.array([2, 3]), chunks=1)

    def describe(external):
        return type(external).__name__, numpy.asarray(external).tolist()

    def callback(zds, *, external):
        return describe(external)

    native = dask.bag.from_sequence(
        [None]).map(lambda item, *, external: describe(external),
                    external=external).compute()
    result = mapping_target.map(callback,
                                external=external,
                                delayed=False,
                                filters='key == 2',
                                selected_variables=('x', )).compute()
    assert len(result) == 1
    assert result[0][1] == native[0]


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
def test_mapping_lazy_array_reads_requested_chunk(mapping_target, monkeypatch):
    """The callback can compute one chunk of a positional lazy array."""
    external_store = zarr.storage.MemoryStore()
    zarr.create_array(store=external_store,
                      data=numpy.arange(12),
                      chunks=(4, ),
                      chunk_key_encoding={
                          'name': 'default',
                          'configuration': {
                              'separator': '/'
                          }
                      })
    reads = []
    get = zarr.storage.MemoryStore.get

    async def observe(store, key, *args, **kwargs):
        if key.startswith('c/'):
            reads.append(key)
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(zarr.storage.MemoryStore, 'get', observe)
    external = dask.array.from_zarr(external_store)

    def callback(zds, external):
        assert isinstance(external, dask.array.Array)
        return external[5:7].compute(scheduler='synchronous')

    bag = mapping_target.map(callback,
                             external,
                             delayed=False,
                             filters='key == 2',
                             selected_variables=('x', ))
    assert reads == []
    result = bag.compute(scheduler='synchronous')
    assert len(result) == 1
    numpy.testing.assert_array_equal(result[0][1], [5, 6])
    assert reads == ['c/1']


@pytest.mark.parametrize('mapping_target', STORAGE_TARGETS, indirect=True)
@pytest.mark.parametrize('delayed', [False, True])
def test_mapping_masks_and_immutable(mapping_target, delayed):
    """Overlap loading preserves masks, immutable inputs and view variables."""
    selected = ['masked', 'constant']
    if isinstance(mapping_target, view.View):
        selected.append('derived')

    def callback(zds, *, partition_info):
        assert set(zds.variables) == set(selected)
        numpy.testing.assert_array_equal(zds['constant'].values, [7, 8])
        return zds['masked'].values[partition_info.indices]

    result = mapping_target.map_overlap(
        callback, delayed=delayed,
        selected_variables=selected).compute(scheduler='synchronous')
    assert len(result) == 3
    expected = numpy.ma.array([10, 11, 20, 21, 30, 31],
                              mask=[False, True, False, False, True, False])
    actual = numpy.ma.concatenate([values for _, values in result])
    numpy.testing.assert_array_equal(actual.mask, expected.mask)
    numpy.testing.assert_array_equal(actual.compressed(),
                                     expected.compressed())


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize(
    'error', [storage.AbsentPartition, FileNotFoundError, ValueError])
def test_mapping_callback_errors(mapping_target, method, error):
    """Callback exceptions propagate, including ``AbsentPartition``."""

    def callback(zds):
        raise error('callback failure')

    bag = getattr(mapping_target, method)(callback,
                                          delayed=False,
                                          selected_variables=('x', ))
    with pytest.raises(error, match='callback failure'):
        bag.compute(scheduler='synchronous')


@pytest.mark.parametrize('mapping_target', STORAGE_TARGETS, indirect=True)
@pytest.mark.parametrize('method', METHODS)
def test_mapping_skips_absent_source(mapping_target, method, caplog):
    """An absent source partition is skipped; its neighbors remain usable."""
    partitions = tuple(mapping_target.partitions())
    mapping_target.fs.rm(partitions[1], recursive=True)
    mapping_target.fs.invalidate_cache()

    def callback(zds, *, partition_info):
        return zds['x'].values[partition_info.indices]

    result = getattr(mapping_target, method)(
        callback, delayed=False,
        selected_variables=('x', )).compute(scheduler='synchronous')
    assert [key for key, _ in result] == [(('key', 1), ), (('key', 3), )]
    assert 'holds no partition' in caplog.text
    numpy.testing.assert_array_equal(
        numpy.concatenate([values for _, values in result]), [10, 11, 30, 31])


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize('method', METHODS)
def test_mapping_preserves_none_result(mapping_target, method):
    """A callback's None result remains paired with its partition scheme."""
    result = getattr(mapping_target,
                     method)(lambda zds: None,
                             delayed=False).compute(scheduler='synchronous')
    assert result == [((('key', n), ), None) for n in (1, 2, 3)]


def test_mapping_rejects_non_callable(
        empty_target,  # noqa: F811
        monkeypatch):
    """An invalid callback is rejected before selecting partitions."""
    monkeypatch.setattr(
        empty_target, 'partitions',
        lambda *args, **kwargs: pytest.fail('Partitions must not be selected'))
    with pytest.raises(TypeError):
        empty_target.map(42)


def test_mapping_rejects_negative_depth(
        empty_target,  # noqa: F811
        monkeypatch):
    """A negative depth is rejected before selecting partitions."""
    monkeypatch.setattr(
        empty_target, 'partitions',
        lambda *args, **kwargs: pytest.fail('Partitions must not be selected'))
    with pytest.raises(ValueError, match='non-negative'):
        empty_target.map_overlap(lambda zds: 1, depth=-1)


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize('method', METHODS)
def test_mapping_reuses_variable_selection(mapping_target, method):
    """A one-shot selection survives all partition and immutable reads."""

    def callback(zds, *, partition_info):
        assert set(zds.variables) == {'x', 'constant'}
        numpy.testing.assert_array_equal(zds['constant'].values, [7, 8])
        return zds['x'].values[partition_info.indices]

    result = getattr(mapping_target, method)(
        callback, delayed=False, selected_variables=iter(
            ('x', 'constant'))).compute(scheduler='synchronous')
    numpy.testing.assert_array_equal(
        numpy.concatenate([values for _, values in result]),
        [10, 11, 20, 21, 30, 31])


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize(('method', 'depth'), [('map', None),
                                               ('map_overlap', 0),
                                               ('map_overlap', 1)])
@pytest.mark.usefixtures('dask_client')
def test_mapping_partition_info(mapping_target, method, depth):
    """Callbacks receive the identity and central slice of the mapped
    partition."""
    reference = mapping_target.view_ref if isinstance(
        mapping_target, view.View) else mapping_target
    names = tuple(reference.partitions(relative=True))

    def callback(zds, *, partition_info):
        assert set(zds.variables) == {'constant'}
        return partition_info

    options = {} if depth is None else {'depth': depth}
    result = getattr(mapping_target, method)(callback,
                                             delayed=False,
                                             selected_variables=('constant', ),
                                             **options).compute()
    indices = [slice(None)] * 3 if not depth else [
        slice(0, 2), slice(2, 4), slice(2, 4)
    ]
    assert len(result) == 3
    for (scheme, info), key, name, index in zip(result, (1, 2, 3),
                                                names,
                                                indices,
                                                strict=True):
        assert isinstance(info, PartitionInfo)
        assert info.partition == {'key': key}
        assert info.name == name
        assert info.dimension == 'row'
        assert info.indices == index
        assert scheme == tuple(info.partition.items())


@pytest.mark.parametrize('mapping_target', TARGETS, indirect=True)
@pytest.mark.parametrize('method', METHODS)
def test_mapping_npartitions(mapping_target, method):
    """``npartitions`` is passed to the Dask bag constructor."""
    bag = getattr(mapping_target, method)(lambda zds: None, npartitions=2)
    assert bag.npartitions == 2


@pytest.mark.parametrize('mapping_target', [('view', 'local_fs')],
                         indirect=True)
@pytest.mark.parametrize('method', METHODS)
def test_mapping_reference_only_view(mapping_target, method):
    """A view can map reference inputs without defining variables itself."""
    mapping_target.drop_variable('derived', distributed=False)
    assert not mapping_target.metadata.variables

    def callback(zds, *, partition_info):
        return zds['x'].values[partition_info.indices]

    result = getattr(mapping_target, method)(
        callback, delayed=False,
        selected_variables=('x', )).compute(scheduler='synchronous')
    numpy.testing.assert_array_equal(
        numpy.concatenate([values for _, values in result]),
        [10, 11, 20, 21, 30, 31])
