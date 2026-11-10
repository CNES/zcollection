# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Update contracts shared by collections and views."""
from __future__ import annotations

import importlib
import weakref

import dask.array
import dask.distributed
import numpy
import pytest

from .. import (
    convenience,
    dask_utils,
    dataset,
    execution,
    fs_utils,
    meta,
    partitioning,
    storage,
    sync,
    variable,
    view,
)
from ..collection.detail import infer_variables
from .cluster import dask_client, dask_cluster  # noqa: F401
from .data import with_process_sync
from .fixture import empty_target  # noqa: F401
from .fs import local_fs, s3, s3_base, s3_fs  # noqa: F401


@pytest.fixture
def update_target(request):
    """Build four partitions with two rows and a writable output."""
    kind, fs_name = request.param
    fs = request.getfixturevalue(fs_name)
    keys = numpy.repeat(numpy.arange(1, 5), 2)
    matrix = numpy.ma.array(numpy.stack([keys, keys + 10]),
                            mask=numpy.stack([keys == 2, keys == 4]),
                            fill_value=-99)
    zds = dataset.Dataset([
        variable.Array('key', keys, ('row', )),
        variable.Array('source', keys + 10, ('row', )),
        variable.Array('target', numpy.zeros_like(keys), ('row', )),
        variable.Array('matrix', matrix, ('column', 'row'), fill_value=-99),
        variable.Array('matrix_target',
                       numpy.zeros((2, keys.size), dtype='int64'),
                       ('column', 'row'),
                       fill_value=-99),
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
    zview.add_variable(meta.VariableSchema('output',
                                           'int64',
                                           dimensions=('row', ),
                                           fill_value=-1),
                       distributed=False)
    return zview


TARGETS = [(kind, fs) for kind in ('collection', 'view')
           for fs in ('local_fs', 's3_fs')]
LOCAL_TARGETS = [(kind, 'local_fs') for kind in ('collection', 'view')]


def output_name(target):
    """Return the writable variable of the target."""
    return 'output' if isinstance(target, view.View) else 'target'


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('delayed', [None, False, True])
@pytest.mark.usefixtures('dask_client')
def test_update_modes_and_selection(update_target, distributed, delayed):
    """Inference and updates use the selected partition and loading mode."""
    target = update_target
    output = output_name(target)
    expected_delayed = distributed if delayed is None else delayed

    def callback(zds, increment, *, expected_key):
        assert set(zds.variables) == {'key', 'source'}
        assert isinstance(
            zds['source'].array,
            dask.array.Array if expected_delayed else numpy.ndarray)
        numpy.testing.assert_array_equal(zds['key'].values, [expected_key] * 2)
        if distributed:
            assert dask.distributed.get_worker() is not None
        return {output: zds['source'].values + increment}

    # A one-shot selection must survive inference. The first stored partition
    # is deliberately excluded.
    target.update(callback,
                  5,
                  expected_key=3,
                  delayed=delayed,
                  distributed=distributed,
                  variables=None,
                  selected_variables=iter(['key', 'source']),
                  selected_partitions=iter(['key=3']))
    result = target.load(delayed=False, distributed=False)
    assert result is not None
    values = result[output].values
    numpy.testing.assert_array_equal(values[4:6], [18, 18])
    numpy.testing.assert_array_equal(values[:4],
                                     0 if output == 'target' else -1)
    numpy.testing.assert_array_equal(values[6:],
                                     0 if output == 'target' else -1)


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
def test_update_empty_selection(update_target, monkeypatch):
    """An empty selection needs neither a client nor a callback."""

    def unexpected(*args, **kwargs):
        pytest.fail('An empty selection must not execute work')

    monkeypatch.setattr(dask_utils, 'get_client', unexpected)
    with pytest.warns((RuntimeWarning, UserWarning)):
        update_target.update(unexpected,
                             selected_partitions=iter(()),
                             variables=None)


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('depth', [0, 1])
def test_update_loads_bounded_window(update_target, depth, monkeypatch):
    """Callbacks run before all partitions are loaded and release their inputs."""
    target = update_target
    output = output_name(target)
    kind = 'view' if isinstance(target, view.View) else 'collection'
    detail = importlib.import_module(f'zcollection.{kind}.detail')
    name = '_load_one_dataset' if kind == 'view' else 'read_partition'
    load = getattr(detail, name)
    alive = weakref.WeakValueDictionary()
    loads = []
    calls = []

    def observe(*args, **kwargs):
        # Inputs from the preceding callback must be released before reading
        # the next partition or overlap window.
        if calls:
            assert not alive
        result = load(*args, **kwargs)
        loads.append(None)
        return result

    def callback(zds):
        if not calls:
            assert len(loads) == depth + 1
        for item in zds.variables.values():
            alive[id(item.array)] = item.array
        calls.append(None)
        return {output: zds['source'].values.copy()}

    monkeypatch.setattr(detail, name, observe)
    target.update(callback,
                  variables=[output],
                  selected_variables=['source'],
                  delayed=False,
                  distributed=False,
                  depth=depth)
    assert len(calls) == 4


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('trim', [False, True])
@pytest.mark.usefixtures('dask_client')
def test_update_overlap_without_reading_output(update_target, trim):
    """Trimming uses output metadata while loading only callback inputs."""
    target = update_target
    output = output_name(target)
    selected = ['source']

    def callback(zds, partition_info):
        assert set(zds.variables) == {'source'}
        dim, indices = partition_info.dimension, partition_info.indices
        assert dim == 'row'
        values = zds['source'].values + 1
        return {output: values if trim else values[indices]}

    target.update(callback,
                  variables=[output],
                  selected_variables=selected,
                  delayed=False,
                  distributed=True,
                  depth=1,
                  trim=trim)
    assert selected == ['source']
    result = target.load(delayed=False, distributed=False)
    assert result is not None
    numpy.testing.assert_array_equal(result[output].values,
                                     numpy.repeat(numpy.arange(12, 16), 2))


@pytest.mark.parametrize('update_target', TARGETS, indirect=True)
@pytest.mark.parametrize('delayed', [False, True])
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.usefixtures('dask_client')
def test_update_overlap_masked_second_axis(update_target, delayed,
                                           distributed):
    """Trim masked results along the schema's partitioned axis."""
    target = update_target
    output = 'matrix_target'
    if isinstance(target, view.View):
        output = 'matrix_output'
        target.add_variable(meta.VariableSchema(output,
                                                'int64',
                                                dimensions=('column', 'row'),
                                                fill_value=-99),
                            distributed=False)

    def callback(zds, *, metadata):
        assert metadata == 'callback argument'
        assert set(zds.variables) == {'matrix'}
        values = zds['matrix'].values
        assert isinstance(values, numpy.ma.MaskedArray)
        return {output: values + 5}

    target.update(callback,
                  variables=[output],
                  selected_variables=['matrix'],
                  delayed=delayed,
                  distributed=distributed,
                  depth=1,
                  metadata='callback argument')
    result = target.load(delayed=False, distributed=False)
    assert result is not None
    expected = result['matrix'].values + 5
    actual = result[output].values
    numpy.testing.assert_array_equal(actual.mask, expected.mask)
    numpy.testing.assert_array_equal(actual.compressed(),
                                     expected.compressed())


@pytest.mark.parametrize('update_target', [(kind, 'local_fs')
                                           for kind in ('collection', 'view')],
                         indirect=True)
@pytest.mark.parametrize('depth', [0, 1])
@pytest.mark.parametrize('infer', [False, True])
@pytest.mark.usefixtures('dask_client')
def test_update_distributes_paths(update_target, depth, infer, monkeypatch,
                                  tmp_path):
    """The coordinator sends one path per task while it holds the lock, and
    returns the paths written."""
    client = dask.distributed.get_client()
    submit = client.submit
    paths = []
    output = output_name(update_target)
    update_target = with_process_sync(update_target, tmp_path)

    def observe_submit(func, *args, **kwargs):
        if func is execution._call:
            assert update_target.is_locked()
            paths.append(args[1])
        return submit(func, *args, **kwargs)

    def callback(zds):
        return {output: zds['source'].values + 2}

    monkeypatch.setattr(dask_utils, 'get_client', lambda: client)
    monkeypatch.setattr(client, 'submit', observe_submit)
    written = update_target.update(callback,
                                   variables=None if infer else [output],
                                   selected_variables=['source'],
                                   delayed=False,
                                   distributed=True,
                                   depth=depth)
    assert len(paths) == 4
    assert all(isinstance(path, str) for path in paths)
    assert written == tuple(
        update_target._relative_path(path) for path in paths)
    assert not update_target.is_locked()


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize(('stop_on_error', 'written', 'not_attempted'), [
    (True, ('key=1', 'key=2'), ('key=4', )),
    (False, ('key=1', 'key=2', 'key=4'), ()),
])
@pytest.mark.usefixtures('dask_client')
def test_update_reports_each_partition(update_target, distributed,
                                       stop_on_error, written, not_attempted,
                                       monkeypatch) -> None:
    """A failed callback gives the partitions written, failed and not
    attempted, and the cache of every selected partition is cleared."""
    output = output_name(update_target)
    cleared: list[str] = []
    monkeypatch.setattr(update_target.fs, 'invalidate_cache', cleared.append)

    def fail_on_three(zds):
        if zds['key'].values[0] == 3:
            raise ValueError('callback failed')
        return {output: zds['source'].values}

    with pytest.raises(execution.PartitionErrors) as info:
        update_target.update(fail_on_three,
                             variables=[output],
                             npartitions=1,
                             distributed=distributed,
                             stop_on_error=stop_on_error)

    assert info.value.written == written
    assert list(info.value.failed) == ['key=3']
    assert 'callback failed' in info.value.failed['key=3']
    assert info.value.not_attempted == not_attempted
    assert {'key=1', 'key=2', 'key=3', 'key=4'
            } <= {update_target._relative_path(path)
                  for path in cleared}


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
def test_update_holds_the_lock_without_dask(update_target, tmp_path) -> None:
    """Without Dask, each partition is updated while the lock is held."""
    output = output_name(update_target)
    update_target = with_process_sync(update_target, tmp_path)
    held = []

    def callback(zds):
        held.append(update_target.is_locked())
        return {output: zds['source'].values}

    assert update_target.update(callback,
                                variables=[output],
                                distributed=False) == ('key=1', 'key=2',
                                                       'key=3', 'key=4')
    assert held == [True] * 4
    assert not update_target.is_locked()


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.usefixtures('dask_client')
def test_update_skips_an_absent_partition(update_target, distributed,
                                          caplog) -> None:
    """A partition the index names and the store lacks is skipped with a
    warning, and left out of the result."""
    output = output_name(update_target)
    update_target.fs.rm(fs_utils.join_path(update_target._partitions_dir,
                                           'key=2'),
                        recursive=True)

    written = update_target.update(lambda zds: {output: zds['source'].values},
                                   variables=[output],
                                   distributed=distributed)

    assert written == ('key=1', 'key=3', 'key=4')
    assert 'key=2' in caplog.text


@pytest.mark.parametrize('missing_all', [False, True])
def test_inference_skips_absent_partitions(tmp_path, missing_all):
    """Inference locks reads, skips absent paths and returns only output names."""
    synchronizer = sync.ProcessSync(str(tmp_path / 'inference.lock'))
    loaded = []
    calls = []

    def load(path):
        assert synchronizer.is_locked()
        loaded.append(path)
        if path == 'key=1' or missing_all:
            raise storage.AbsentPartition(path)
        return dataset.Dataset([
            variable.Array('source', numpy.arange(2), ('row', )),
        ],
                               delayed=False)

    def callback(zds):
        assert not synchronizer.is_locked()
        calls.append(None)
        return {'target': zds['source'].values + 1}

    result = infer_variables(('key=1', 'key=2'),
                             load,
                             callback,
                             'row', (), {},
                             synchronizer,
                             partitioning=partitioning.Sequence(('key', )))
    assert result == (() if missing_all else ('target', ))
    assert len(calls) == (0 if missing_all else 1)
    assert loaded == ['key=1', 'key=2']


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('infer', [False, True])
def test_update_rejects_unknown_outputs(update_target, infer):
    """Declared and inferred unknown outputs fail before writing."""
    calls = []

    def callback(zds):
        calls.append(None)
        return {'missing': zds['source'].values}

    with pytest.raises(ValueError, match='missing'):
        update_target.update(callback,
                             variables=None if infer else ['missing'],
                             selected_variables=['source'],
                             distributed=False)
    assert len(calls) == int(infer)


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
def test_update_invalidates_selected_paths(update_target, monkeypatch):
    """Both update methods invalidate the selected partition paths at the end."""
    partitions = tuple(update_target.partitions())
    selected = partitions[1:2]
    invalidate = update_target.fs.invalidate_cache
    invalidated = []
    output = output_name(update_target)

    def observe(path=None):
        invalidated.append(path)
        invalidate(path)

    def callback(zds):
        return {output: zds['source'].values.copy()}

    monkeypatch.setattr(update_target.fs, 'invalidate_cache', observe)
    update_target.update(callback,
                         variables=[output],
                         selected_partitions=selected,
                         selected_variables=['source'],
                         distributed=False)
    assert set(selected).issubset(invalidated)


def test_update_rejects_negative_depth_before_callback(
        empty_target,  # noqa: F811
        monkeypatch):
    """Invalid overlap depths fail before inference or execution."""

    def unexpected(*args, **kwargs):
        pytest.fail('Partitions must not be selected')

    monkeypatch.setattr(empty_target, '_select_update_partitions', unexpected)
    with pytest.raises(ValueError, match='Depth must be non-negative'):
        empty_target.update(unexpected, depth=-1, distributed=False)


def test_partition_resources_are_read_only(empty_target):  # noqa: F811
    """A partition index and its operations retain the same filesystem and lock."""
    fs = empty_target.fs
    synchronizer = empty_target.synchronizer
    with pytest.raises(AttributeError):
        empty_target.fs = fs
    with pytest.raises(AttributeError):
        empty_target.synchronizer = sync.NoSync()
    assert empty_target.fs is fs
    assert empty_target.synchronizer is synchronizer


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('depth', [0, 1])
def test_update_keeps_callback_keywords(update_target, depth):
    """Callback keywords stay separate from the partition writer's settings."""
    output = output_name(update_target)
    payload = dict.fromkeys(('indices', 'dim', 'fs', 'path', 'metadata',
                             'func', 'zds', 'func_args', 'func_kwargs'),
                            'argument')

    def callback(ds, partition_info=None, **kwargs):
        assert kwargs == payload
        assert partition_info.dimension == 'row'
        return {output: ds['source'].values.copy()}

    update_target.update(callback,
                         variables=[output],
                         selected_variables=['source'],
                         distributed=False,
                         depth=depth,
                         **payload)
    result = update_target.load(delayed=False, distributed=False)
    assert result is not None
    numpy.testing.assert_array_equal(result[output].values,
                                     result['source'].values)


@pytest.mark.parametrize('update_target', LOCAL_TARGETS, indirect=True)
@pytest.mark.parametrize('depth', [0, 1])
@pytest.mark.usefixtures('dask_client')
def test_update_partition_info(update_target, depth):
    """Inference and writes receive the identity of the partition they load."""
    output = output_name(update_target)
    reference = update_target.view_ref if isinstance(
        update_target, view.View) else update_target
    names = dict(
        zip((1, 2, 3, 4), reference.partitions(relative=True), strict=True))

    def callback(zds, *, partition_info):
        values = zds['source'].values
        # Inference loads one partition even when writes use overlap windows.
        if not depth or values.size == 2:
            assert partition_info.indices == slice(None)
        central = values[partition_info.indices]
        key = int(central[0] - 10)
        assert partition_info.partition == {'key': key}
        assert partition_info.name == names[key]
        assert partition_info.dimension == 'row'
        numpy.testing.assert_array_equal(central, [key + 10, key + 10])
        return {output: numpy.full_like(values, key)}

    update_target.update(callback,
                         selected_variables=('source', ),
                         delayed=False,
                         distributed=True,
                         depth=depth)
    result = update_target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(result[output].values,
                                     numpy.repeat(numpy.arange(1, 5), 2))
