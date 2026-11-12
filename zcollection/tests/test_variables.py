# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Adding and dropping variables, on collections and views."""
from __future__ import annotations

from unittest import mock

import numpy
import pytest

from .. import (
    collection as collection_module,
    convenience,
    dask_utils,
    dataset,
    execution,
    fs_utils,
    meta,
    partitioning,
    storage,
    store,
    variable,
    view,
)
from ..view import detail as view_detail
from .cluster import dask_client, dask_cluster  # noqa: F401
from .data import fail_after, record_lock, with_process_sync
from .fs import local_fs  # noqa: F401

#: The variable the tests add.
EXTRA = meta.VariableSchema('extra', 'float64', dimensions=('row', ))

#: The partitions of a target.
NAMES = ('key=1', 'key=2', 'key=3', 'key=4')


def make_collection(fs, *, insert: bool = True):
    """Return a collection of four partitions of two rows, or with no
    partitions if `insert` is false."""
    keys = numpy.repeat(numpy.arange(1, 5), 2)
    zds = dataset.Dataset([
        variable.Array('key', keys, ('row', )),
        variable.Array('source', keys + 10.0, ('row', )),
        variable.Array('grid', numpy.arange(3.0), ('cell', )),
    ],
                          delayed=False)
    zcol = convenience.create_collection('key',
                                         zds,
                                         partitioning.Sequence(('key', )),
                                         str(fs.collection),
                                         filesystem=fs.fs)
    if insert:
        zcol.insert(zds, distributed=False)
    return zcol


@pytest.fixture
def target(request, local_fs):  # noqa: F811
    """Return a collection, or a view of it, with four partitions."""
    make_collection(local_fs)
    if request.param == 'collection':
        return convenience.open_collection(str(local_fs.collection),
                                           mode='w',
                                           filesystem=local_fs.fs)
    return convenience.create_view(str(local_fs.view),
                                   view.ViewReference(str(local_fs.collection),
                                                      local_fs.fs),
                                   filesystem=local_fs.fs,
                                   distributed=False)


KINDS = pytest.mark.parametrize('target', ['collection', 'view'],
                                indirect=True)


def reopened(target):
    """Return the target opened again from its configuration."""
    if isinstance(target, view.View):
        return view.View.from_config(target.base_dir, filesystem=target.fs)
    return collection_module.Collection.from_config(
        target.partition_properties.dir, mode='w', filesystem=target.fs)


def partitions_with(target, name: str) -> list[str]:
    """Return the partitions of the target containing the array `name`."""
    return [
        target._relative_path(path) for path in target.partitions()
        if target.fs.exists(fs_utils.join_path(path, name))
    ]


def tasks(target):
    """Return the module defining the tasks that add and drop the arrays of
    `target`."""
    return view_detail if isinstance(target, view.View) else collection_module


@KINDS
@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_a_failed_addition_is_undone(target, monkeypatch, distributed) -> None:
    """When a partition fails, the variable is removed from every partition
    and from the schema."""
    fail_after(monkeypatch, tasks(target)._add_array, 'key=3')

    with pytest.raises(execution.PartitionErrors) as info:
        target.add_variable(EXTRA, distributed, npartitions=1)

    assert info.value.completed == NAMES[:2]
    assert list(info.value.failed) == ['key=3']
    assert info.value.not_attempted == NAMES[3:]
    assert not getattr(info.value, '__notes__', None)
    assert 'extra' not in target.metadata.variables
    assert 'extra' not in reopened(target).metadata.variables
    assert partitions_with(target, 'extra') == []


@KINDS
def test_a_failed_undo_leaves_the_variable_declared(target,
                                                    monkeypatch) -> None:
    """When the removal of the added arrays fails too, the schema still
    declares the variable, the error names the partitions concerned, and drop_variable
    finishes the removal."""
    fail_after(monkeypatch, tasks(target)._add_array, 'key=3')
    fail_after(monkeypatch, tasks(target)._drop_array, 'key=1')

    with pytest.raises(execution.PartitionErrors) as info:
        target.add_variable(EXTRA, False)

    note = ' '.join(info.value.__notes__)
    assert 'key=1' in note
    assert 'drop_variable' in note
    assert 'extra' in target.metadata.variables
    assert 'extra' in reopened(target).metadata.variables

    monkeypatch.undo()
    target.drop_variable('extra', False)

    assert 'extra' not in reopened(target).metadata.variables
    assert partitions_with(target, 'extra') == []


@KINDS
@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_a_failed_removal_is_finished_by_a_second_call(target, monkeypatch,
                                                       distributed) -> None:
    """A failed removal leaves the variable declared, and a second call
    removes what remains."""
    target.add_variable(EXTRA, False)
    fail_after(monkeypatch, tasks(target)._drop_array, 'key=3')

    with pytest.raises(execution.PartitionErrors) as info:
        target.drop_variable('extra', distributed, npartitions=1)

    assert info.value.completed == NAMES[:2]
    assert 'extra' in target.metadata.variables
    assert 'extra' in reopened(target).metadata.variables
    assert partitions_with(target, 'extra') == list(NAMES[3:])

    monkeypatch.undo()
    target.drop_variable('extra', distributed)

    assert 'extra' not in target.metadata.variables
    assert 'extra' not in reopened(target).metadata.variables
    assert partitions_with(target, 'extra') == []


@KINDS
@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_variables_are_added_and_dropped_under_the_lock(
        target, monkeypatch, tmp_path, distributed) -> None:
    """Each array is created and removed while the target holds its lock."""
    target = with_process_sync(target, tmp_path)
    held = record_lock(monkeypatch, target)
    target.add_variable(EXTRA, distributed)
    target.drop_variable('extra', distributed)

    assert held == [True] * 4
    assert not target.is_locked()


@KINDS
def test_a_failed_configuration_write_keeps_the_schema(target,
                                                       monkeypatch) -> None:
    """The schema of the target changes only once its configuration is
    written."""
    with mock.patch.object(type(target),
                           '_write_config',
                           side_effect=OSError('write failed')), \
            pytest.raises(OSError, match='write failed'):
        target.add_variable(EXTRA, False)

    assert partitions_with(target, 'extra') == []
    assert 'extra' not in target.metadata.variables

    target.add_variable(EXTRA, False)
    with mock.patch.object(type(target),
                           '_write_config',
                           side_effect=OSError('write failed')), \
            pytest.raises(OSError, match='write failed'):
        target.drop_variable('extra', False)

    assert 'extra' in target.metadata.variables
    assert partitions_with(target, 'extra') == []

    target.drop_variable('extra', False)
    assert 'extra' not in reopened(target).metadata.variables


@KINDS
@pytest.mark.usefixtures('dask_client')
def test_an_invalid_npartitions_leaves_the_schema(target) -> None:
    """An npartitions below one is refused before the schema changes."""
    with pytest.raises(ValueError, match='npartitions'):
        target.add_variable(EXTRA, True, npartitions=0)

    assert 'extra' not in target.metadata.variables
    assert 'extra' not in reopened(target).metadata.variables


@KINDS
def test_a_missing_client_leaves_the_schema(target) -> None:
    """When no Dask client can be obtained, the addition stops before the
    schema changes."""
    with mock.patch.object(dask_utils,
                           'get_client',
                           side_effect=ValueError('no client')), \
            pytest.raises(ValueError, match='no client'):
        target.add_variable(EXTRA, True)

    assert 'extra' not in target.metadata.variables
    assert 'extra' not in reopened(target).metadata.variables


@KINDS
def test_a_vanished_partition_is_skipped(target) -> None:
    """add_variable and drop_variable skip a partition listed in the index
    but missing from the store."""
    vanished = list(target.partitions())[1]
    target.fs.rm(vanished, recursive=True)
    target.fs.invalidate_cache()

    target.add_variable(EXTRA, False)
    assert not target.fs.exists(vanished)

    target.drop_variable('extra', False)
    assert not target.fs.exists(vanished)


@pytest.mark.parametrize('distributed', [False, True])
def test_a_collection_without_partitions_asks_for_no_client(
        local_fs, distributed) -> None:  # noqa: F811
    """Without partitions, the schema changes, no Dask client is asked for,
    and npartitions is not checked."""
    zcol = make_collection(local_fs, insert=False)

    with mock.patch.object(dask_utils,
                           'get_client',
                           side_effect=AssertionError):
        zcol.add_variable(EXTRA, distributed, npartitions=0)
        assert 'extra' in zcol.metadata.variables
        zcol.drop_variable('extra', distributed, npartitions=0)

    assert 'extra' not in zcol.metadata.variables


def test_a_failed_immutable_addition_is_undone(
        local_fs, monkeypatch) -> None:  # noqa: F811
    """An immutable variable that cannot be written is removed from the
    schema."""
    zcol = make_collection(local_fs)
    fail_after(monkeypatch, collection_module._add_array, '.immutable')
    immutable = meta.VariableSchema('mask', 'int8', dimensions=('cell', ))

    with pytest.raises(execution.PartitionErrors) as info:
        zcol.add_variable(immutable, True, npartitions=0)

    assert list(info.value.failed) == ['.immutable']
    assert 'task failed' in info.value.failed['.immutable']
    assert not local_fs.fs.exists(fs_utils.join_path(zcol._immutable, 'mask'))
    assert 'mask' not in zcol.metadata.variables
    assert 'mask' not in reopened(zcol).metadata.variables


def test_an_immutable_variable_is_processed_without_dask(
        local_fs) -> None:  # noqa: F811
    """An immutable variable is added and removed without a Dask client,
    whatever npartitions."""
    zcol = make_collection(local_fs)
    immutable = meta.VariableSchema('mask', 'int8', dimensions=('cell', ))

    with mock.patch.object(dask_utils,
                           'get_client',
                           side_effect=AssertionError):
        zcol.add_variable(immutable, True, npartitions=0)
        assert local_fs.fs.exists(fs_utils.join_path(zcol._immutable, 'mask'))
        zcol.drop_variable('mask', True, npartitions=0)

    assert not local_fs.fs.exists(fs_utils.join_path(zcol._immutable, 'mask'))
    assert 'mask' not in reopened(zcol).metadata.variables


def test_dropping_an_absent_array_consolidates_the_group(
        local_fs) -> None:  # noqa: F811
    """A removal that deleted the array but did not consolidate the group is
    finished by a second one."""
    zcol = make_collection(local_fs)
    partition = next(zcol.partitions())
    local_fs.fs.rm(fs_utils.join_path(partition, 'source'), recursive=True)

    storage.del_zarr_array(partition, 'source', local_fs.fs)

    names = {
        name
        for name, _ in store.open_group(local_fs.fs, partition).arrays()
    }
    assert 'source' not in names
    assert 'key' in names
