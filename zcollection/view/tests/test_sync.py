# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Synchronizing a view, and creating one."""
from __future__ import annotations

from unittest import mock

import pytest

from ... import convenience, dask_utils, execution, sync, view
from ...fs_utils import join_path
from ...tests.cluster import dask_client, dask_cluster  # noqa: F401
from ...tests.data import (
    KEYS,
    create_key_collection,
    fail_after,
    record_lock,
    with_process_sync,
)
from ...tests.fs import local_fs  # noqa: F401
from ..detail import CHECKSUM_FILE, _sync


def create(fs, **kwargs) -> view.View:
    """Create a view of the collection at `fs.collection`."""
    return convenience.create_view(str(fs.view),
                                   view.ViewReference(str(fs.collection),
                                                      fs.fs),
                                   filesystem=fs.fs,
                                   **kwargs)


def listed(zview: view.View, **kwargs) -> tuple[str, ...]:
    """Return the partitions of `zview`, relative to the view."""
    return tuple(
        zview._relative_path(path) for path in zview.partitions(**kwargs))


def stored(zview: view.View) -> list[str]:
    """Return the partitions of `zview` found in its store."""
    zview.fs.invalidate_cache()
    return sorted(
        zview._relative_path(path) for path in zview.fs.ls(zview.base_dir)
        if zview.fs.isdir(path))


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_a_failed_sync_reports_each_partition(
        local_fs,  # noqa: F811
        monkeypatch,
        distributed) -> None:
    """A failed sync names the partitions completed, failed and not
    attempted, leaves out those already synchronized, and leaves the index
    matching the store."""
    create_key_collection(local_fs)
    zview = create(local_fs, filters='key == 1', distributed=False)
    fail_after(monkeypatch, _sync, 'key=3')

    with pytest.raises(execution.PartitionErrors) as info:
        zview.sync(distributed=distributed, npartitions=1)

    assert info.value.completed == ('key=2', )
    assert list(info.value.failed) == ['key=3']
    assert info.value.not_attempted == ('key=4', )
    assert zview._index._read() is not None
    assert sorted(listed(zview)) == stored(zview)


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_sync_selects_the_partitions_it_writes(
        local_fs,  # noqa: F811
        distributed) -> None:
    """The predicate that ``sync`` returns selects the partitions ``sync``
    wrote."""
    create_key_collection(local_fs)
    zview = create(local_fs, filters='key == 1', distributed=False)

    written = zview.sync(distributed=distributed)

    assert tuple(listed(zview, filters=written)) == KEYS[1:]
    assert zview.is_synced(distributed=False)


def test_sync_records_its_filters_once_it_succeeds(
        local_fs, monkeypatch) -> None:  # noqa: F811
    """sync records its filters only once every selected partition is
    synchronized."""
    create_key_collection(local_fs)
    zview = create(local_fs, filters='key == 1', distributed=False)
    fail_after(monkeypatch, _sync, 'key=3')

    with pytest.raises(execution.PartitionErrors):
        zview.sync(filters='key <= 3', distributed=False)

    assert zview.filters == 'key == 1'
    assert convenience.open_view(str(local_fs.view),
                                 filesystem=local_fs.fs).filters == 'key == 1'

    monkeypatch.undo()
    zview.sync(filters='key <= 3', distributed=False)

    assert convenience.open_view(str(local_fs.view),
                                 filesystem=local_fs.fs).filters == 'key <= 3'


def test_a_failed_configuration_write_keeps_the_filters(
        local_fs) -> None:  # noqa: F811
    """When the configuration cannot be written, the view keeps its filters,
    and the partitions stay synchronized."""
    create_key_collection(local_fs)
    zview = create(local_fs, filters='key == 1', distributed=False)

    with mock.patch.object(view.View,
                           '_write_config',
                           side_effect=OSError('write failed')), \
            pytest.raises(OSError, match='write failed'):
        zview.sync(filters='key <= 3', distributed=False)

    assert zview.filters == 'key == 1'
    assert listed(zview) == KEYS[:3]


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_sync_and_creation_hold_the_lock(
        local_fs,  # noqa: F811
        monkeypatch,
        tmp_path,
        distributed) -> None:
    """The view holds its lock while it writes the partitions, when it is
    created and when it is synchronized."""
    create_key_collection(local_fs)
    synchronizer = sync.ProcessSync(str(tmp_path / 'create.lock'))
    held = record_lock(monkeypatch, synchronizer)
    zview = create(local_fs,
                   filters='key <= 2',
                   synchronizer=synchronizer,
                   distributed=distributed)
    assert held == [True, True]
    monkeypatch.undo()

    zview = with_process_sync(zview, tmp_path)
    held = record_lock(monkeypatch, zview)
    zview.sync(distributed=distributed)

    assert held == [True, True]
    assert not zview.is_locked()


@pytest.mark.usefixtures('dask_client')
def test_a_failed_creation_is_resumed(
        local_fs,  # noqa: F811
        monkeypatch) -> None:
    """After a failed creation, validate_partitions(fix=True) then
    sync(filters=view.filters) complete the view."""
    create_key_collection(local_fs)
    fail_after(monkeypatch, _sync, 'key=2')

    with pytest.raises(execution.PartitionErrors) as info:
        create(local_fs, filters='key <= 3', distributed=True, npartitions=1)

    assert info.value.completed == ('key=1', )
    monkeypatch.undo()

    zview = convenience.open_view(str(local_fs.view), filesystem=local_fs.fs)
    zview.validate_partitions(fix=True, distributed=False)
    zview.sync(filters=zview.filters, distributed=False)

    assert listed(zview) == KEYS[:3]
    assert zview.is_synced(filters='key <= 3', distributed=False)


def test_no_partition_asks_for_no_client(local_fs) -> None:  # noqa: F811
    """Creating, synchronizing or checking a view whose filters select no
    partition asks for no Dask client, whatever npartitions."""
    create_key_collection(local_fs)

    with mock.patch.object(dask_utils,
                           'get_client',
                           side_effect=AssertionError):
        zview = create(local_fs,
                       filters='key == 9',
                       distributed=True,
                       npartitions=0)
        zview.sync(filters='key == 9', distributed=True, npartitions=0)
        assert zview.is_synced(filters='key == 9',
                               distributed=True,
                               npartitions=0)

    assert tuple(zview.partitions()) == ()


@pytest.mark.usefixtures('dask_client')
def test_creation_refuses_an_invalid_npartitions(
        local_fs) -> None:  # noqa: F811
    """npartitions below one is refused when there are partitions to
    write."""
    create_key_collection(local_fs)

    with pytest.raises(ValueError, match='npartitions'):
        create(local_fs, distributed=True, npartitions=0)


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_is_synced_reports_a_partition_it_cannot_check(
        local_fs,  # noqa: F811
        distributed) -> None:
    """A partition without its checksum array makes is_synced raise
    PartitionErrors, whose completed names the partitions found out of sync,
    and validate_partitions(fix=True) writes that partition again."""
    create_key_collection(local_fs)
    zview = create(local_fs, filters='key <= 3', distributed=False)
    local_fs.fs.rm(join_path(zview.base_dir, 'key=2', CHECKSUM_FILE),
                   recursive=True)

    with pytest.raises(execution.PartitionErrors) as info:
        zview.is_synced(distributed=distributed,
                        npartitions=1,
                        stop_on_error=False)

    assert list(info.value.failed) == ['key=2']
    assert info.value.completed == ('key=4', )

    zview.validate_partitions(fix=True, distributed=False)
    assert zview.is_synced(filters='key <= 3', distributed=distributed)


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_is_synced_takes_no_lock(
        local_fs,  # noqa: F811
        monkeypatch,
        tmp_path,
        distributed) -> None:
    """The view does not hold its lock while is_synced checks the
    partitions."""
    create_key_collection(local_fs)
    zview = with_process_sync(create(local_fs, distributed=False), tmp_path)
    held = record_lock(monkeypatch, zview)

    assert zview.is_synced(distributed=distributed)
    assert held == [False, False]
