# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Copying a collection."""
from __future__ import annotations

import contextlib

import pytest

from ... import collection, convenience, execution, partitions, sync
from ...tests.cluster import dask_client, dask_cluster  # noqa: F401
from ...tests.data import KEYS, create_key_collection, fail_after, record_lock
from ...tests.fs import local_fs  # noqa: F401


def test_a_copy_holds_the_immutable_variables(local_fs) -> None:  # noqa: F811
    """Loading the copy returns the immutable variables of the collection."""
    zcol = create_key_collection(local_fs)

    copied = zcol.copy(str(local_fs.root / 'copy'), distributed=False)

    loaded = copied.load(distributed=False)
    assert loaded is not None
    assert loaded['grid'].values.tolist() == [0.0, 1.0, 2.0]


def test_a_copy_holds_the_files_of_the_root(local_fs) -> None:  # noqa: F811
    """The copy holds the files of the root directory of the collection."""
    zcol = create_key_collection(local_fs)
    local_fs.fs.pipe_file(f'{local_fs.collection}/table.parquet', b'table')
    target = str(local_fs.root / 'copy')

    zcol.copy(target, distributed=False)

    assert local_fs.fs.cat_file(f'{target}/table.parquet') == b'table'


def test_a_copy_refuses_a_target_holding_files(local_fs) -> None:  # noqa: F811
    """A copy to a directory that holds files raises ValueError and writes
    nothing."""
    zcol = create_key_collection(local_fs)
    target = str(local_fs.root / 'copy')
    local_fs.fs.makedirs(target)
    local_fs.fs.touch(f'{target}/other')

    with pytest.raises(ValueError, match='not empty'):
        zcol.copy(target, distributed=False)

    assert local_fs.fs.ls(target, detail=False) == [
        local_fs.fs._strip_protocol(f'{target}/other')
    ]


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_a_failed_copy_reports_each_partition(
        local_fs,  # noqa: F811
        monkeypatch,
        distributed) -> None:
    """The error of a failed copy names the partitions completed, failed and
    not attempted, and the index of the copy is rebuilt with the partitions
    its store holds."""
    zcol = create_key_collection(local_fs)
    target = str(local_fs.root / 'copy')
    fail_after(monkeypatch, collection._copy_partition, KEYS[1])

    with pytest.raises(execution.PartitionErrors) as info:
        zcol.copy(target, npartitions=1, distributed=distributed)

    assert info.value.completed == KEYS[:1]
    assert list(info.value.failed) == [KEYS[1]]
    assert info.value.not_attempted == KEYS[2:]
    copied = convenience.open_collection(target, filesystem=local_fs.fs)
    assert copied._index._read() is not None
    assert tuple(copied.partitions(relative=True)) == KEYS[:2]


def interrupt(*args, **kwargs) -> None:
    """Raise KeyboardInterrupt."""
    raise KeyboardInterrupt


def test_a_copy_lists_its_partitions_without_scanning_its_store(
        local_fs,  # noqa: F811
        monkeypatch) -> None:
    """Once its partitions are copied, the new collection lists them, even
    when a scan of its store is interrupted."""
    zcol = create_key_collection(local_fs)
    target = str(local_fs.root / 'copy')
    monkeypatch.setattr(partitions.PartitionIndex, 'walk', interrupt)

    with contextlib.suppress(KeyboardInterrupt):
        zcol.copy(target, distributed=False)

    monkeypatch.undo()
    copied = convenience.open_collection(target, filesystem=local_fs.fs)
    assert tuple(copied.partitions(relative=True)) == KEYS


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_a_copy_takes_the_lock_of_its_target(
        local_fs,  # noqa: F811
        monkeypatch,
        tmp_path,
        distributed) -> None:
    """``copy`` takes the lock of the new collection while it copies the
    partitions, and releases it afterwards."""
    zcol = create_key_collection(local_fs)
    synchronizer = sync.ProcessSync(str(tmp_path / 'copy.lock'))
    held = record_lock(monkeypatch, synchronizer)

    zcol.copy(str(local_fs.root / 'copy'),
              synchronizer=synchronizer,
              distributed=distributed)

    assert held == [True, True]
    assert not synchronizer.is_locked()
