# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Validating the partitions of collections and views."""
from __future__ import annotations

import pytest

from .. import execution, fs_utils, view
from ..view import detail as view_detail
from .cluster import dask_client, dask_cluster  # noqa: F401
from .data import (
    KINDS,
    create_target,
    fail_after,
    record_lock,
    with_process_sync,
)
from .fs import local_fs  # noqa: F401


@pytest.fixture
def target(request, local_fs):  # noqa: F811
    """Return a collection, or a view of it, with four partitions."""
    return create_target(request.param, local_fs)


def break_partition(target, name: str) -> None:
    """Remove from the partition `name` an array the target requires."""
    array = (view_detail.CHECKSUM_FILE
             if isinstance(target, view.View) else 'source')
    path = fs_utils.join_path(target._partitions_dir, name)
    target.fs.rm(fs_utils.join_path(path, array), recursive=True)
    target.fs.invalidate_cache()


def stored(target) -> list[str]:
    """Return the partitions of `target` found in its store."""
    target.fs.invalidate_cache()
    root = target._partitions_dir
    return sorted(
        target._relative_path(path) for path in target.fs.ls(root)
        if target.fs.isdir(path) and not path.endswith('.immutable'))


def listed(target) -> list[str]:
    """Return the partitions `target` lists, relative to it."""
    return sorted(target._relative_path(path) for path in target.partitions())


@pytest.mark.parametrize('target', ['collection'], indirect=True)
@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_a_partition_missing_a_declared_array_is_invalid(target,
                                                         distributed) -> None:
    """A partition without an array the target requires is invalid, even
    when the consolidated metadata of the partition still names it."""
    break_partition(target, 'key=2')

    with pytest.warns(RuntimeWarning, match='key=2'):
        assert target.validate_partitions(
            distributed=distributed) == ('key=2', )
    with pytest.warns(RuntimeWarning, match='key=2'):
        target.validate_partitions(distributed=distributed, fix=True)

    assert target.validate_partitions(distributed=distributed) == ()


@pytest.mark.parametrize('target', ['view'], indirect=True)
@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_a_failed_repair_leaves_the_index_matching_the_store(
        target, monkeypatch, distributed) -> None:
    """After a failed repair, the partitions listed are those the store
    holds."""
    break_partition(target, 'key=1')
    break_partition(target, 'key=2')
    fail_after(monkeypatch, view_detail._repair_partition, 'key=1')

    with pytest.warns(RuntimeWarning), \
            pytest.raises(execution.PartitionErrors) as info:
        target.validate_partitions(distributed=distributed,
                                   fix=True,
                                   npartitions=1)

    assert list(info.value.failed) == ['key=1']
    assert target._index._read() is not None
    assert listed(target) == stored(target)


@KINDS
def test_only_fix_takes_the_lock(target, monkeypatch, tmp_path) -> None:
    """The target holds its lock while it checks and repairs the
    partitions, and not while it only checks them."""
    target = with_process_sync(target, tmp_path)
    break_partition(target, 'key=1')
    held = record_lock(monkeypatch, target)

    with pytest.warns(RuntimeWarning):
        target.validate_partitions(distributed=False)
    assert held == [False, False]

    held.clear()
    with pytest.warns(RuntimeWarning):
        target.validate_partitions(distributed=False, fix=True)
    assert held == [True] * 4
    assert not target.is_locked()
