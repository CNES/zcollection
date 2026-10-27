# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the sync module.
========================
"""
import pickle

from .. import sync


def test_no_sync():
    """Test the no_sync class."""
    touch = False
    with sync.NoSync() as _:
        touch = True
    assert touch


def test_process_sync_reports_acquisition_and_release(tmp_path) -> None:
    """A released lock is reported as released even if its file remains."""
    synchronizer = sync.ProcessSync(str(tmp_path / 'lock'))

    assert synchronizer.is_locked() is False
    with synchronizer:
        assert synchronizer.is_locked() is True
    assert synchronizer.lock.exists()
    assert synchronizer.is_locked() is False


def test_process_sync_preserves_path_when_serialized(tmp_path) -> None:
    """Worker transfers preserve the lock's path across repeated transfers."""
    path = tmp_path / 'lock'
    synchronizer = sync.ProcessSync(str(path))
    expected = synchronizer.lock.path
    for _ in range(2):
        synchronizer = pickle.loads(pickle.dumps(synchronizer))
        assert synchronizer.lock.path == expected
        with synchronizer:
            assert path.exists()
