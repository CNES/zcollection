# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the sync module.
========================
"""
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
