# Copyright (c) 2022 CNES
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
