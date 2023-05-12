# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Fixtures for the tests.
=======================
"""
from typing import Literal

import pytest


@pytest.fixture
def dask_arrays() -> Literal[True]:
    """Load the data in Dask arrays."""
    return True


@pytest.fixture
def numpy_arrays() -> Literal[False]:
    """Load the data in NumPy arrays."""
    return False
