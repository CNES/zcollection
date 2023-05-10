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
