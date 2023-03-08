# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing utilities
=================
"""
import dask.distributed
import pytest

from .. import dask_utils
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint: disable=unused-import


@pytest.mark.filterwarnings('ignore:Port \\d+ is already in use.*')
def test_get_client_with_no_cluster():
    """Test the get_client function with no cluster."""
    with dask_utils.get_client() as client:
        assert isinstance(client, dask.distributed.Client)


def test_get_client_with_cluster(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the get_client function with a cluster."""
    with dask_utils.get_client() as client:
        assert isinstance(client, dask.distributed.Client)


def test_dask_workers(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the dask_workers function."""
    assert dask_utils.dask_workers(dask_client, cores_only=True) == len(
        dask_client.ncores())  # type: ignore
    assert dask_utils.dask_workers(dask_client, cores_only=False) == sum(
        item for item in dask_client.nthreads().values())  # type: ignore


def test_split_sequence():
    """Test the split_sequence function."""
    assert list(dask_utils.split_sequence(list(range(10)), 2)) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 3)) == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 4)) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
        [8, 9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 5)) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 6)) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8],
        [9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 7)) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 8)) == [
        [0, 1],
        [2, 3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 9)) == [
        [0, 1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 10)) == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)), 11)) == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert list(dask_utils.split_sequence(list(range(10)))) == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    ]
    with pytest.raises(ValueError):
        list(dask_utils.split_sequence(list(range(10)), 0))
