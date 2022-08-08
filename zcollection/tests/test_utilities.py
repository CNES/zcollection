# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing utilities
=================
"""
import pathlib
import platform

import dask.distributed
import fsspec
import pytest

from .. import utilities
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint: disable=unused-import


def test_fs_walk(tmpdir):
    """Test the fs_walk function."""
    for ix, item in enumerate([
        ('year=2014', 'month=5'),
        ('year=2014', 'month=5', 'day=2'),
        ('year=2014', 'month=5', 'day=1'),
        ('year=2014', 'month=5', 'day=3'),
        ('year=2014', 'month=4'),
        ('year=2014', 'month=4', 'day=16'),
        ('year=2014', 'month=4', 'day=24'),
        ('year=2014', 'month=4', 'day=27'),
        ('year=2014', 'month=4', 'day=20'),
        ('year=2014', 'month=4', 'day=29'),
        ('year=2014', 'month=4', 'day=14'),
        ('year=2014', 'month=4', 'day=25'),
        ('year=2014', 'month=4', 'day=19'),
        ('year=2014', 'month=4', 'day=12'),
        ('year=2014', 'month=4', 'day=23'),
        ('year=2014', 'month=4', 'day=17'),
        ('year=2014', 'month=4', 'day=28'),
        ('year=2014', 'month=4', 'day=13'),
        ('year=2014', 'month=4', 'day=21'),
        ('year=2014', 'month=4', 'day=15'),
        ('year=2014', 'month=4', 'day=18'),
        ('year=2014', 'month=4', 'day=26'),
        ('year=2014', 'month=4', 'day=22'),
        ('year=2014', 'month=4', 'day=30'),
    ]):
        path = pathlib.Path(tmpdir).joinpath(*item)
        path.mkdir(parents=True, exist_ok=False)
        if 'day' in item[-1]:
            with path.joinpath(f'file_{ix}.txt').open(mode='w',
                                                      encoding='utf-8'):
                ...

    fs = utilities.get_fs()
    listing1 = []
    for root, _dirs, files in utilities.fs_walk(fs, tmpdir, sort=True):
        for item in files:
            listing1.append(fs.sep.join([root, item]))

    listing2 = []
    for root, _dirs, files in utilities.fs_walk(fs, tmpdir, sort=False):
        for item in files:
            listing2.append(fs.sep.join([root, item]))

    assert listing1 == sorted(listing2)

    assert list(
        utilities.fs_walk(fs,
                          str(pathlib.Path(tmpdir).joinpath('inexistent')),
                          sort=True)) == [('', [], [])]


def test_get_client_with_no_cluster():
    """Test the get_client function with no cluster."""
    with utilities.get_client() as client:
        assert isinstance(client, dask.distributed.Client)


def test_get_client_with_cluster(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the get_client function with a cluster."""
    with utilities.get_client() as client:
        assert isinstance(client, dask.distributed.Client)


def test_dask_workers(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the dask_workers function."""
    assert utilities.dask_workers(dask_client, cores_only=True) == len(
        dask_client.ncores())  # type: ignore
    assert utilities.dask_workers(dask_client, cores_only=False) == sum(
        item for item in dask_client.nthreads().values())  # type: ignore


def test_calculation_stream(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the calculation_stream function."""

    def add_1_return_list(item):
        """Add 1 to each item in the list and return the result in a list."""
        return [item + 1]

    def add_1(item):
        """Add 1 to each item in the list and return the result."""
        return item + 1

    stream = utilities.calculation_stream(add_1_return_list,
                                          iter(list(range(100))),
                                          max_workers=4)
    assert sorted(list(stream)) == [[item + 1] for item in range(100)]

    stream = utilities.calculation_stream(add_1,
                                          iter(list(range(100))),
                                          max_workers=4)
    assert sorted(list(stream)) == [item + 1 for item in range(100)]

    stream = utilities.calculation_stream(add_1, iter([1]), max_workers=4)
    assert sorted(list(stream)) == [2]


def test_split_sequence():
    """Test the split_sequence function."""
    assert list(utilities.split_sequence(list(range(10)), 2)) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]
    assert list(utilities.split_sequence(list(range(10)), 3)) == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert list(utilities.split_sequence(list(range(10)), 4)) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
        [8, 9],
    ]
    assert list(utilities.split_sequence(list(range(10)), 5)) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]
    assert list(utilities.split_sequence(list(range(10)), 6)) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8],
        [9],
    ]
    assert list(utilities.split_sequence(list(range(10)), 7)) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert list(utilities.split_sequence(list(range(10)), 8)) == [
        [0, 1],
        [2, 3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert list(utilities.split_sequence(list(range(10)), 9)) == [
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
    assert list(utilities.split_sequence(list(range(10)), 10)) == [
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
    assert list(utilities.split_sequence(list(range(10)), 11)) == [
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
    assert list(utilities.split_sequence(list(range(10)))) == [
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
        list(utilities.split_sequence(list(range(10)), 0))


def test_normalize_path():
    """Test the normalize_path function."""
    fs = fsspec.filesystem('file')
    root = str(pathlib.Path('/').resolve())
    if platform.system() == 'Windows':
        # fsspec returns only the drive letter for the root path.
        root = root.replace('\\', '')
        sep = '\\'
    else:
        sep = '/'

    assert utilities.normalize_path(fs, '/') == root
    assert utilities.normalize_path(fs, './foo') == str(
        pathlib.Path('.').resolve() / 'foo')

    fs = fsspec.filesystem('memory')
    assert utilities.normalize_path(fs, '/') == sep
    assert utilities.normalize_path(fs, './foo') == f'{sep}foo'

    fs = fsspec.filesystem('s3')
    assert utilities.normalize_path(fs, '/') == '/'
    assert utilities.normalize_path(fs, './foo') == './foo'
