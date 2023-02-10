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

import fsspec
import fsspec.implementations.local

from .. import fs_utils
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint: disable=unused-import


def test_join_path():
    """Test the join_path function."""
    assert fs_utils.join_path('a', 'b', 'c') == 'a/b/c'
    assert fs_utils.join_path('a', 'b', 'c', 'd') == 'a/b/c/d'
    assert fs_utils.join_path('a', 'b', 'c', 'd', 'e') == 'a/b/c/d/e'
    assert fs_utils.join_path('a', 'b', 'c', 'd', 'e', 'f') == 'a/b/c/d/e/f'


def test_get_fs():
    """Test the get_fs function."""
    fs = fs_utils.get_fs('file')
    assert isinstance(fs, fsspec.implementations.local.LocalFileSystem)
    fs = fs_utils.get_fs()
    assert isinstance(fs, fsspec.implementations.local.LocalFileSystem)


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

    fs = fs_utils.get_fs()
    listing1 = []
    for root, _dirs, files in fs_utils.fs_walk(fs, tmpdir, sort=True):
        for item in files:
            listing1.append(fs.sep.join([root, item]))

    listing2 = []
    for root, _dirs, files in fs_utils.fs_walk(fs, tmpdir, sort=False):
        for item in files:
            listing2.append(fs.sep.join([root, item]))

    assert listing1 == sorted(listing2)

    assert list(
        fs_utils.fs_walk(fs,
                         str(pathlib.Path(tmpdir).joinpath('inexistent')),
                         sort=True)) == [('', [], [])]


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

    assert fs_utils.normalize_path(fs, '/') == root
    assert fs_utils.normalize_path(fs, './foo') == str(
        pathlib.Path('.').resolve() / 'foo')

    fs = fsspec.filesystem('memory')
    assert fs_utils.normalize_path(fs, '/') == sep
    assert fs_utils.normalize_path(fs, './foo') == f'{sep}foo'

    fs = fsspec.filesystem('s3')
    assert fs_utils.normalize_path(fs, '/') == '/'
    assert fs_utils.normalize_path(fs, './foo') == './foo'
