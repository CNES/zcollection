# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing utilities
=================
"""
import os
import pathlib
import platform

import fsspec
import fsspec.implementations.local

from .. import fs_utils
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint: disable=unused-import

#: Test data
TEXT = '''Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam porta
turpis dictum, porta tellus eu, convallis mi. Integer at placerat diam. Donec in
various neque. Morbi sed nisi finibus, mattis velit non, pulvinar metus. Duis
feugiat diam eget augue posuere, nec aliquam dolor tristique. Aliquam a dolor
vel ante sagittis dictum vel at dolor. Suspendisse velit dolor, vestibulum eget
aliquet ut, imperdiet at justo. Nullam sit amet suscipit orci, bibendum sagittis
orci. Aliquam mattis feugiat rutrum. Vivamus fermentum ex non mauris faucibus
vehicula. Donec odio lacus, viverra et hendrerit eu, mollis eget mauris. Duis
suscipit, velit nec finibus ullamcorper, nisi lorem fermentum tellus, ut viverra
nunc lorem ut odio. Duis eget ligula maximus, venenatis nulla a, commodo dolor.
Aenean justo sapien, mollis aliquam vestibulum id, suscipit a ligula. Phasellus
porta arcu erat, elementum faucibus leo auctor vel. Integer vel pharetra leo.'''


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
    for root, _dirs, files in fs_utils.fs_walk(fs, str(tmpdir), sort=True):
        for item in files:
            listing1.append(fs.sep.join([root, item]))

    listing2 = []
    for root, _dirs, files in fs_utils.fs_walk(fs, str(tmpdir), sort=False):
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

    def istrcmp(str1, str2):
        """Case insensitive string comparison."""
        return str1.lower() == str2.lower()

    assert istrcmp(fs_utils.normalize_path(fs, '/'), root)
    assert istrcmp(fs_utils.normalize_path(fs, './foo'),
                   str(pathlib.Path('.').resolve() / 'foo'))

    fs = fsspec.filesystem('memory')
    assert fs_utils.normalize_path(fs, '/') == os.path.sep
    assert fs_utils.normalize_path(fs, './foo') == f'{os.path.sep}foo'

    fs = fsspec.filesystem('s3')
    assert fs_utils.normalize_path(fs, '/') == '/'
    assert fs_utils.normalize_path(fs, './foo') == './foo'


def test_copy_file(tmpdir):
    """Test the copy file across different file systems."""
    fs_source = fsspec.filesystem('file')
    fs_target = fsspec.filesystem('memory')
    path = str(tmpdir / 'foo.txt')
    with fs_source.open(path, mode='wb', encoding='utf-8') as stream:
        stream.write(TEXT.encode('utf-8'))
    fs_utils.copy_file(path, 'foo.txt', fs_source, fs_target)

    assert fs_target.cat('foo.txt').decode('utf-8') == TEXT


def test_copy_files(tmpdir):
    """Test the copy files across different file systems."""
    source = tmpdir / 'source'
    target = tmpdir / 'target'
    fs_source = fsspec.filesystem('file')
    fs_target = fsspec.filesystem('file')
    fs_source.mkdir(source)
    fs_target.mkdir(target)
    paths = [
        str(source / item) for item in (
            'foo.txt',
            'bar.txt',
            'baz.txt',
        )
    ]
    for path in paths:
        with fs_source.open(path, mode='wb', encoding='utf-8') as stream:
            stream.write(TEXT.encode('utf-8'))
    fs_utils.copy_files(paths, str(target), fs_source, fs_target)

    for item in fs_target.ls(str(target)):
        assert fs_target.cat(item).decode('utf-8') == TEXT


def test_copy_tree(tmpdir):
    """Test the copy tree across different file systems."""
    fs_source = fsspec.filesystem('file')
    fs_target = fsspec.filesystem('memory')

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
        path = fs_utils.join_path(str(tmpdir), *item)
        fs_source.makedirs(path, exist_ok=False)
        if 'day' in item[-1]:
            with fs_source.open(fs_utils.join_path(path, f'file_{ix}.txt'),
                                mode='wb',
                                encoding='utf-8') as stream:
                stream.write(TEXT.encode('utf-8'))

    fs_utils.copy_tree(str(tmpdir), '/tree', fs_source, fs_target)

    for root, dirs, files in fs_utils.fs_walk(fs_target, '/tree'):
        for item in files:
            assert fs_target.cat(fs_utils.join_path(
                root, item)).decode('utf-8') == TEXT
        for item in dirs:
            item = item.replace('\\', '/')
            parts = item.replace('/tree/', '').split(fs_target.sep)
            assert parts[0] == 'year=2014'
            if len(parts) > 1:
                assert parts[1] in ['month=4', 'month=5']
            if len(parts) > 2:
                assert 'day=' in parts[2]
            if len(parts) > 3:
                assert 'file_' in parts[3]
                assert parts[3].endswith('.txt')
