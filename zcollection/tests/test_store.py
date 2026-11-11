# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the store module
========================

The defaults of :func:`zcollection.store.create_array` decide what an array
holds where nothing was written to it, and what its chunks are held behind.
A storage layer that changes them changes the data without raising, so they
are pinned here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast
import contextlib
from unittest import mock
import warnings

import numcodecs
import numpy
import pytest
import zarr

from .. import store
from ..fs_utils import join_path
# pylint: disable=unused-import # Need to import for fixtures
from .fs import local_fs  # noqa: F401

# pylint: enable=unused-import

if TYPE_CHECKING:
    from collections.abc import Iterator

    import fsspec


@contextlib.contextmanager
def keys_read(fs: fsspec.AbstractFileSystem, path: str) -> Iterator[list[str]]:
    """Record the keys read from a store.

    The record holds the keys read from every store of the kind that
    holds this path.

    Args:
        fs: The file system the store is held on.
        path: The path of the store.

    Yields:
        The keys, in the order they are read.
    """
    #: pylint: disable=protected-access
    target = type(store._store(fs, path))
    #: pylint: enable=protected-access
    held: list[str] = []
    asked = target.get

    async def record(self, key, *args, **kwargs):
        held.append(key)
        return await asked(self, key, *args, **kwargs)

    with mock.patch.object(target, 'get', record):
        yield held


def test_uninitialized_values_read_as_zero(local_fs) -> None:  # noqa: F811
    """An array created without a fill value reads as zero."""
    path = join_path(str(local_fs.root), 'array')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'))

    assert array.fill_value == 0
    numpy.testing.assert_array_equal(array[...], numpy.zeros(4))


def test_s3_stores_do_not_open_extra_sessions(s3_fs) -> None:
    """Zarr operations open no S3 session on a second filesystem."""
    fs = s3_fs.fs
    unexpected_sessions: list[int] = []
    set_session = type(fs).set_session

    async def record_session(owner, *args, **kwargs):
        if owner is not fs:
            unexpected_sessions.append(id(owner))
        return await set_session(owner, *args, **kwargs)

    with mock.patch.object(type(fs), 'set_session', record_session):
        for index in range(2):
            path = join_path(str(s3_fs.collection), f'array{index}')
            array = store.create_array(fs,
                                       path,
                                       shape=(4, ),
                                       dtype=numpy.dtype('int32'))
            array[...] = [1, 2, 3, 4]
            numpy.testing.assert_array_equal(
                store.open_array(fs, path)[...], [1, 2, 3, 4])

    assert not unexpected_sessions


def test_fill_value_is_the_one_given(local_fs) -> None:  # noqa: F811
    """An array created with a fill value reads as that value."""
    path = join_path(str(local_fs.root), 'array')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'),
                               fill_value=-999.0)

    assert array.fill_value == -999.0
    numpy.testing.assert_array_equal(array[...], numpy.full(4, -999.0))


def test_no_fill_value_becomes_the_default_of_the_type(
        local_fs) -> None:  # noqa: F811
    """An array always carries a fill value.

    A fill value of ``None`` becomes the default of the data type. A
    variable that carries none is marked by the absence of
    :data:`zcollection.variable.abc.FILL_VALUE`.
    """
    path = join_path(str(local_fs.root), 'array')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'),
                               fill_value=None)

    assert array.fill_value == 0.0


def test_chunks_are_held_behind_the_default_compressor(
        local_fs) -> None:  # noqa: F811
    """An array created without a compressor carries the Zarr default."""
    path = join_path(str(local_fs.root), 'array')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'))

    assert [
        cast('zarr.abc.codec.BytesBytesCodec', codec).to_dict()['name']
        for codec in array.compressors
    ] == ['zstd']


def test_filters_are_applied(local_fs) -> None:  # noqa: F811
    """The values pass through the filters before compression."""
    path = join_path(str(local_fs.root), 'array')
    codec = numcodecs.FixedScaleOffset(offset=0,
                                       scale=1000,
                                       dtype='f8',
                                       astype='i4')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'),
                               filters=(codec, ))
    array[...] = [0.001, 0.002, 0.003, 0.004]

    assert [
        cast('zarr.abc.codec.ArrayArrayCodec', item).to_dict()['name']
        for item in array.filters
    ] == ['numcodecs.fixedscaleoffset']
    numpy.testing.assert_allclose(cast('numpy.ndarray', array[...]),
                                  [0.001, 0.002, 0.003, 0.004])


def test_name_holds_the_array_inside_the_store(local_fs) -> None:  # noqa: F811
    """``name`` is the key the array is held under."""
    path = join_path(str(local_fs.root), 'group')
    store.create_group(local_fs.fs, path)
    store.create_array(local_fs.fs,
                       path,
                       name='variable',
                       shape=(4, ),
                       dtype=numpy.dtype('int32'))
    store.consolidate(local_fs.fs, path)

    group = store.open_group(local_fs.fs, path)
    assert set(group) == {'variable'}
    assert cast('zarr.Array', group['variable']).shape == (4, )


def test_overwrite_replaces_an_array(local_fs) -> None:  # noqa: F811
    """``overwrite`` replaces an array already held at that path."""
    path = join_path(str(local_fs.root), 'array')
    store.create_array(local_fs.fs,
                       path,
                       shape=(4, ),
                       dtype=numpy.dtype('int32'))
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(8, ),
                               dtype=numpy.dtype('int32'),
                               overwrite=True)

    assert array.shape == (8, )
    assert store.open_array(local_fs.fs, path).shape == (8, )


def test_creating_over_an_array_is_refused(local_fs) -> None:  # noqa: F811
    """Creating an array where one already is raises."""
    path = join_path(str(local_fs.root), 'array')
    store.create_array(local_fs.fs,
                       path,
                       shape=(4, ),
                       dtype=numpy.dtype('int32'))

    with pytest.raises(zarr.errors.ContainsArrayError):
        store.create_array(local_fs.fs,
                           path,
                           shape=(8, ),
                           dtype=numpy.dtype('int32'))


def test_an_empty_group_holds_no_array(local_fs) -> None:  # noqa: F811
    """A group created on its own holds nothing."""
    path = join_path(str(local_fs.root), 'group')
    store.create_group(local_fs.fs, path)
    store.consolidate(local_fs.fs, path)

    assert set(store.open_group(local_fs.fs, path)) == set()


def test_a_group_opens_through_its_consolidated_metadata(
        local_fs) -> None:  # noqa: F811
    """A group that was never consolidated does not open."""
    path = join_path(str(local_fs.root), 'group')
    store.create_group(local_fs.fs, path)

    with pytest.raises(ValueError, match='Consolidated metadata'):
        store.open_group(local_fs.fs, path)

    store.consolidate(local_fs.fs, path)
    assert store.open_group(local_fs.fs, path) is not None


def test_a_read_only_array_refuses_a_write(local_fs) -> None:  # noqa: F811
    """An array opened for reading cannot be written to."""
    path = join_path(str(local_fs.root), 'array')
    store.create_array(local_fs.fs,
                       path,
                       shape=(4, ),
                       dtype=numpy.dtype('int32'))

    array = store.open_array(local_fs.fs, path, mode='r')
    with pytest.raises(ValueError, match='read-only'):
        array[...] = 1


def test_a_group_opens_read_only(local_fs) -> None:  # noqa: F811
    """A group opened with the default mode refuses a write.

    Consolidated metadata is read-only whatever the mode; a group opened
    in ``r+`` hands back writable arrays.
    """
    path = join_path(str(local_fs.root), 'group')
    store.create_group(local_fs.fs, path)
    store.create_array(local_fs.fs,
                       path,
                       name='variable',
                       shape=(4, ),
                       dtype=numpy.dtype('int32'))
    store.consolidate(local_fs.fs, path)

    group = store.open_group(local_fs.fs, path)
    assert group.read_only
    with pytest.raises(ValueError, match='read-only'):
        cast('zarr.Array', group['variable'])[...] = 1

    assert not store.open_group(local_fs.fs, path, mode='r+').read_only


def test_consolidation_says_nothing(local_fs) -> None:  # noqa: F811
    """Consolidating a group emits no warning.

    The core specification names the field a group holds the consolidated
    metadata in, and a reader that does not know it leaves it alone.
    """
    path = join_path(str(local_fs.root), 'group')
    store.create_group(local_fs.fs, path)
    store.create_array(local_fs.fs,
                       path,
                       name='variable',
                       shape=(4, ),
                       dtype=numpy.dtype('int32'))

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        store.consolidate(local_fs.fs, path)
        quiet = list(seen)
        warnings.warn('another matter',
                      zarr.errors.ZarrUserWarning,
                      stacklevel=1)

    assert not quiet
    assert [str(item.message) for item in seen] == ['another matter']


@pytest.mark.parametrize('mode', ['a', 'w', 'w-'])
def test_opening_a_node_with_a_creating_mode_raises(
        mode, local_fs) -> None:  # noqa: F811
    """Both openers refuse a mode that creates a node."""
    path = join_path(str(local_fs.root), 'group')
    store.create_group(local_fs.fs, path)

    with pytest.raises(ValueError, match='creates a node'):
        store.open_group(local_fs.fs, path, mode=mode)

    with pytest.raises(ValueError, match='creates a node'):
        store.open_array(local_fs.fs, path, mode=mode)


def test_opening_an_absent_array_raises(local_fs) -> None:  # noqa: F811
    """Opening an array the store does not hold builds none of it."""
    path = join_path(str(local_fs.root), 'absent')

    with pytest.raises(FileNotFoundError):
        store.open_array(local_fs.fs, path)

    assert not local_fs.fs.exists(path)


def test_opening_a_node_reads_its_metadata_once(
        local_fs) -> None:  # noqa: F811
    """A group and an array each cost one request to open.

    The key read is the metadata of the format this release writes.
    """
    path = join_path(str(local_fs.root), 'group')
    store.create_group(local_fs.fs, path)
    store.create_array(local_fs.fs,
                       path,
                       name='variable',
                       shape=(4, ),
                       dtype=numpy.dtype('int32'))
    store.consolidate(local_fs.fs, path)

    with keys_read(local_fs.fs, path) as group_keys:
        store.open_group(local_fs.fs, path)

    with keys_read(local_fs.fs, path) as array_keys:
        store.open_array(local_fs.fs, join_path(path, 'variable'))

    assert group_keys == ['zarr.json']
    assert array_keys == ['zarr.json']
