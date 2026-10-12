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

import numcodecs
import numpy
import pytest
import zarr

from .. import store
from ..fs_utils import join_path
# pylint: disable=unused-import # Need to import for fixtures
from .fs import local_fs  # noqa: F401

# pylint: enable=unused-import


def test_uninitialized_values_read_as_zero(local_fs) -> None:  # noqa: F811
    """An array created without a fill value reads as zero."""
    path = join_path(str(local_fs.root), 'array')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'))

    assert array.fill_value == 0
    numpy.testing.assert_array_equal(array[...], numpy.zeros(4))


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


def test_no_fill_value_leaves_the_array_unmasked(
        local_fs) -> None:  # noqa: F811
    """A fill value of ``None`` is kept as ``None``."""
    path = join_path(str(local_fs.root), 'array')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'),
                               fill_value=None)

    assert array.fill_value is None


def test_chunks_are_held_behind_the_default_compressor(
        local_fs) -> None:  # noqa: F811
    """An array created without a compressor carries the Zarr default."""
    path = join_path(str(local_fs.root), 'array')
    array = store.create_array(local_fs.fs,
                               path,
                               shape=(4, ),
                               dtype=numpy.dtype('float64'))

    assert array.compressor == zarr.storage.default_compressor


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

    assert array.filters == [codec]
    numpy.testing.assert_allclose(array[...], [0.001, 0.002, 0.003, 0.004])


def test_name_holds_the_array_inside_the_store(local_fs) -> None:  # noqa: F811
    """``name`` holds the array under that key instead of at the root."""
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
    assert group['variable'].shape == (4, )


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

    with pytest.raises(KeyError):
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
    with pytest.raises(zarr.errors.ReadOnlyError):
        array[...] = 1


def test_a_group_opens_read_only(local_fs) -> None:  # noqa: F811
    """A group opened with the default mode refuses a write.

    Consolidated metadata is read-only whatever the mode, the chunks are
    not: a group opened in ``r+`` hands back writable arrays.
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
    with pytest.raises(zarr.errors.ReadOnlyError):
        group['variable'][...] = 1

    assert not store.open_group(local_fs.fs, path, mode='r+').read_only
