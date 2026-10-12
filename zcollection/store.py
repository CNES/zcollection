# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Access to the Zarr store
========================

Every call that opens, creates or consolidates a Zarr store goes through
this module. The rest of the library names a store by a file system and a
path, and knows nothing of the mapping between the two.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import zarr
import zarr.storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    import fsspec
    import numcodecs.abc

    from .type_hints import DTypeLike

#: Marker selecting the compressor Zarr applies when none is named.
DEFAULT_COMPRESSOR = 'default'


def _mapper(fs: fsspec.AbstractFileSystem, path: str) -> fsspec.FSMap:
    """Map a path on a file system to the store Zarr addresses.

    Args:
        fs: The file system the store is held on.
        path: The path of the store.

    Returns:
        The mapping.
    """
    return fs.get_mapper(path)


def open_array(
    fs: fsspec.AbstractFileSystem,
    path: str,
    *,
    mode: str = 'a',
) -> zarr.Array:
    """Open an array.

    Args:
        fs: The file system the array is held on.
        path: The path of the array.
        mode: The mode to open the array with.

    Returns:
        The array.
    """
    return zarr.open_array(_mapper(fs, path), mode=mode)


def open_group(
    fs: fsspec.AbstractFileSystem,
    path: str,
    *,
    mode: str = 'r',
) -> zarr.Group:
    """Open a group through its consolidated metadata.

    Args:
        fs: The file system the group is held on.
        path: The path of the group.
        mode: The mode to open the group with.

    Returns:
        The group.
    """
    return zarr.open_consolidated(
        _mapper(fs, path),  # type: ignore[arg-type]
        mode=mode)


# The parameters are the ones Zarr names an array with.
def create_array(  # noqa: PLR0913
    fs: fsspec.AbstractFileSystem,
    path: str,
    *,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    name: str | None = None,
    chunks: tuple[int, ...] | list[int] | bool = True,
    compressor: numcodecs.abc.Codec | str | None = DEFAULT_COMPRESSOR,
    fill_value: object = 0,
    filters: Sequence[numcodecs.abc.Codec] | None = None,
    overwrite: bool = False,
    write_empty_chunks: bool = True,
) -> zarr.Array:
    """Create an array.

    Args:
        fs: The file system to hold the array on.
        path: The path of the store.
        shape: The shape of the array.
        dtype: The data type of the array.
        name: The key of the array inside the store. Defaults to ``None``,
            which holds the array at the root of the store.
        chunks: The shape of a chunk. Defaults to ``True``, which lets Zarr
            pick it.
        compressor: The compressor the chunks are held behind. Defaults to
            :data:`DEFAULT_COMPRESSOR`.
        fill_value: The value an uninitialized portion of the array reads
            as.
        filters: The codecs applied to the values before compression.
        overwrite: Whether to replace an array already at that path.
        write_empty_chunks: Whether to write a chunk whose values all equal
            the fill value.

    Returns:
        The array.
    """
    return zarr.create(
        shape=shape,
        chunks=chunks,  # type: ignore[arg-type]
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,  # type: ignore[arg-type]
        store=_mapper(fs, path),
        path=name,
        overwrite=overwrite,
        filters=filters,
        write_empty_chunks=write_empty_chunks)


def create_group(fs: fsspec.AbstractFileSystem, path: str) -> None:
    """Create an empty group.

    Args:
        fs: The file system to hold the group on.
        path: The path of the group.
    """
    zarr.storage.init_group(store=_mapper(fs, path))


def consolidate(fs: fsspec.AbstractFileSystem, path: str) -> None:
    """Gather the metadata of a group and of its arrays in one entry.

    Args:
        fs: The file system the group is held on.
        path: The path of the group.
    """
    zarr.consolidate_metadata(_mapper(fs, path))  # type: ignore[arg-type]
