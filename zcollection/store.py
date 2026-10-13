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

from typing import TYPE_CHECKING, Any

import fsspec.implementations.local
import zarr
import zarr.abc.codec
import zarr.abc.store
import zarr.errors
import zarr.storage

from . import codecs
from .codecs import DEFAULT_COMPRESSOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numcodecs.abc

    from .type_hints import DTypeLike

__all__ = ('DEFAULT_COMPRESSOR', 'consolidate', 'contains_group',
           'create_array', 'create_group', 'open_array', 'open_group',
           'write_group_attributes')

#: Marker letting Zarr pick the shape of a chunk.
DEFAULT_CHUNKS = 'auto'


def _store(
    fs: fsspec.AbstractFileSystem,
    path: str,
) -> zarr.abc.store.Store:
    """Map a path on a file system to the store Zarr addresses.

    Args:
        fs: The file system the store is held on.
        path: The path of the store.

    Returns:
        The store. A local file system is addressed by the store Zarr holds
        for it, which creates the directories a chunk is written in.
    """
    if isinstance(fs, fsspec.implementations.local.LocalFileSystem):
        return zarr.storage.LocalStore(path)
    return zarr.storage.FsspecStore.from_mapper(fs.get_mapper(path))


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
    return zarr.open_array(store=_store(fs, path), mode=mode)


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
    return zarr.open_consolidated(store=_store(fs, path), mode=mode)


# The parameters are the ones Zarr names an array with.
def create_array(  # noqa: PLR0913
    fs: fsspec.AbstractFileSystem,
    path: str,
    *,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    name: str | None = None,
    chunks: tuple[int, ...] | Sequence[int] | str = DEFAULT_CHUNKS,
    compressor: numcodecs.abc.Codec | zarr.abc.codec.Codec | str
    | None = DEFAULT_COMPRESSOR,
    fill_value: Any = 0,
    filters: Sequence[numcodecs.abc.Codec | zarr.abc.codec.Codec]
    | None = None,
    dimensions: Sequence[str] | None = None,
    attributes: dict[str, Any] | None = None,
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
        chunks: The shape of a chunk. Defaults to :data:`DEFAULT_CHUNKS`,
            which lets Zarr pick it.
        compressor: The compressor the chunks are held behind. Defaults to
            :data:`DEFAULT_COMPRESSOR`.
        fill_value: The value an uninitialized portion of the array reads
            as.
        filters: The codecs applied to the values before compression.
        dimensions: The name of each dimension of the array.
        attributes: The attributes the array carries.
        overwrite: Whether to replace an array already at that path.
        write_empty_chunks: Whether to write a chunk whose values all equal
            the fill value.

    Returns:
        The array.
    """
    return zarr.create_array(
        store=_store(fs, path),
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,  # type: ignore[arg-type]
        compressors=codecs.compressors(compressor),
        filters=codecs.filters(filters),
        fill_value=fill_value,
        dimension_names=dimensions,
        attributes=attributes,
        overwrite=overwrite,
        config={'write_empty_chunks': write_empty_chunks})


def create_group(fs: fsspec.AbstractFileSystem, path: str) -> None:
    """Create an empty group.

    Args:
        fs: The file system to hold the group on.
        path: The path of the group.
    """
    zarr.create_group(store=_store(fs, path))


def contains_group(fs: fsspec.AbstractFileSystem, path: str) -> bool:
    """Say whether a path holds a group.

    Args:
        fs: The file system to look on.
        path: The path to look at.

    Returns:
        Whether a group is held there.
    """
    try:
        zarr.open_group(store=_store(fs, path), mode='r')
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        return False
    return True


def write_group_attributes(
    fs: fsspec.AbstractFileSystem,
    path: str,
    attributes: dict[str, Any],
) -> None:
    """Hold the attributes of a group, in place of the ones it carries.

    Args:
        fs: The file system the group is held on.
        path: The path of the group.
        attributes: The attributes to hold.
    """
    zarr.open_group(store=_store(fs, path), mode='r+').attrs.put(attributes)


def consolidate(fs: fsspec.AbstractFileSystem, path: str) -> None:
    """Gather the metadata of a group and of its arrays in one entry.

    Args:
        fs: The file system the group is held on.
        path: The path of the group.
    """
    zarr.consolidate_metadata(_store(fs, path))
