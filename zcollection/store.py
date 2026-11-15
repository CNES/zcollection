# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Access to Zarr stores
=====================

Zarr stores are opened, created and consolidated through this module,
which maps a file system and a path to the corresponding Zarr store. The rest
of the library identifies stores only by their file system and path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar
import warnings

from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
import fsspec.implementations.local
import numpy
import zarr
import zarr.abc.codec
import zarr.abc.store
import zarr.errors
import zarr.storage

from . import codecs
from .codecs import DEFAULT_COMPRESSOR

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numcodecs.abc
    from zarr.core.buffer import NDArrayLikeOrScalar
    from zarr.core.common import JSON

    from .type_hints import DTypeLike, NDArray

__all__ = ('DEFAULT_COMPRESSOR', 'consolidate', 'contains_group',
           'create_array', 'create_group', 'open_array', 'open_group',
           'write_group_attributes')

#: Version of the Zarr format this release writes.
_FORMAT = 3

#: A Zarr group or array stored in a Zarr store.
_Node = TypeVar('_Node', zarr.Array, zarr.Group)

#: Modes supported when opening a Zarr group or array.
_READ_MODES: frozenset[str] = frozenset({'r', 'r+'})

#: Marker for letting Zarr choose the chunk shape.
DEFAULT_CHUNKS = 'auto'

#: Chunk filenames use the ``c`` prefix and dot-separated indices.
#: This places each chunk beside its array metadata. A ``/`` separator
#: would instead create a nested directory for each dimension.
CHUNK_KEY_ENCODING: dict[str, JSON] = {
    'name': 'default',
    'configuration': {
        'separator': '.'
    }
}

#: Prefix of the warning Zarr emits when consolidating group metadata.
#: The Zarr 3 core specification defines the metadata field, and readers
#: that do not recognize it ignore it.
_CONSOLIDATION_WARNING = 'Consolidated metadata is currently not part'


def _store(
    fs: fsspec.AbstractFileSystem,
    path: str,
) -> zarr.abc.store.Store:
    """Return the Zarr store associated with a file system path.

    Args:
        fs: The file system containing the store.
        path: The store path.

    Returns:
        A :class:`zarr.storage.LocalStore` for local files, or a
        :class:`zarr.storage.FsspecStore` for other file systems.
    """
    if isinstance(fs, fsspec.implementations.local.LocalFileSystem):
        return zarr.storage.LocalStore(path)
    mapper = fs.get_mapper(path)
    # Zarr's from_mapper recreates synchronous async-capable file systems,
    # opening a separate S3 session. The wrapper lets Zarr reuse the supplied
    # file system and its session.
    if isinstance(fs, AsyncFileSystem) and not fs.asynchronous:
        return zarr.storage.FsspecStore(fs=AsyncFileSystemWrapper(
            fs, asynchronous=True),
                                        path=mapper.root)
    return zarr.storage.FsspecStore.from_mapper(mapper)


def _open(
    opener: Callable[..., _Node],
    fs: fsspec.AbstractFileSystem,
    path: str,
    mode: str,
) -> _Node:
    """Open a Zarr group or array, trying the current format first.

    If the format-specific call raises a Zarr error, retry without specifying
    a format so Zarr can detect an older one. Specifying the format avoids
    the additional metadata requests required for automatic detection.

    Args:
        opener: The Zarr function used to open the group or array.
        fs: The file system containing the group or array.
        path: The path of the group or array.
        mode: Either ``'r'`` or ``'r+'``.

    Returns:
        The opened group or array.

    Raises:
        ValueError: If ``mode`` is not ``'r'`` or ``'r+'``.
    """
    if mode not in _READ_MODES:
        raise ValueError(f'the mode {mode!r} creates a node; the modes '
                         f'reading one are {sorted(_READ_MODES)}')
    try:
        return opener(store=_store(fs, path), mode=mode, zarr_format=_FORMAT)
    except zarr.errors.BaseZarrError:
        return opener(store=_store(fs, path), mode=mode)


def open_array(
    fs: fsspec.AbstractFileSystem,
    path: str,
    *,
    mode: str = 'r',
) -> zarr.Array:
    """Open a Zarr array.

    Args:
        fs: The file system containing the array.
        path: The array path.
        mode: ``'r'`` for reading or ``'r+'`` for reading and writing.

    Returns:
        The Zarr array.

    Raises:
        ValueError: If the mode would create an array.
        FileNotFoundError: If no array exists at the path.
    """
    return _open(zarr.open_array, fs, path, mode)


def open_ndarray(
    fs: fsspec.AbstractFileSystem,
    path: str,
) -> NDArray:
    """Read every value of an array.

    Args:
        fs: File system containing the array.
        path: Path of the array.

    Returns:
        Values of the array.

    Raises:
        FileNotFoundError: If no array exists at that path.
    """
    return numpy.asarray(open_array(fs, path)[...])


def open_group(
    fs: fsspec.AbstractFileSystem,
    path: str,
    *,
    mode: str = 'r',
) -> zarr.Group:
    """Open a Zarr group using consolidated metadata.

    Args:
        fs: The file system containing the group.
        path: The group path.
        mode: ``'r'`` for reading or ``'r+'`` for reading and writing.

    Returns:
        The Zarr group.

    Raises:
        ValueError: If the mode would create a group.
        FileNotFoundError: If the path does not exist.
        zarr.errors.GroupNotFoundError: If the path exists but contains no
            Zarr group.
    """
    return _open(zarr.open_consolidated, fs, path, mode)


# These parameters match those accepted by Zarr for creating arrays.
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
    """Create a Zarr array.

    Args:
        fs: The file system where the array will be created.
        path: The path of the store.
        shape: The shape of the array.
        dtype: The data type of the array.
        name: The array name within the store. ``None`` creates the array
            at the store root.
        chunks: The chunk shape. Defaults to :data:`DEFAULT_CHUNKS`, which
            lets Zarr choose the shape.
        compressor: The compressor applied to chunks. Defaults to
            :data:`DEFAULT_COMPRESSOR`.
        fill_value: The fill value used for uninitialized portions of the array.
        filters: The codecs applied to values before compression.
        dimensions: The names of the array dimensions.
        attributes: The array attributes.
        overwrite: Whether to replace an existing array at the path.
        write_empty_chunks: Whether to store chunks containing only the
            fill value. Defaults to ``True``.

    Returns:
        The created array.
    """
    return zarr.create_array(
        store=_store(fs, path),
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,  # type: ignore[arg-type]
        compressors=codecs.compressors(compressor),
        # Zarr checks whether each translated codec is valid as a filter.
        filters=codecs.filters(filters),  # type: ignore[arg-type]
        fill_value=fill_value,
        dimension_names=dimensions,
        attributes=attributes,
        overwrite=overwrite,
        chunk_key_encoding=CHUNK_KEY_ENCODING,
        config={'write_empty_chunks': write_empty_chunks})


def create_group(fs: fsspec.AbstractFileSystem, path: str) -> None:
    """Create an empty Zarr group.

    Args:
        fs: The file system for the new group.
        path: The group path.
    """
    zarr.create_group(store=_store(fs, path))


def roundtrip_filters(values: NDArray,
                      filters: Sequence[codecs.Codec]) -> NDArrayLikeOrScalar:
    """Encode and decode values through filters in a temporary memory store.

    Args:
        values: The values to encode.
        filters: The filters applied before serialization.

    Returns:
        The decoded values.
    """
    array = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        data=values,
        # Zarr checks the role of each codec.
        filters=codecs.filters(filters),  # type: ignore[arg-type]
        compressors=None,
        config={'write_empty_chunks': True})
    return array[...]


def contains_group(fs: fsspec.AbstractFileSystem, path: str) -> bool:
    """Check whether a path contains a Zarr group.

    Args:
        fs: The file system to check.
        path: The path to check.

    Returns:
        ``True`` if a group exists at the path, ``False`` otherwise.
    """
    try:
        _open(zarr.open_group, fs, path, 'r')
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        return False
    return True


def write_group_attributes(
    fs: fsspec.AbstractFileSystem,
    path: str,
    attributes: dict[str, Any],
) -> None:
    """Replace all attributes of a Zarr group.

    Args:
        fs: The file system containing the group.
        path: The group path.
        attributes: The new group attributes.
    """
    _open(zarr.open_group, fs, path, 'r+').attrs.put(attributes)


def consolidate(fs: fsspec.AbstractFileSystem, path: str) -> zarr.Group:
    """Consolidate metadata for a Zarr group and its arrays.

    Args:
        fs: The file system containing the group.
        path: The group path.

    Returns:
        The Zarr group with its arrays.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message=_CONSOLIDATION_WARNING,
                                category=zarr.errors.ZarrUserWarning)
        return zarr.consolidate_metadata(_store(fs, path))
