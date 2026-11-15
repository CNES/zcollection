# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partition index
===============

The partition index of a collection or view is stored in a file at its
root. Reading the index requires one request; listing partitions directly
from storage requires one request per directory.

Each partition key contains a value for every name defined by the
partitioning scheme. The index stores these names and the partition keys,
ordered by their values. A partition path is constructed from its key using
:meth:`zcollection.partitioning.abc.Partitioning.join`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import contextlib
import gzip
import itertools
import json
import logging
import zlib

from . import fs_utils

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import fsspec

    from . import partitioning

#: Name of the partition index file at the root of a collection or view.
FILENAME = '.partitions'

#: Version of the partition index layout.
VERSION = 2

#: Partition key: one value for each name defined by the partitioning scheme.
Key = tuple[int, ...]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class PartitionIndex:
    """Index of partitions in a collection or view.

    Args:
        fs: The file system containing the collection or view.
        base_dir: The collection or view directory.
        partition_handler: The partitioning scheme.
        writable: Whether the index can be written.
    """
    __slots__ = (
        '_base_dir',
        '_fs',
        '_names',
        '_partitioning',
        '_pending',
        '_writable',
    )

    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        base_dir: str,
        partition_handler: partitioning.Partitioning,
        *,
        writable: bool,
    ) -> None:
        self._fs = fs
        self._base_dir = base_dir
        self._partitioning = partition_handler
        self._writable = writable
        self._names: tuple[str, ...] = tuple(
            name for name, _ in partition_handler.dtype())
        self._pending: set[Key] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Return the names associated with the values of a partition key."""
        return self._names

    @property
    def path(self) -> str:
        """Return the path of the partition index file."""
        return fs_utils.join_path(self._base_dir, FILENAME)

    def path_of(self, key: Key) -> str:
        """Return the path of a partition, relative to the root.

        Args:
            key: The key of the partition.

        Returns:
            The path of the partition.
        """
        return self._partitioning.join(
            tuple(zip(self._names, key, strict=True)), self._fs.sep)

    def keys(self) -> tuple[Key, ...]:
        """Return the key of each partition, ordered by their values.

        If the index is absent, unreadable, written in another layout or
        for another partitioning, or marked as updating, the
        partitions are listed from the store. Reading leaves the stored
        index unchanged.

        Returns:
            The key of each partition.
        """
        keys: tuple[Key, ...] | None = self._read()
        if keys is not None:
            return keys

        keys = self.walk()
        _LOGGER.warning(
            'The partitions of %r were listed from the store, because its '
            'index is unavailable or its update is unfinished. '
            'Rebuilding the index with reindex requires a writable view or '
            "a collection opened with mode='w'.", self._base_dir)
        return keys

    @contextlib.contextmanager
    def updating(self, *, reindex_on_error: bool = False) -> Iterator[None]:
        """Mark the index as updating during partition changes.

        On successful exit, write the updated index. After an error, the index
        remains marked as updating unless it is rebuilt successfully.

        Args:
            reindex_on_error: Whether to rebuild the index after an error.
                If rebuilding fails, log the failure and leave the index marked
                as updating.
        """
        keys = set(self.keys())
        self.write(sorted(keys), updating=True)
        self._pending = keys
        try:
            yield
        except Exception:
            if reindex_on_error:
                self._reindex_or_log()
            raise
        finally:
            self._pending = None
        self.write(sorted(keys))

    def _reindex_or_log(self) -> None:
        """Rebuild the index, logging a failure."""
        try:
            self.reindex()
        except Exception:
            _LOGGER.exception('The index of %r could not be rebuilt',
                              self._base_dir)

    def reindex(self) -> tuple[Key, ...]:
        """Rebuild the index from a listing of the partitions in the store.

        Returns:
            The key of each partition, ordered by their values.
        """
        keys = self.walk()
        self.write(keys)
        return keys

    def walk(self) -> tuple[Key, ...]:
        """List the partitions in the store.

        Returns:
            The key of each partition, ordered by their values.
        """
        self._fs.invalidate_cache()
        try:
            paths = tuple(
                self._partitioning.list_partitions(self._fs, self._base_dir))
        except FileNotFoundError:
            if self._fs.exists(self._base_dir):
                raise
            paths = ()
        return self.keys_of(paths)

    def write(self, keys: Iterable[Key], *, updating: bool = False) -> None:
        """Write the index.

        Args:
            keys: The key of each partition, ordered by their values.
            updating: Whether to mark the index as updating.
        """
        if not self._writable:
            raise PermissionError('The partition index is read only')
        data: dict[str, Any] = {
            'version': VERSION,
            'state': 'updating' if updating else 'ready',
            'names': list(self._names),
            'partitions': [list(key) for key in keys],
        }
        fs_utils.replace_file(
            self._fs, self.path,
            gzip.compress(json.dumps(data, separators=(',', ':')).encode(),
                          mtime=0))

    def add(self, paths: Iterable[str]) -> None:
        """Add partitions to the index.

        Args:
            paths: Partition paths, relative to the root or including it.
        """
        self._update(added=self.keys_of(paths), removed=())

    def remove(self, paths: Iterable[str]) -> None:
        """Remove partitions from the index.

        Args:
            paths: Partition paths, relative to the root or including it.
        """
        self._update(added=(), removed=self.keys_of(paths))

    def _update(self, *, added: Iterable[Key], removed: Iterable[Key]) -> None:
        """Write the index with partitions added and removed.

        Args:
            added: The key of each partition to add.
            removed: The key of each partition to remove.
        """
        keys: set[Key] = (set(self.keys())
                          if self._pending is None else self._pending)
        keys |= set(added)
        keys -= set(removed)
        if self._pending is None:
            self.write(sorted(keys))

    def keys_of(self, paths: Iterable[str]) -> tuple[Key, ...]:
        """Return the partition keys corresponding to paths.

        Args:
            paths: Partition paths, relative to the root or including it.

        Returns:
            The partition key for each path.
        """
        handler: partitioning.Partitioning = self._partitioning
        return tuple(
            tuple(value for _, value in handler.parse(path)) for path in paths)

    def _read(self) -> tuple[Key, ...] | None:
        """Read the index.

        Returns:
            The key of each partition, or ``None`` when the index is
            absent, unreadable, written in another layout or for another
            partitioning, or marked as being updated.
        """
        try:
            data: Any = json.loads(
                gzip.decompress(self._fs.cat_file(self.path)))
            if type(data['version']) is not int or data['version'] != VERSION:
                return None
            if data['names'] != list(self._names):
                return None
            if data['state'] != 'ready':
                return None
            rows = data['partitions']
            if not isinstance(rows, list) or any(
                    not isinstance(row, list) or len(row) != len(self._names)
                    or any(type(value) is not int for value in row)
                    for row in rows):
                raise ValueError('Invalid partition keys')
            keys = tuple(tuple(row) for row in rows)
            if any(left >= right for left, right in itertools.pairwise(keys)):
                raise ValueError('Partition keys must be unique and ordered')
            return keys
        except (OSError, ValueError, KeyError, TypeError, EOFError,
                zlib.error):
            return None
