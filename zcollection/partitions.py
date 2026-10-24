# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Index of the partitions
=======================

The index of the partitions a collection or a view holds, kept in one
file at its root. A listing reads that file instead of walking the
directory tree, which costs one request per directory it visits.

A partition is named by its key, the value each of the names of the
partitioning takes. The index holds the key of each partition, ordered
by their values, and the names. A path is built from a key by
:meth:`zcollection.partitioning.abc.Partitioning.join`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import gzip
import json
import logging

from . import fs_utils

if TYPE_CHECKING:
    from collections.abc import Iterable

    import fsspec

    from . import partitioning

#: Name of the file holding the index, at the root of the collection or
#: of the view.
FILENAME = '.partitions'

#: Version of the layout the index is written in.
VERSION = 1

#: The key of a partition: the value each of its names takes.
Key = tuple[int, ...]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class PartitionIndex:
    """The index of the partitions of a collection or of a view.

    Args:
        fs: The file system holding the store.
        base_dir: The path of the collection or of the view.
        partition_handler: The partitioning the store is written with.
        writable: Whether the index may be written. One that may neither
            be read nor written is built in memory and held for the
            lifetime of this instance.
    """
    __slots__ = ('_base_dir', '_fs', '_held', '_names', '_partitioning',
                 '_writable')

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
        self._held: tuple[Key, ...] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Return the name the partitioning gives to each value of a key."""
        return self._names

    @property
    def path(self) -> str:
        """Return the path of the file holding the index."""
        return fs_utils.join_path(self._base_dir, FILENAME)

    def path_of(self, key: Key) -> str:
        """Return the path of a partition, relative to the root.

        Args:
            key: The key of the partition.

        Returns:
            The path of the partition.
        """
        return self._partitioning.join(tuple(zip(self._names, key)),
                                       self._fs.sep)

    def keys(self) -> tuple[Key, ...]:
        """Return the key of each partition, ordered by their values.

        The index is read. One that is absent, unreadable, written in
        another layout or for another partitioning is replaced by a walk
        of the directory tree, written back when this index may be
        written.

        Returns:
            The key of each partition.
        """
        keys: tuple[Key, ...] | None = self._read()
        if keys is not None:
            return keys

        # An index written since the walk below is read above, so a reader
        # that could not write one follows a writer that could.
        if self._held is not None:
            return self._held

        keys = self.walk()
        if self._writable and self._publish(keys):
            return keys

        self._held = keys
        _LOGGER.warning(
            'The partitions of %r were read by walking its directory tree, '
            'and the result is held for this session alone. Opening it for '
            'writing, or calling reindex on it, writes the index.',
            self._base_dir)
        return keys

    def _publish(self, keys: Iterable[Key]) -> bool:
        """Write the index, and return whether it was written.

        Args:
            keys: The key of each partition, ordered by their values.

        Returns:
            Whether the index was written.
        """
        try:
            self.write(keys)
        except Exception as err:
            # A read-only credential, a read-only mount or a quota stops
            # the write. The index is state this instance can rebuild, so a
            # listing goes on without it.
            _LOGGER.warning(
                'The index of the partitions of %r could not be '
                'written: %s', self._base_dir, err)
            return False
        return True

    def walk(self) -> tuple[Key, ...]:
        """Return the key of each partition the store holds.

        Returns:
            The key of each partition, ordered by their values.
        """
        handler: partitioning.Partitioning = self._partitioning
        return tuple(
            sorted(
                tuple(value for _, value in handler.parse(path))
                for path in handler.list_partitions(self._fs, self._base_dir)))

    def write(self, keys: Iterable[Key]) -> None:
        """Write the index.

        Args:
            keys: The key of each partition, ordered by their values.
        """
        data: dict[str, Any] = {
            'version': VERSION,
            'names': list(self._names),
            'partitions': [list(key) for key in keys],
        }
        fs_utils.replace_file(
            self._fs, self.path,
            gzip.compress(json.dumps(data, separators=(',', ':')).encode()))
        self._held = None

    def add(self, paths: Iterable[str]) -> None:
        """Add partitions to the index.

        Args:
            paths: The path of each partition to add, relative to the root
                or not.
        """
        self._update(added=self.keys_of(paths), removed=())

    def remove(self, paths: Iterable[str]) -> None:
        """Remove partitions from the index.

        Args:
            paths: The path of each partition to remove, relative to the
                root or not.
        """
        self._update(added=(), removed=self.keys_of(paths))

    def _update(self, *, added: Iterable[Key], removed: Iterable[Key]) -> None:
        """Write the index with partitions added and removed.

        Args:
            added: The key of each partition to add.
            removed: The key of each partition to remove.
        """
        keys: set[Key] = set(self.keys())
        keys |= set(added)
        keys -= set(removed)
        self.write(sorted(keys))

    def keys_of(self, paths: Iterable[str]) -> tuple[Key, ...]:
        """Return the key each path names.

        Args:
            paths: The path of each partition, relative to the root or
                not.

        Returns:
            The key of each path.
        """
        handler: partitioning.Partitioning = self._partitioning
        return tuple(
            tuple(value for _, value in handler.parse(path)) for path in paths)

    def _read(self) -> tuple[Key, ...] | None:
        """Read the index.

        Returns:
            The key of each partition, or ``None`` when the index is
            absent, unreadable, written in another layout or for another
            partitioning.
        """
        try:
            data: Any = json.loads(
                gzip.decompress(self._fs.cat_file(self.path)))
            if data['version'] != VERSION:
                return None
            if tuple(data['names']) != self._names:
                return None
            return tuple(tuple(key) for key in data['partitions'])
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError):
            _LOGGER.warning(
                'The index of the partitions of %r is unreadable and is '
                'replaced by a walk of its directory tree.', self._base_dir)
            return None
