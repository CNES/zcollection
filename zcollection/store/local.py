# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Local-FS store backed by :class:`zarr.storage.LocalStore`."""

from typing import Any
from collections.abc import Iterator
import os
from pathlib import Path
import shutil

import zarr.storage

from .base import Store


class LocalStore(Store):
    """File-system backed store rooted at ``path``."""

    def __init__(
        self, path: str | os.PathLike[str], *, read_only: bool = False
    ) -> None:
        """Initialize the store, creating the root directory if needed.

        Args:
            path: Filesystem path of the store root.
            read_only: When true, writes raise :class:`PermissionError`.

        """
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._store = zarr.storage.LocalStore(
            str(self._path), read_only=read_only
        )

    @property
    def root_uri(self) -> str:
        """Return the ``file://`` URI of the store root."""
        return f"file://{self._path}"

    @property
    def root_path(self) -> Path:
        """Return the filesystem path of the store root."""
        return self._path

    def zarr_store(self) -> Any:
        """Return the underlying :class:`zarr.storage.LocalStore`."""
        return self._store

    def _full(self, key: str) -> Path:
        return self._path / key.replace("/", os.sep)

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists on disk."""
        return self._full(key).exists()

    def read_bytes(self, key: str) -> bytes | None:
        """Return the raw bytes at ``key`` or ``None`` if absent."""
        p = self._full(key)
        if not p.exists():
            return None
        return p.read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key`` atomically via a temporary file."""
        if self._read_only:
            raise PermissionError(f"store at {self.root_uri} is read-only")
        p = self._full(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield direct children (files and dirs) under ``prefix``."""
        base = self._full(prefix) if prefix else self._path
        if not base.exists():
            return
        for child in base.iterdir():
            yield child.name

    def list_dir(self, prefix: str) -> Iterator[str]:
        """Yield direct sub-directory names under ``prefix``."""
        base = self._full(prefix) if prefix else self._path
        if not base.is_dir():
            return
        for child in base.iterdir():
            if child.is_dir():
                yield child.name

    def delete_prefix(self, prefix: str) -> None:
        """Recursively delete everything under ``prefix``."""
        if self._read_only:
            raise PermissionError(f"store at {self.root_uri} is read-only")
        target = self._full(prefix)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"LocalStore({self._path!s})"
