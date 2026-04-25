"""Local-FS store backed by :class:`zarr.storage.LocalStore`."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import zarr.storage

from .base import Store


class LocalStore(Store):
    """File-system backed store rooted at ``path``."""

    def __init__(self, path: str | os.PathLike[str], *, read_only: bool = False) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._store = zarr.storage.LocalStore(str(self._path), read_only=read_only)

    @property
    def root_uri(self) -> str:
        return f"file://{self._path}"

    @property
    def root_path(self) -> Path:
        return self._path

    def zarr_store(self) -> Any:
        return self._store

    def _full(self, key: str) -> Path:
        return self._path / key.replace("/", os.sep)

    def exists(self, key: str) -> bool:
        return self._full(key).exists()

    def read_bytes(self, key: str) -> bytes | None:
        p = self._full(key)
        if not p.exists():
            return None
        return p.read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        if self._read_only:
            raise PermissionError(f"store at {self.root_uri} is read-only")
        p = self._full(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def list_prefix(self, prefix: str) -> Iterator[str]:
        base = self._full(prefix) if prefix else self._path
        if not base.exists():
            return
        for child in base.iterdir():
            yield child.name

    def list_dir(self, prefix: str) -> Iterator[str]:
        base = self._full(prefix) if prefix else self._path
        if not base.is_dir():
            return
        for child in base.iterdir():
            if child.is_dir():
                yield child.name

    def delete_prefix(self, prefix: str) -> None:
        if self._read_only:
            raise PermissionError(f"store at {self.root_uri} is read-only")
        target = self._full(prefix)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    def __repr__(self) -> str:
        return f"LocalStore({self._path!s})"
