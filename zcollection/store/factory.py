# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""URL-driven store factory."""

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ..errors import StoreError
from .local import LocalStore
from .memory import MemoryStore


if TYPE_CHECKING:
    from .base import Store

_CLOUD_SCHEMES = {"s3", "gs", "az", "azure", "abfs", "http", "https"}


def open_store(
    path: str,
    *,
    read_only: bool = False,
    storage_options: dict[str, Any] | None = None,
) -> Store:
    """Open a Store given a URL or filesystem path.

    Schemes:

    - ``file://`` (default for bare paths) → :class:`LocalStore`
    - ``memory://`` → :class:`MemoryStore`
    - ``s3://``, ``gs://``, ``az://``, ``http(s)://`` → :class:`ObjectStore`
      (obstore-backed, the only cloud path)
    - ``icechunk://`` → :class:`IcechunkStore` (transactional)
    """
    if path == "memory://" or path.startswith("memory://"):
        return MemoryStore()

    # A path without ``://`` is a bare filesystem path. We route it
    # straight to ``LocalStore`` instead of feeding it to
    # :func:`urllib.parse.urlparse`, which would otherwise interpret a
    # Windows drive letter (``C:\foo``) as a one-character URL scheme
    # and raise "unrecognised store URL".
    if "://" not in path:
        return LocalStore(path, read_only=read_only)

    parsed = urlparse(path)
    scheme = parsed.scheme

    if scheme == "file":
        return LocalStore(parsed.path, read_only=read_only)

    if scheme == "icechunk":
        from .icechunk_store import IcechunkStore

        # ``icechunk://`` carries a filesystem path; remote storage goes
        # through the IcechunkStore constructor with a prebuilt Storage.
        local = parsed.path or path[len("icechunk://") :]
        return IcechunkStore(local, read_only=read_only)

    if scheme in _CLOUD_SCHEMES:
        from .obstore_store import ObjectStore

        return ObjectStore(
            path,
            client_options=storage_options,
            read_only=read_only,
        )

    raise StoreError(f"unrecognised store URL: {path!r}")
