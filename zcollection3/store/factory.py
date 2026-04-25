"""URL-driven store factory."""
from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from ..errors import StoreError
from .local import LocalStore
from .memory import MemoryStore

if TYPE_CHECKING:
    from .base import Store


def open_store(
    path: str,
    *,
    read_only: bool = False,
) -> Store:
    """Open a Store given a URL or filesystem path.

    Phase 1 supports ``file://`` and ``memory://``. ``s3://``, ``icechunk://``
    arrive in later phases.
    """
    if path == "memory://" or path.startswith("memory://"):
        return MemoryStore()

    parsed = urlparse(path)
    scheme = parsed.scheme or "file"

    if scheme == "file":
        local = parsed.path or path
        if path.startswith("file://"):
            local = parsed.path
        return LocalStore(local, read_only=read_only)

    if scheme in {"s3", "gs", "az", "https", "http"}:
        raise StoreError(
            f"scheme {scheme!r} is not supported in Phase 1 of the rewrite; "
            "S3/cloud arrives in Phase 3."
        )
    if scheme == "icechunk":
        raise StoreError("Icechunk integration arrives in Phase 5.")

    raise StoreError(f"unrecognised store URL: {path!r}")
