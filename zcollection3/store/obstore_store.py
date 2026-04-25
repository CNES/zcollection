"""obstore-backed Store — the recommended S3 path.

obstore is a Rust-backed, async object-store client wrapped by zarr's
``zarr.storage.ObjectStore``. For S3 workloads it is materially faster
than fsspec/aiobotocore and produces fewer GETs on cold opens.

The dependency is optional — we only import obstore when this module is
constructed, so missing it raises a clear, actionable error instead of
breaking ``import zcollection3``.
"""
from __future__ import annotations

from typing import Any, Iterator
from urllib.parse import urlparse

import zarr.storage

from ..errors import StoreError
from ._async_bridge import run_sync, to_list_async
from .base import Store


class ObjectStore(Store):
    """Wrap :class:`zarr.storage.ObjectStore` (obstore) with our Store API.

    Construct from a URL (``s3://bucket/prefix``, ``gs://``, ``az://``,
    ``http(s)://``) plus optional ``client_options`` and ``credentials``
    forwarded to obstore.
    """

    def __init__(
        self,
        url: str,
        *,
        client_options: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        read_only: bool = False,
    ) -> None:
        try:
            import obstore  # noqa: PLC0415  (optional dependency)
        except ImportError as exc:
            raise StoreError(
                "ObjectStore requires the 'obstore' package; "
                "install it via `pip install obstore` or use FsspecStore."
            ) from exc

        self._url = url.rstrip("/")
        self._read_only = read_only

        parsed = urlparse(url)
        if not parsed.scheme:
            raise StoreError(f"ObjectStore needs a scheme://; got {url!r}")

        self._inner_obstore = _build_obstore(obstore, parsed, client_options, credentials)
        self._store = zarr.storage.ObjectStore(self._inner_obstore, read_only=read_only)

    @property
    def root_uri(self) -> str:
        return self._url

    def zarr_store(self) -> Any:
        return self._store

    def exists(self, key: str) -> bool:
        async def _check() -> bool:
            if await self._store.exists(key):
                return True
            async for _ in self._store.list_dir(key.rstrip("/")):
                return True
            return False
        return run_sync(_check())

    def read_bytes(self, key: str) -> bytes | None:
        async def _get() -> bytes | None:
            from zarr.core.buffer import default_buffer_prototype  # noqa: PLC0415

            try:
                buf = await self._store.get(key, prototype=default_buffer_prototype())
            except FileNotFoundError:
                return None
            if buf is None:
                return None
            return bytes(buf.to_bytes())
        return run_sync(_get())

    def write_bytes(self, key: str, data: bytes) -> None:
        if self._read_only:
            raise PermissionError(f"store at {self.root_uri} is read-only")

        async def _put() -> None:
            from zarr.core.buffer import default_buffer_prototype  # noqa: PLC0415

            proto = default_buffer_prototype()
            await self._store.set(key, proto.buffer.from_bytes(data))
        run_sync(_put())

    def list_prefix(self, prefix: str) -> Iterator[str]:
        return iter(to_list_async(self._store.list_dir(prefix.rstrip("/"))))

    def list_dir(self, prefix: str) -> Iterator[str]:
        return self.list_prefix(prefix)

    def delete_prefix(self, prefix: str) -> None:
        if self._read_only:
            raise PermissionError(f"store at {self.root_uri} is read-only")
        run_sync(self._store.delete_dir(prefix.rstrip("/")))

    def __repr__(self) -> str:
        return f"ObjectStore({self._url!r})"


def _build_obstore(
    obstore: Any,
    parsed: Any,
    client_options: dict[str, Any] | None,
    credentials: dict[str, Any] | None,
) -> Any:
    """Construct the underlying ``obstore`` store for a parsed URL."""
    scheme = parsed.scheme
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/") or None

    kwargs: dict[str, Any] = {}
    if client_options:
        kwargs["client_options"] = client_options
    if credentials:
        kwargs.update(credentials)
    if prefix:
        kwargs["prefix"] = prefix

    if scheme == "s3":
        return obstore.store.S3Store(bucket, **kwargs)
    if scheme == "gs":
        return obstore.store.GCSStore(bucket, **kwargs)
    if scheme in {"az", "azure", "abfs"}:
        return obstore.store.AzureStore(bucket, **kwargs)
    if scheme in {"http", "https"}:
        return obstore.store.HTTPStore.from_url(parsed.geturl(), **kwargs)
    raise StoreError(f"ObjectStore does not handle scheme {scheme!r}")
