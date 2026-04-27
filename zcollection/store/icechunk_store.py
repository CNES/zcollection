# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Icechunk-backed store with transactional commit/rollback.

Icechunk is a transactional log layered over object storage; every write
goes into a :class:`Session`, which the caller commits or discards
atomically. We expose that as :class:`StoreSession`'s ``commit`` /
``rollback`` so collection-level mutations (``insert``, ``drop``,
``update``) become a single transaction.

Key impedance mismatch: icechunk only accepts *zarr-formatted* keys —
``zarr.json`` payloads and chunk paths. zcollection's small JSON sidecars
(``_zcollection.json``, ``_catalog/state.json``, view config) don't fit,
so we route them into the ``attributes`` of dedicated tiny zarr groups
under :data:`META_DIR`. From the caller's perspective the
:meth:`read_bytes` / :meth:`write_bytes` API is unchanged.
"""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterator
from contextlib import contextmanager
import json

from zarr.core.buffer import default_buffer_prototype

from ._async_bridge import run_sync, to_list_async
from .base import Store, StoreSession


if TYPE_CHECKING:  # pragma: no cover
    import icechunk

#: Reserved top-level group that holds zcollection's non-zarr config blobs.
META_DIR: str = "_zc_meta"

_PAYLOAD_ATTR = "_payload"
_GROUP_DOC = json.dumps(
    {"node_type": "group", "zarr_format": 3, "attributes": {}},
    separators=(",", ":"),
).encode("utf-8")


def _is_zarr_key(key: str) -> bool:
    """Keys icechunk accepts directly (zarr.json or chunk paths)."""
    return (
        key == "zarr.json"
        or key.endswith("/zarr.json")
        or key.startswith("c/")
        or "/c/" in key
    )


def _slug(key: str) -> str:
    return key.replace("/", "__")


def _meta_path(key: str) -> str:
    return f"{META_DIR}/{_slug(key)}/zarr.json"


def _meta_doc(payload: bytes) -> bytes:
    return json.dumps(
        {
            "node_type": "group",
            "zarr_format": 3,
            "attributes": {_PAYLOAD_ATTR: payload.decode("utf-8")},
        },
        separators=(",", ":"),
    ).encode("utf-8")


class IcechunkSession(StoreSession):
    """User-facing handle to set a commit message inside ``with store.session():``."""

    transactional: bool = True

    def __init__(self, owner: IcechunkStore) -> None:
        """Initialize the session.

        Args:
            owner: The :class:`IcechunkStore` whose session this wraps.

        """
        self._owner = owner
        self.message: str | None = None

    def commit(self, message: str | None = None) -> None:
        """Commit the underlying icechunk session."""
        self._owner._commit(message or self.message)

    def rollback(self) -> None:
        """Discard pending changes in the underlying icechunk session."""
        self._owner._discard()


class IcechunkStore(Store):
    """Icechunk repository fronted as a zcollection :class:`Store`.

    The store owns a long-lived writable session whose ``store`` attribute
    is what zarr writes through. Each :meth:`session` block is one
    transaction: success commits and reopens, failure discards and reopens.
    """

    transactional: bool = True

    def __init__(
        self,
        path_or_storage: str | icechunk.Storage,
        *,
        branch: str = "main",
        read_only: bool = False,
    ) -> None:
        """Open or create an icechunk repository.

        Args:
            path_or_storage: Either a local filesystem path or an
                :class:`icechunk.Storage` instance.
            branch: Branch to operate on.
            read_only: When true, opens a read-only session.

        """
        import icechunk

        if isinstance(path_or_storage, str):
            self._uri = path_or_storage
            storage = icechunk.local_filesystem_storage(path_or_storage)
        else:
            self._uri = repr(path_or_storage)
            storage = path_or_storage

        self._repo = icechunk.Repository.open_or_create(storage)
        self._branch = branch
        self._read_only = read_only
        self._session = self._open_session()
        if not read_only:
            self._ensure_root_group()

    # --- session lifecycle ------------------------------------------

    def _open_session(self) -> icechunk.Session:
        if self._read_only:
            return self._repo.readonly_session(branch=self._branch)
        return self._repo.writable_session(self._branch)

    def _commit(self, message: str | None) -> None:
        if not self._session.has_uncommitted_changes:
            return
        self._session.commit(message or "zcollection mutation")
        self._session = self._open_session()

    def _discard(self) -> None:
        if self._session.has_uncommitted_changes:
            self._session.discard_changes()
        self._session = self._open_session()

    @contextmanager
    def session(self) -> Iterator[StoreSession]:
        """Yield an :class:`IcechunkSession`; commits on success, rolls back on error."""
        sess = IcechunkSession(self)
        try:
            yield sess
        except BaseException:
            sess.rollback()
            raise
        else:
            sess.commit()

    # --- bookkeeping -------------------------------------------------

    def _ensure_root_group(self) -> None:
        proto = default_buffer_prototype()
        if not run_sync(self._session.store.exists("zarr.json")):
            run_sync(
                self._session.store.set(
                    "zarr.json",
                    proto.buffer.from_bytes(_GROUP_DOC),
                )
            )
        if not run_sync(self._session.store.exists(f"{META_DIR}/zarr.json")):
            run_sync(
                self._session.store.set(
                    f"{META_DIR}/zarr.json",
                    proto.buffer.from_bytes(_GROUP_DOC),
                )
            )
        # Persist the bootstrap so reopens see a valid hierarchy even if
        # the caller never explicitly commits.
        if self._session.has_uncommitted_changes:
            self._session.commit("bootstrap")
            self._session = self._open_session()

    # --- Store ABC ---------------------------------------------------

    @property
    def root_uri(self) -> str:
        """Return a human-readable URI for this repository."""
        return f"icechunk://{self._uri}"

    def zarr_store(self) -> Any:
        """Return the underlying zarr store backed by the active session."""
        return self._session.store

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists, routing non-zarr keys to the meta group."""
        target = key if _is_zarr_key(key) else _meta_path(key)
        return run_sync(self._session.store.exists(target))

    def read_bytes(self, key: str) -> bytes | None:
        """Read raw bytes from ``key``, decoding the meta-group payload when needed."""
        proto = default_buffer_prototype()
        if _is_zarr_key(key):
            buf = run_sync(self._session.store.get(key, prototype=proto))
            return bytes(buf.to_bytes()) if buf is not None else None
        buf = run_sync(
            self._session.store.get(_meta_path(key), prototype=proto)
        )
        if buf is None:
            return None
        try:
            doc = json.loads(bytes(buf.to_bytes()).decode("utf-8"))
        except UnicodeDecodeError, json.JSONDecodeError:
            return None
        payload = doc.get("attributes", {}).get(_PAYLOAD_ATTR)
        return payload.encode("utf-8") if isinstance(payload, str) else None

    def write_bytes(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key``, wrapping non-zarr blobs into the meta group."""
        self._require_writable()
        proto = default_buffer_prototype()
        if _is_zarr_key(key):
            run_sync(
                self._session.store.set(
                    key,
                    proto.buffer.from_bytes(data),
                )
            )
            return
        run_sync(
            self._session.store.set(
                _meta_path(key),
                proto.buffer.from_bytes(_meta_doc(data)),
            )
        )

    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield direct children of ``prefix``."""
        return self.list_dir(prefix)

    def list_dir(self, prefix: str) -> Iterator[str]:
        """Yield direct children of ``prefix``, excluding the zarr.json marker."""
        names = to_list_async(self._session.store.list_dir(prefix.strip("/")))
        return (n for n in names if n != "zarr.json")

    def delete_prefix(self, prefix: str) -> None:
        """Recursively delete everything under ``prefix`` and its meta sidecars."""
        self._require_writable()
        target = prefix.strip("/")
        if target:
            run_sync(self._session.store.delete_dir(target))
        # Also sweep meta blobs whose original key falls under this prefix.
        meta_children = to_list_async(self._session.store.list_dir(META_DIR))
        pfx_slash = target + "/" if target else ""
        for slug in meta_children:
            if slug == "zarr.json":
                continue
            original = slug.replace("__", "/")
            if target and (
                original == target or original.startswith(pfx_slash)
            ):
                run_sync(self._session.store.delete_dir(f"{META_DIR}/{slug}"))

    # --- helpers -----------------------------------------------------

    def _require_writable(self) -> None:
        if self._read_only:
            raise PermissionError(f"{self.root_uri} is read-only")

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"IcechunkStore({self._uri!r})"
