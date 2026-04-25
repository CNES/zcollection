"""Partition catalog — fast O(1) listing without S3 LIST.

The catalog is a small JSON document at ``<root>/_catalog/state.json``
holding the *complete* set of partition relative paths in sorted order,
plus a content checksum used to detect tampering or torn writes.

Why not a sharded VLenUTF8 array (as the original plan called out)? At
realistic scale (tens of thousands of partitions, each path under 80 bytes)
the catalog comfortably fits in a single object, so a JSON blob written
through :meth:`Store.write_bytes` is both simpler and atomic on every
backend we care about (local FS via ``os.replace``; S3 PUT is an atomic
overwrite). We can promote to a sharded array later without changing the
caller-visible API.

Two-phase / crash safety:

- ``write`` is a single :meth:`Store.write_bytes` call, so the catalog is
  never observed half-written on any supported backend.
- The danger is the *outer* operation crashing between writing partition
  data and writing the new catalog. The catalog then disagrees with the
  on-disk truth. :meth:`Catalog.read` exposes a ``checksum`` that callers
  may verify against a recompute from the listed paths; mismatch ⇒ fall
  back to walking. :func:`reconcile` performs the recovery write-back.
"""

from typing import TYPE_CHECKING
from dataclasses import dataclass
import hashlib
import json

from ..store.layout import CATALOG_DIR, join_path

if TYPE_CHECKING:
    from ..store import Store

#: Relative key under the collection root.
CATALOG_FILE: str = join_path(CATALOG_DIR, "state.json")

CATALOG_FORMAT_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class CatalogState:
    """Decoded catalog payload."""

    paths: tuple[str, ...]
    checksum: str

    def matches(self, paths: list[str] | tuple[str, ...]) -> bool:
        """Return whether ``paths`` hash to the stored checksum."""
        return _checksum(paths) == self.checksum


class Catalog:
    """Read/write the partition catalog for one collection."""

    def __init__(self, store: Store) -> None:
        """Initialize the catalog reader/writer for ``store``."""
        self._store = store

    # --- read --------------------------------------------------------

    def exists(self) -> bool:
        """Return whether the catalog file exists in the store."""
        return self._store.exists(CATALOG_FILE)

    def read(self) -> CatalogState | None:
        """Read and decode the catalog, or ``None`` if missing/corrupt."""
        raw = self._store.read_bytes(CATALOG_FILE)
        if raw is None:
            return None
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Treat a corrupted catalog as missing; callers will rebuild.
            return None
        paths = tuple(doc.get("paths", ()))
        checksum = doc.get("checksum") or _checksum(paths)
        return CatalogState(paths=paths, checksum=checksum)

    def read_paths(self) -> list[str] | None:
        """Return the catalog's path list, or ``None`` if missing."""
        state = self.read()
        return list(state.paths) if state is not None else None

    # --- write -------------------------------------------------------

    def write(self, paths: list[str] | tuple[str, ...]) -> None:
        """Write ``paths`` to the catalog (sorted and deduplicated)."""
        sorted_paths = sorted(set(paths))
        payload = {
            "format_version": CATALOG_FORMAT_VERSION,
            "paths": sorted_paths,
            "checksum": _checksum(sorted_paths),
        }
        self._store.write_bytes(
            CATALOG_FILE,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )

    def add(self, paths: list[str] | tuple[str, ...]) -> None:
        """Merge ``paths`` into the catalog."""
        current = self.read_paths() or []
        merged = sorted(set(current) | set(paths))
        self.write(merged)

    def remove(self, paths: list[str] | tuple[str, ...]) -> None:
        """Remove ``paths`` from the catalog."""
        current = self.read_paths() or []
        gone = set(paths)
        self.write([p for p in current if p not in gone])

    def drop(self) -> None:
        """Delete the catalog directory."""
        self._store.delete_prefix(CATALOG_DIR)


def reconcile(
    catalog: Catalog, walked: list[str] | tuple[str, ...]
) -> CatalogState:
    """Rewrite the catalog from a fresh walk and return the new state."""
    catalog.write(walked)
    state = catalog.read()
    assert state is not None  # we just wrote it
    return state


def _checksum(paths: list[str] | tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for p in sorted(set(paths)):
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"
