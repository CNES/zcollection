"""Path-layout constants for a zcollection v3 collection on disk.

Path conventions:

- root group: ``<root>/zarr.json`` (Zarr v3)
- root config: ``<root>/_zcollection.json``
- partition catalog: ``<root>/_catalog/``
- immutable variables: ``<root>/_immutable/``
- a partition: ``<root>/<partition-path>/`` (e.g. ``year=2024/month=03``)
"""

#: Subdirectory at the root holding the partition catalog.
CATALOG_DIR: str = "_catalog"

#: Subdirectory at the root holding immutable variables.
IMMUTABLE_DIR: str = "_immutable"


def join_path(*parts: str) -> str:
    """POSIX-style join used everywhere in stores (Zarr v3 keys use ``/``)."""
    cleaned = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join(cleaned)


def parent_path(path: str) -> str:
    """Return the parent of ``path``, or the empty string at the root."""
    p = path.rstrip("/")
    if "/" not in p:
        return ""
    return p.rsplit("/", 1)[0]


def relative_path(path: str, base: str) -> str:
    """Return ``path`` expressed relative to ``base``, falling back to ``path``."""
    p = path.strip("/")
    b = base.strip("/")
    if not b:
        return p
    if p == b:
        return ""
    if p.startswith(b + "/"):
        return p[len(b) + 1 :]
    return p
