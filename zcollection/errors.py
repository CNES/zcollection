"""Exception taxonomy for zcollection v3."""
from __future__ import annotations


class ZCollectionError(Exception):
    """Base class for all zcollection errors."""


class SchemaError(ZCollectionError):
    """Raised when a schema is invalid or inconsistent with data."""


class FormatVersionError(ZCollectionError):
    """Raised when an on-disk format version is unsupported."""


class PartitionError(ZCollectionError):
    """Raised when a partition cannot be located, encoded, or decoded."""


class ExpressionError(ZCollectionError):
    """Raised when a partition filter expression is invalid."""


class StoreError(ZCollectionError):
    """Raised by the store layer for I/O or transactional failures."""


class CollectionExistsError(ZCollectionError):
    """Raised when create_collection targets a path that already exists."""


class CollectionNotFoundError(ZCollectionError):
    """Raised when open_collection cannot locate a collection."""


class ReadOnlyError(ZCollectionError):
    """Raised when a write op is attempted on a read-only collection."""


class UpgradeError(ZCollectionError):
    """Raised when a v2 entry point or removed argument is used."""
