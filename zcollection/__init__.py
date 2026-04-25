"""zcollection — Zarr v3 native, async-friendly partitioned collections."""

from . import aio, codecs, partitioning, view
from .api import (
    Collection,
    CollectionNotFoundError,
    ReadOnlyError,
    create_collection,
    open_collection,
)
from .collection import merge
from .data import Dataset, Variable
from .errors import (
    CollectionExistsError,
    SchemaError,
    StoreError,
    ZCollectionError,
)
from .schema import (
    Attribute,
    DatasetSchema,
    Dimension,
    SchemaBuilder,
    VariableRole,
    VariableSchema,
)
from .store import LocalStore, MemoryStore, Store, open_store

__all__ = (
    "Attribute",
    "Collection",
    "CollectionExistsError",
    "CollectionNotFoundError",
    "Dataset",
    "DatasetSchema",
    "Dimension",
    "LocalStore",
    "MemoryStore",
    "ReadOnlyError",
    "SchemaBuilder",
    "SchemaError",
    "Store",
    "StoreError",
    "Variable",
    "VariableRole",
    "VariableSchema",
    "ZCollectionError",
    "aio",
    "codecs",
    "create_collection",
    "merge",
    "open_collection",
    "open_store",
    "partitioning",
    "view",
)


def Schema() -> SchemaBuilder:
    """Shorthand for :class:`SchemaBuilder`."""
    return SchemaBuilder()
