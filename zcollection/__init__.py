# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
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
from .data import Dataset, Group, Variable
from .errors import (
    CollectionExistsError,
    SchemaError,
    StoreError,
    ZCollectionError,
)
from .schema import (
    DatasetSchema,
    Dimension,
    GroupSchema,
    SchemaBuilder,
    VariableRole,
    VariableSchema,
)
from .store import LocalStore, MemoryStore, Store, open_store


__all__ = (
    "Collection",
    "CollectionExistsError",
    "CollectionNotFoundError",
    "Dataset",
    "DatasetSchema",
    "Dimension",
    "Group",
    "GroupSchema",
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
