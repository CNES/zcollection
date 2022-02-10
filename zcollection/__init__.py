# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Handle a collection of Zarr groups.
===================================
"""
from . import merging, partitioning
from .collection import (
    Collection,
    Indexer,
    PartitionCallable,
    PartitionFilter,
    create_collection,
    open_collection,
)
from .dataset import Attribute, Dataset, Variable
from .version import __version__
from .view import View, create_view, open_view

__all__ = [
    "__version__",
    "Attribute",
    "Collection",
    "create_collection",
    "create_view",
    "Dataset",
    "Indexer",
    "merging",
    "open_collection",
    "open_view",
    "PartitionCallable",
    "PartitionFilter",
    "partitioning",
    "Variable",
    "version",
    "View",
]
