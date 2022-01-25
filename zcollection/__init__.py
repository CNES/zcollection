# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Handle a collection of Zarr groups.
===================================
"""
from . import merging, partitioning, version
from .collection import (Collection, Indexer, PartitionCallable,
                         create_collection, open_collection)
from .dataset import Attribute, Dataset, Variable
from .view import View, create_view, open_view

__version__ = version.release()
__date__ = version.date()
del version

__all__ = [
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
    "partitioning",
    "Variable",
    "version",
    "View",
]
