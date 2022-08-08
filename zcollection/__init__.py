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
    UpdateCallable,
)
from .convenience import (
    create_collection,
    create_view,
    open_collection,
    open_view,
)
from .dataset import Attribute, Dataset, Variable
from .version import __version__
from .view import View, ViewReference

__all__ = [
    '__version__',
    'Attribute',
    'Collection',
    'create_collection',
    'create_view',
    'Dataset',
    'Indexer',
    'merging',
    'open_collection',
    'open_view',
    'PartitionCallable',
    'PartitionFilter',
    'partitioning',
    'UpdateCallable',
    'Variable',
    'version',
    'View',
    'ViewReference',
]
