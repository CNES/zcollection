# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Handle a collection of Zarr groups.
===================================
"""
from . import merging, partitioning
from .collection import Collection
from .collection.abc import Indexer, PartitionFilter, PartitionFilterCallback
from .collection.callable_objects import (
    MapCallable,
    PartitionCallable,
    UpdateCallable,
)
from .convenience import (
    create_collection,
    create_view,
    open_collection,
    open_view,
    update_deprecated_collection,
)
from .dataset import Dataset, Expression
from .meta import Attribute
from .variable import Array, DelayedArray, Variable, new_variable
from .version import __version__
from .view import View, ViewReference, ViewUpdateCallable

__all__ = (
    'Array',
    'Attribute',
    'Collection',
    'Dataset',
    'DelayedArray',
    'Expression',
    'Indexer',
    'MapCallable',
    'PartitionCallable',
    'PartitionFilter',
    'PartitionFilterCallback',
    'UpdateCallable',
    'Variable',
    'View',
    'ViewReference',
    'ViewUpdateCallable',
    '__version__',
    'create_collection',
    'create_view',
    'merging',
    'new_variable',
    'open_collection',
    'open_view',
    'partitioning',
    'update_deprecated_collection',
    'version',
)
