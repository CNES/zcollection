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
)
from .dataset import Dataset, Expression
from .meta import Attribute
from .variable import Array, DelayedArray, Variable
from .version import __version__
from .view import View, ViewReference, ViewUpdateCallable

__all__ = (
    '__version__',
    'Array',
    'Attribute',
    'Collection',
    'create_collection',
    'create_view',
    'Dataset',
    'DelayedArray',
    'Expression',
    'Indexer',
    'MapCallable',
    'merging',
    'open_collection',
    'open_view',
    'PartitionCallable',
    'PartitionFilter',
    'PartitionFilterCallback',
    'partitioning',
    'UpdateCallable',
    'Variable',
    'version',
    'View',
    'ViewReference',
    'ViewUpdateCallable',
)
