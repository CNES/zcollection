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
    check,
    check_view,
    create_collection,
    create_view,
    migrate,
    migrate_view,
    open_collection,
    open_view,
    prune,
    prune_view,
    status,
    view_status,
)
from .dataset import Dataset, Expression
from .meta import Attribute, DatasetSchema, Dimension, VariableSchema
from .migration import Outcome, Status
from .variable import Array, DelayedArray, Variable
from .version import __version__
from .view import View, ViewReference, ViewUpdateCallable

__all__ = (
    'Array',
    'Attribute',
    'Collection',
    'Dataset',
    'DatasetSchema',
    'DelayedArray',
    'Dimension',
    'Expression',
    'Indexer',
    'MapCallable',
    'Outcome',
    'PartitionCallable',
    'PartitionFilter',
    'PartitionFilterCallback',
    'Status',
    'UpdateCallable',
    'Variable',
    'VariableSchema',
    'View',
    'ViewReference',
    'ViewUpdateCallable',
    '__version__',
    'check',
    'check_view',
    'create_collection',
    'create_view',
    'merging',
    'migrate',
    'migrate_view',
    'open_collection',
    'open_view',
    'partitioning',
    'prune',
    'prune_view',
    'status',
    'version',
    'view_status',
)
