# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Handle a collection of Zarr groups.
===================================
"""
from . import merging, migration, partitioning
from .collection import Collection
from .collection.abc import Indexer, PartitionFilter, PartitionFilterCallback
from .collection.callable_objects import (
    MapCallable,
    PartitionCallable,
    PartitionInfo,
    UpdateCallable,
)
from .convenience import (
    create_collection,
    create_view,
    open_collection,
    open_view,
)
from .dataset import Dataset, Expression
from .execution import PartitionErrors
from .meta import Attribute, DatasetSchema, Dimension, VariableSchema
from .storage import AbsentPartition
from .variable import Array, DelayedArray, Variable
from .variable.delayed_array import ModifiedVariableError
from .version import __version__
from .view import View, ViewReference, ViewUpdateCallable

__all__ = (
    'AbsentPartition',
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
    'ModifiedVariableError',
    'PartitionCallable',
    'PartitionErrors',
    'PartitionFilter',
    'PartitionFilterCallback',
    'PartitionInfo',
    'UpdateCallable',
    'Variable',
    'VariableSchema',
    'View',
    'ViewReference',
    'ViewUpdateCallable',
    '__version__',
    'create_collection',
    'create_view',
    'merging',
    'migration',
    'open_collection',
    'open_view',
    'partitioning',
    'version',
)
