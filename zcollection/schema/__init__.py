# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Metadata model for zcollection v3.

Frozen dataclasses describing dimensions, attributes, variables, and the full
dataset schema. This module is the single source of truth for what a
collection *is*; nothing outside ``schema/`` reads or writes the
``_zcollection.json`` config.
"""

from .builder import SchemaBuilder
from .dataset import DatasetSchema
from .dimension import Dimension
from .group import GroupSchema
from .variable import VariableRole, VariableSchema
from .versioning import FORMAT_VERSION


__all__ = (
    "FORMAT_VERSION",
    "DatasetSchema",
    "Dimension",
    "GroupSchema",
    "SchemaBuilder",
    "VariableRole",
    "VariableSchema",
)
