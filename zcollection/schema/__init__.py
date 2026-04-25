"""Metadata model for zcollection v3.

Frozen dataclasses describing dimensions, attributes, variables, and the full
dataset schema. This module is the single source of truth for what a
collection *is*; nothing outside ``schema/`` reads or writes the
``_zcollection.json`` config.
"""

from .attribute import Attribute
from .builder import SchemaBuilder
from .dataset import DatasetSchema
from .dimension import Dimension
from .variable import VariableRole, VariableSchema
from .versioning import FORMAT_VERSION

__all__ = (
    "FORMAT_VERSION",
    "Attribute",
    "DatasetSchema",
    "Dimension",
    "SchemaBuilder",
    "VariableRole",
    "VariableSchema",
)
