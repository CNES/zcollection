"""Fluent builder for :class:`DatasetSchema`.

Usage::

    schema = (
        SchemaBuilder()
        .with_dimension("time", chunks=4096)
        .with_dimension("x_ac", size=240, chunks=240)
        .with_variable("ssh", dtype="float32", dimensions=("time", "x_ac"))
        .build()
    )
"""

from typing import Any
from collections.abc import Iterable

import numpy

from ..codecs import CodecStack
from .dataset import DatasetSchema
from .dimension import Dimension
from .variable import VariableRole, VariableSchema


class SchemaBuilder:
    """Mutable builder that produces an immutable :class:`DatasetSchema`."""

    def __init__(self) -> None:
        """Initialize an empty builder."""
        #: Registered dimensions, keyed by name.
        self._dims: dict[str, Dimension] = {}
        #: Registered variables, keyed by name.
        self._vars: dict[str, VariableSchema] = {}
        #: Registered dataset-level attributes, keyed by name.
        self._attrs: dict[str, Any] = {}

    def with_dimension(
        self,
        name: str,
        *,
        size: int | None = None,
        chunks: int | None = None,
        shards: int | None = None,
    ) -> SchemaBuilder:
        """Register a dimension.

        Args:
            name: Dimension name.
            size: Fixed size, or ``None`` if unknown (e.g. partitioning axis).
            chunks: Chunk size along this dimension; ``None`` to use the full extent.
            shards: Shard size along this dimension; ``None`` for no sharding.

        Returns:
            This builder, to allow chaining.

        """
        self._dims[name] = Dimension(
            name, size=size, chunks=chunks, shards=shards
        )
        return self

    def with_variable(
        self,
        name: str,
        *,
        dtype: Any,
        dimensions: Iterable[str],
        fill_value: Any | None = None,
        codecs: CodecStack | None = None,
        attrs: dict[str, Any] | None = None,
        role: VariableRole = VariableRole.USER,
    ) -> SchemaBuilder:
        """Register a variable.

        Args:
            name: Variable name.
            dtype: NumPy dtype or anything :func:`numpy.dtype` accepts.
            dimensions: Names of the dimensions this variable spans.
            fill_value: Optional fill value.
            codecs: Optional explicit codec stack; auto-detected when ``None``.
            attrs: Optional attribute mapping.
            role: Provenance role of the variable.

        Returns:
            This builder, to allow chaining.

        """
        self._vars[name] = VariableSchema(
            name=name,
            dtype=numpy.dtype(dtype),
            dimensions=tuple(dimensions),
            fill_value=fill_value,
            codecs=codecs or CodecStack(),
            attrs=dict(attrs or {}),
            role=role,
        )
        return self

    def with_attribute(self, name: str, value: Any) -> SchemaBuilder:
        """Register a dataset-level attribute.

        Args:
            name: Attribute name.
            value: Attribute value.

        Returns:
            This builder, to allow chaining.

        """
        self._attrs[name] = value
        return self

    def build(self) -> DatasetSchema:
        """Build and return the immutable :class:`DatasetSchema`."""
        return DatasetSchema(
            dimensions=dict(self._dims),
            variables=dict(self._vars),
            attrs=dict(self._attrs),
        )
