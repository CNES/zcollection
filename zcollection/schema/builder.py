"""Fluent builder for :class:`DatasetSchema`.

Usage::

    schema = (SchemaBuilder()
        .with_dimension("time", chunks=4096)
        .with_dimension("x_ac", size=240, chunks=240)
        .with_variable("ssh", dtype="float32", dimensions=("time", "x_ac"))
        .build())
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy

from ..codecs import CodecStack
from .dataset import DatasetSchema
from .dimension import Dimension
from .variable import VariableRole, VariableSchema


class SchemaBuilder:
    """Mutable builder that produces an immutable :class:`DatasetSchema`."""

    def __init__(self) -> None:
        self._dims: dict[str, Dimension] = {}
        self._vars: dict[str, VariableSchema] = {}
        self._attrs: dict[str, Any] = {}

    def with_dimension(
        self,
        name: str,
        *,
        size: int | None = None,
        chunks: int | None = None,
        shards: int | None = None,
    ) -> "SchemaBuilder":
        self._dims[name] = Dimension(name, size=size, chunks=chunks, shards=shards)
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
    ) -> "SchemaBuilder":
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

    def with_attribute(self, name: str, value: Any) -> "SchemaBuilder":
        self._attrs[name] = value
        return self

    def build(self) -> DatasetSchema:
        return DatasetSchema(
            dimensions=dict(self._dims),
            variables=dict(self._vars),
            attrs=dict(self._attrs),
        )
