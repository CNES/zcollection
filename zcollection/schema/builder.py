# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Fluent builder for :class:`DatasetSchema`.

Usage::

    schema = (
        SchemaBuilder()
        .with_dimension("time", chunks=4096)
        .with_dimension("x_ac", size=240, chunks=240)
        .with_variable("ssh", dtype="float32", dimensions=("time", "x_ac"))
        .build()
    )

Hierarchical groups can be declared by passing ``group=`` to
:meth:`with_variable` or :meth:`with_dimension`, or by calling
:meth:`with_group` to attach group-level attributes::

    schema = (
        SchemaBuilder()
        .with_dimension("time", chunks=4096)
        .with_group("/data_01/ku", attrs={"band": "Ku"})
        .with_dimension("range", size=240, group="/data_01/ku")
        .with_variable(
            "power",
            dtype="float32",
            dimensions=("time", "range"),
            group="/data_01/ku",
        )
        .build()
    )
"""

from typing import Any
from collections.abc import Iterable

import numpy

from ..codecs import CodecStack
from .dataset import DatasetSchema
from .dimension import Dimension
from .group import GroupSchema, _split_path
from .variable import VariableRole, VariableSchema


class _GroupBuilder:
    """Mutable scratch state for one group during building."""

    __slots__ = ("attrs", "dims", "groups", "name", "vars")

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.dims: dict[str, Dimension] = {}
        self.vars: dict[str, VariableSchema] = {}
        self.groups: dict[str, _GroupBuilder] = {}
        self.attrs: dict[str, Any] = {}

    def freeze(self) -> GroupSchema:
        return GroupSchema(
            name=self.name,
            dimensions=dict(self.dims),
            variables=dict(self.vars),
            groups={n: g.freeze() for n, g in self.groups.items()},
            attrs=dict(self.attrs),
        )


class SchemaBuilder:
    """Mutable builder that produces an immutable :class:`DatasetSchema`."""

    def __init__(self) -> None:
        """Initialize an empty builder."""
        #: Root group scratch state.
        self._root: _GroupBuilder = _GroupBuilder("/")

    # Group resolution -------------------------------------------------

    def _resolve(self, path: str | None, *, create: bool) -> _GroupBuilder:
        """Return the group builder at ``path``, creating it if requested."""
        if path is None or path in ("", "/"):
            return self._root
        node = self._root
        for segment in _split_path(path):
            child = node.groups.get(segment)
            if child is None:
                if not create:
                    raise KeyError(f"unknown group {path!r}")
                child = _GroupBuilder(segment)
                node.groups[segment] = child
            node = child
        return node

    # Public API -------------------------------------------------------

    def with_dimension(
        self,
        name: str,
        *,
        size: int | None = None,
        chunks: int | None = None,
        shards: int | None = None,
        group: str | None = None,
    ) -> SchemaBuilder:
        """Register a dimension on the root or a nested group.

        Args:
            name: Dimension name.
            size: Fixed size, or ``None`` if unknown (e.g. partitioning axis).
            chunks: Chunk size along this dimension; ``None`` to use the full
                extent.
            shards: Shard size along this dimension; ``None`` for no sharding.
            group: Optional path of the group this dimension belongs to.
                Defaults to the root group.

        Returns:
            This builder, to allow chaining.

        """
        target = self._resolve(group, create=True)
        target.dims[name] = Dimension(
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
        group: str | None = None,
    ) -> SchemaBuilder:
        """Register a variable on the root or a nested group.

        Args:
            name: Variable name.
            dtype: NumPy dtype or anything :func:`numpy.dtype` accepts.
            dimensions: Names of the dimensions this variable spans. Each
                dimension must be declared on the same group or an ancestor.
            fill_value: Optional fill value.
            codecs: Optional explicit codec stack; auto-detected when ``None``.
            attrs: Optional attribute mapping.
            role: Provenance role of the variable.
            group: Optional path of the group this variable belongs to.
                Defaults to the root group.

        Returns:
            This builder, to allow chaining.

        """
        target = self._resolve(group, create=True)
        target.vars[name] = VariableSchema(
            name=name,
            dtype=numpy.dtype(dtype),
            dimensions=tuple(dimensions),
            fill_value=fill_value,
            codecs=codecs or CodecStack(),
            attrs=dict(attrs or {}),
            role=role,
        )
        return self

    def with_attribute(
        self,
        name: str,
        value: Any,
        *,
        group: str | None = None,
    ) -> SchemaBuilder:
        """Register an attribute on the root or a nested group.

        Args:
            name: Attribute name.
            value: Attribute value.
            group: Optional path of the group this attribute belongs to.
                Defaults to the root group.

        Returns:
            This builder, to allow chaining.

        """
        target = self._resolve(group, create=True)
        target.attrs[name] = value
        return self

    def with_group(
        self,
        path: str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> SchemaBuilder:
        """Declare a nested group at ``path``, optionally with attributes.

        Intermediate groups along the path are created if missing. Calling
        this is only required to attach attributes to a group ahead of time;
        otherwise :meth:`with_variable` and :meth:`with_dimension` will
        create groups lazily.

        Args:
            path: Absolute or relative group path (e.g. ``"/data_01/ku"``).
            attrs: Optional group-level attributes.

        Returns:
            This builder, to allow chaining.

        """
        target = self._resolve(path, create=True)
        if attrs:
            target.attrs.update(attrs)
        return self

    def build(self) -> DatasetSchema:
        """Build and return the immutable :class:`DatasetSchema`."""
        root = self._root.freeze()
        return DatasetSchema(
            dimensions=dict(root.dimensions),
            variables=dict(root.variables),
            groups=dict(root.groups),
            attrs=dict(root.attrs),
        )
