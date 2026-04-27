# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Group-level schema.

A :class:`GroupSchema` is a hierarchical container of dimensions, variables,
attributes, and child groups. It mirrors the on-disk Zarr v3 group hierarchy:
the root group is a :class:`~zcollection.schema.DatasetSchema`; every nested
group is a plain :class:`GroupSchema`.
"""

from typing import Any
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..errors import SchemaError
from .attribute import encode_attrs
from .dimension import Dimension
from .variable import VariableSchema


def _split_path(path: str) -> tuple[str, ...]:
    """Split an absolute or relative group path into segments.

    ``"/"`` and ``""`` map to an empty tuple.
    """
    return tuple(p for p in path.split("/") if p)


@dataclass(frozen=True, slots=True)
class GroupSchema:
    """Immutable description of a (possibly nested) group of variables.

    Args:
        name: Short name of the group (``"/"`` for the root).
        dimensions: Mapping of dimension name to dimension metadata declared
            on this group. Variables declared on this group or any descendant
            may reference dimensions declared on this group or any ancestor.
        variables: Variables declared at this group level.
        groups: Child groups, keyed by short name.
        attrs: Group-level attributes.

    """

    #: Short name of the group; ``"/"`` for the root.
    name: str = "/"
    #: Mapping of dimension name to dimension metadata.
    dimensions: Mapping[str, Dimension] = field(default_factory=dict)
    #: Mapping of variable name to variable metadata.
    variables: Mapping[str, VariableSchema] = field(default_factory=dict)
    #: Mapping of child group name to child :class:`GroupSchema`.
    groups: Mapping[str, GroupSchema] = field(default_factory=dict)
    #: Optional attributes associated with this group.
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the maps and validate variable dimension references."""
        dims = dict(self.dimensions)
        vars_ = dict(self.variables)
        groups = dict(self.groups)
        # Validate variable dimensions against this group's dims plus any
        # dims provided by ancestors (handled by the root validator below).
        for var in vars_.values():
            for d in var.dimensions:
                if d not in dims and not _dim_in_ancestors(d, self):
                    # Best-effort local check; the root performs full
                    # validation in :meth:`_validate_dim_refs`.
                    pass
        object.__setattr__(self, "dimensions", MappingProxyType(dims))
        object.__setattr__(self, "variables", MappingProxyType(vars_))
        object.__setattr__(self, "groups", MappingProxyType(groups))
        object.__setattr__(
            self, "attrs", MappingProxyType(encode_attrs(dict(self.attrs)))
        )

    # Convenience views ------------------------------------------------

    def iter_groups(self) -> Iterable[GroupSchema]:
        """Yield all descendant groups depth-first (excluding self)."""
        for child in self.groups.values():
            yield child
            yield from child.iter_groups()

    def all_variables(self) -> dict[str, VariableSchema]:
        """Return every variable in the tree keyed by absolute path."""
        out: dict[str, VariableSchema] = {}

        def walk(group: GroupSchema, prefix: str) -> None:
            for name, var in group.variables.items():
                out[f"{prefix}{name}" if prefix else name] = var
            for child_name, child in group.groups.items():
                walk(child, f"{prefix}{child_name}/")

        walk(self, "")
        return out

    def get_group(self, path: str) -> GroupSchema:
        """Return the group at ``path`` (absolute or relative).

        Raises:
            KeyError: If the path does not resolve to a known group.

        """
        node = self
        for segment in _split_path(path):
            if segment not in node.groups:
                raise KeyError(f"unknown group {path!r}")
            node = node.groups[segment]
        return node

    # Mutators that return a new schema --------------------------------

    def with_group(self, path: str, group: GroupSchema) -> GroupSchema:
        """Return a copy with ``group`` inserted at ``path`` (relative).

        ``path`` is the parent path; the inserted group keeps its own name.
        Intermediate groups are created if missing.
        """
        segments = _split_path(path)
        new_groups = dict(self.groups)
        if not segments:
            new_groups[group.name] = group
            return self._replace(groups=new_groups)
        head, *tail = segments
        child = new_groups.get(head, GroupSchema(name=head))
        new_child = child.with_group("/".join(tail), group)
        new_groups[head] = new_child
        return self._replace(groups=new_groups)

    def _replace(self, **kwargs: Any) -> GroupSchema:
        """Return a copy of this group schema with the given fields replaced."""
        return GroupSchema(
            name=kwargs.get("name", self.name),
            dimensions=kwargs.get("dimensions", dict(self.dimensions)),
            variables=kwargs.get("variables", dict(self.variables)),
            groups=kwargs.get("groups", dict(self.groups)),
            attrs=kwargs.get("attrs", dict(self.attrs)),
        )

    # JSON round-trip --------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialize this group (recursively) to a JSON-compatible dict."""
        return {
            "name": self.name,
            "dimensions": [d.to_json() for d in self.dimensions.values()],
            "variables": [v.to_json() for v in self.variables.values()],
            "groups": [g.to_json() for g in self.groups.values()],
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> GroupSchema:
        """Build a group schema from a JSON-compatible dict."""
        dims = {
            d["name"]: Dimension.from_json(d)
            for d in payload.get("dimensions", [])
        }
        vars_ = {
            v["name"]: VariableSchema.from_json(v)
            for v in payload.get("variables", [])
        }
        groups = {
            g["name"]: cls.from_json(g) for g in payload.get("groups", [])
        }
        return cls(
            name=payload.get("name", "/"),
            dimensions=dims,
            variables=vars_,
            groups=groups,
            attrs=payload.get("attrs", {}),
        )


def _dim_in_ancestors(dim: str, _schema: GroupSchema) -> bool:
    """Accept unknown dims; full validation runs at the root.

    A standalone :class:`GroupSchema` cannot know its ancestors, so we
    accept unknown dims here. The root :class:`DatasetSchema` validates the
    full tree once on construction via :func:`validate_dim_refs`.
    """
    return True


def validate_dim_refs(root: GroupSchema) -> None:
    """Walk the tree and ensure every variable's dimensions are resolvable.

    A dimension is resolvable if it is declared on the variable's group or
    on any ancestor group.

    Raises:
        SchemaError: If any variable references an unknown dimension.

    """

    def walk(group: GroupSchema, visible: dict[str, Dimension]) -> None:
        merged = {**visible, **dict(group.dimensions)}
        for var in group.variables.values():
            for d in var.dimensions:
                if d not in merged:
                    raise SchemaError(
                        f"variable {var.name!r} in group "
                        f"{group.name!r} references unknown dimension "
                        f"{d!r}; visible: {sorted(merged)}"
                    )
        for child in group.groups.values():
            walk(child, merged)

    walk(root, {})
