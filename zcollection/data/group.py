# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Hierarchical container of variables, attributes, and child groups.

A :class:`Group` mirrors a Zarr v3 group: it owns a set of variables, a set
of attributes, a set of declared dimensions, and any number of child groups.
:class:`~zcollection.Dataset` is a :class:`Group` subclass acting as the
root.
"""

from typing import TYPE_CHECKING, Any
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping


if TYPE_CHECKING:
    from ..schema import GroupSchema
    from ..schema.dimension import Dimension
    from .variable import Variable


def _split_path(path: str) -> tuple[tuple[str, ...], str]:
    """Split a path into (group segments, leaf name).

    ``"/a/b/var"`` -> ``(("a", "b"), "var")``;
    ``"var"`` -> ``((), "var")``;
    ``"/a/b/"`` is treated as ``"/a/b"`` (no leaf — the caller should call
    a group-only resolver).
    """
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ((), "")
    return (tuple(parts[:-1]), parts[-1])


class Group:
    """A named container of variables, attributes, dimensions, and child groups.

    The root group has ``name == "/"`` and ``parent is None``. Child groups
    keep a back-reference to their parent so absolute paths
    (:meth:`long_name`) and dimension inheritance (:meth:`find_dimension`)
    can be computed cheaply.

    Args:
        schema: Schema describing this group's variables, dimensions, and
            attributes.
        variables: Variables for this group, as a mapping or iterable.
        groups: Child groups for this group, as a mapping or iterable.
        attrs: Group-level attributes; defaults to ``schema.attrs``.
        name: Short name of this group; ``"/"`` for the root.
        parent: Parent group reference; ``None`` for the root.

    """

    __slots__ = ("_attrs", "_groups", "_variables", "name", "parent", "schema")

    def __init__(
        self,
        schema: GroupSchema,
        variables: Mapping[str, Variable] | Iterable[Variable] = (),
        groups: Mapping[str, Group] | Iterable[Group] = (),
        attrs: Mapping[str, Any] | None = None,
        *,
        name: str = "/",
        parent: Group | None = None,
    ) -> None:
        """Initialize a group."""
        self.schema = schema
        self.name = name
        self.parent = parent

        if isinstance(variables, Mapping):
            v_items: Iterable[tuple[str, Variable]] = variables.items()
        else:
            v_items = ((v.name, v) for v in variables)
        self._variables: OrderedDict[str, Variable] = OrderedDict(v_items)

        if isinstance(groups, Mapping):
            g_items: Iterable[tuple[str, Group]] = groups.items()
        else:
            g_items = ((g.name, g) for g in groups)
        self._groups: OrderedDict[str, Group] = OrderedDict()
        for n, g in g_items:
            g.parent = self
            self._groups[n] = g

        self._attrs: dict[str, Any] = dict(
            attrs if attrs is not None else schema.attrs
        )
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        sizes: dict[str, int] = {}
        for var in self._variables.values():
            for dim, size in zip(var.dimensions, var.shape, strict=False):
                if size is None:
                    continue
                prev = sizes.setdefault(dim, size)
                if prev != size:
                    raise ValueError(
                        f"inconsistent size for dimension {dim!r}: "
                        f"{prev} vs {size} (variable {var.name!r})"
                    )

    # State accessors --------------------------------------------------

    @property
    def variables(self) -> Mapping[str, Variable]:
        """Return the variables declared on this group (root level only)."""
        return self._variables

    @property
    def groups(self) -> Mapping[str, Group]:
        """Return the child groups of this group."""
        return self._groups

    @property
    def attrs(self) -> dict[str, Any]:
        """Return this group's attributes."""
        return self._attrs

    @property
    def dimensions(self) -> dict[str, int]:
        """Return a mapping of dimension names declared on this group to sizes.

        Sizes are derived from the variables held by this group; ancestor
        dimensions are not included (see :meth:`find_dimension`).
        """
        sizes: dict[str, int] = {}
        for var in self._variables.values():
            for dim, size in zip(var.dimensions, var.shape, strict=False):
                sizes.setdefault(dim, size)
        return sizes

    @property
    def is_lazy(self) -> bool:
        """Return whether any variable in the tree wraps lazy data."""
        if any(v.is_lazy for v in self._variables.values()):
            return True
        return any(child.is_lazy for child in self._groups.values())

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes held by the tree rooted here."""
        own = sum(v.nbytes for v in self._variables.values())
        return own + sum(g.nbytes for g in self._groups.values())

    def is_root(self) -> bool:
        """Return ``True`` if this is the root group."""
        return self.parent is None

    def long_name(self) -> str:
        """Return the absolute path of this group (e.g. ``"/data_01/ku"``)."""
        if self.parent is None:
            return "/"
        parts: list[str] = [self.name]
        node = self.parent
        while node is not None and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return "/" + "/".join(reversed(parts))

    # Mapping-style API (root variables only) --------------------------

    def __getitem__(self, name: str) -> Variable:
        """Return the variable at ``name`` (short name or absolute path)."""
        if "/" in name:
            return self.get_variable(name)
        return self._variables[name]

    def __contains__(self, name: object) -> bool:
        """Return whether the dataset contains a variable named ``name``."""
        return name in self._variables

    def __iter__(self) -> Iterator[str]:
        """Iterate over root-level variable names."""
        return iter(self._variables)

    def __len__(self) -> int:
        """Return the number of root-level variables."""
        return len(self._variables)

    # Path-based access ------------------------------------------------

    def get_group(self, path: str) -> Group:
        """Return the group at ``path`` (absolute or relative).

        Raises:
            KeyError: If the path does not resolve to a known group.

        """
        node = self
        for segment in [p for p in path.split("/") if p]:
            child = node._groups.get(segment)
            if child is None:
                raise KeyError(f"unknown group {path!r}")
            node = child
        return node

    def get_variable(self, path: str) -> Variable:
        """Return the variable at ``path`` (absolute or relative).

        Raises:
            KeyError: If the path does not resolve to a known variable.

        """
        groups, leaf = _split_path(path)
        if not leaf:
            raise KeyError(f"path {path!r} has no leaf variable name")
        node = self
        for segment in groups:
            child = node._groups.get(segment)
            if child is None:
                raise KeyError(f"unknown group {path!r}")
            node = child
        if leaf not in node._variables:
            raise KeyError(f"unknown variable {path!r}")
        return node._variables[leaf]

    def find_dimension(self, name: str) -> Dimension | None:
        """Search for a declared dimension by name, walking up the tree.

        Returns the first matching :class:`Dimension` from this group's
        schema or any ancestor's schema, or ``None`` if not found.
        """
        node: Group | None = self
        while node is not None:
            dim = node.schema.dimensions.get(name) if node.schema else None
            if dim is not None:
                return dim
            node = node.parent
        return None

    def find_variable(self, name: str) -> Variable | None:
        """Search for a variable by short name in this group and its descendants."""
        if name in self._variables:
            return self._variables[name]
        for child in self.iter_groups():
            if name in child._variables:
                return child._variables[name]
        return None

    def find_group(self, name: str) -> Group | None:
        """Search for a child group by short name (recursive)."""
        if name in self._groups:
            return self._groups[name]
        for child in self.iter_groups():
            if name in child._groups:
                return child._groups[name]
        return None

    def iter_groups(self) -> Iterator[Group]:
        """Yield all descendant groups depth-first (excluding self)."""
        for child in self._groups.values():
            yield child
            yield from child.iter_groups()

    # Mutation ---------------------------------------------------------

    def add_group(self, group: Group) -> Group:
        """Attach ``group`` as a child of this group.

        Raises:
            ValueError: If a child with the same name already exists.

        """
        if group.name in self._groups:
            raise ValueError(f"group {group.name!r} already exists")
        group.parent = self
        self._groups[group.name] = group
        return group

    def add_variable(self, variable: Variable) -> None:
        """Register ``variable`` on this group.

        Raises:
            ValueError: If a variable with the same name already exists.

        """
        if variable.name in self._variables:
            raise ValueError(f"variable {variable.name!r} already exists")
        self._variables[variable.name] = variable
        self._validate_dimensions()

    def update_attributes(self, **attributes: Any) -> None:
        """Update this group's attributes in place."""
        self._attrs.update(attributes)

    # Tree helpers -----------------------------------------------------

    def walk(self) -> Iterator[tuple[str, Group]]:
        """Yield ``(absolute_path, group)`` for self and every descendant."""
        yield (self.long_name(), self)
        for child in self.iter_groups():
            yield (child.long_name(), child)

    def all_variables(self) -> dict[str, Variable]:
        """Return every variable in the tree keyed by absolute path.

        Variables at the root keep their short name; nested variables are
        keyed by ``"<group>/<name>"`` (no leading slash) for direct passing
        back to :class:`~zcollection.Dataset`.
        """
        out: dict[str, Variable] = {}

        def walk(group: Group, prefix: str) -> None:
            for name, var in group.variables.items():
                key = f"{prefix}{name}" if prefix else name
                out[key] = var
            for child_name, child in group.groups.items():
                walk(
                    child,
                    f"{prefix}{child_name}/" if prefix else f"{child_name}/",
                )

        walk(self, "")
        return out

    # Repr -------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a multi-line, xarray-like representation."""
        from ._repr import group_repr

        return group_repr(self)
