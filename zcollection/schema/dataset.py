"""Dataset-level schema (root :class:`GroupSchema`)."""

from typing import Any
from collections.abc import Iterable
from dataclasses import dataclass

from ..errors import SchemaError
from .dimension import Dimension
from .group import GroupSchema, validate_dim_refs
from .variable import VariableSchema
from .versioning import FORMAT_VERSION, upgrade


@dataclass(frozen=True, slots=True)
class DatasetSchema(GroupSchema):
    """Immutable description of a collection's dataset.

    Extends :class:`GroupSchema` with a :attr:`format_version` field and the
    JSON envelope used by the on-disk ``_zcollection.json`` config. The root
    name is always ``"/"``.
    """

    #: Format version of this schema; used for compatibility checks and
    #: upgrades.
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        """Freeze maps and validate dimension references over the full tree."""
        # Force the root name to "/".
        object.__setattr__(self, "name", "/")
        super().__post_init__()
        validate_dim_refs(self)

    # Convenience views ------------------------------------------------

    @property
    def dim_sizes(self) -> dict[str, int | None]:
        """Return a mapping of dimension name to declared size (root only)."""
        return {n: d.size for n, d in self.dimensions.items()}

    @property
    def dim_chunks(self) -> dict[str, int | None]:
        """Return a mapping of dimension name to declared chunk size.

        Walks the entire tree so chunk hints declared on ancestor groups are
        visible to descendants.
        """
        out: dict[str, int | None] = {
            n: d.chunks for n, d in self.dimensions.items()
        }
        for grp in self.iter_groups():
            for n, d in grp.dimensions.items():
                out.setdefault(n, d.chunks)
        return out

    def variables_by_role(
        self, *, immutable: bool | None = None
    ) -> tuple[VariableSchema, ...]:
        """Return root-group variables, optionally filtered by ``immutable``.

        Args:
            immutable: If given, keep only variables whose ``immutable``
                attribute matches this value.

        Returns:
            A tuple of the matching variables in declaration order.

        """
        return tuple(
            v
            for v in self.variables.values()
            if immutable is None or v.immutable == immutable
        )

    def all_variables_by_role(
        self, *, immutable: bool | None = None
    ) -> dict[str, VariableSchema]:
        """Return all variables across the tree (keyed by absolute path)."""
        return {
            path: v
            for path, v in self.all_variables().items()
            if immutable is None or v.immutable == immutable
        }

    # Mutators that return a new schema --------------------------------

    def with_partition_axis(self, axis: str) -> DatasetSchema:
        """Mark each variable immutable iff it does not span ``axis``.

        Recurses into nested groups. Variables in nested groups that don't
        span ``axis`` are also marked immutable.
        """
        if axis not in self.dimensions and not any(
            axis in g.dimensions for g in self.iter_groups()
        ):
            raise SchemaError(
                f"partitioning axis {axis!r} is not a known dimension"
            )

        def _retag(group: GroupSchema) -> GroupSchema:
            new_vars = {
                name: var.with_immutable(axis not in var.dimensions)
                for name, var in group.variables.items()
            }
            new_groups = {n: _retag(g) for n, g in group.groups.items()}
            return group._replace(variables=new_vars, groups=new_groups)

        retagged = _retag(self)
        return DatasetSchema(
            dimensions=dict(retagged.dimensions),
            variables=dict(retagged.variables),
            groups=dict(retagged.groups),
            attrs=dict(retagged.attrs),
            format_version=self.format_version,
        )

    def select(self, names: Iterable[str]) -> DatasetSchema:
        """Return a new schema restricted to the named variables.

        ``names`` may be short names (resolved against the root group) or
        absolute paths (``/grp/sub/var``). Empty groups in the tree are
        pruned from the result.

        Raises:
            SchemaError: If any of ``names`` is not a known variable.

        """
        wanted_paths = set()
        all_vars = self.all_variables()
        # Resolve each requested name to its absolute path.
        for n in names:
            if n.startswith("/"):
                key = n.lstrip("/")
            else:
                key = n
            if key in all_vars:
                wanted_paths.add(key)
            elif n in self.variables:  # short name at root
                wanted_paths.add(n)
            else:
                raise SchemaError(f"unknown variable {n!r}")

        def _filter(group: GroupSchema, prefix: str) -> GroupSchema | None:
            kept_vars = {
                vn: v
                for vn, v in group.variables.items()
                if (f"{prefix}{vn}" if prefix else vn) in wanted_paths
            }
            kept_groups: dict[str, GroupSchema] = {}
            for gn, g in group.groups.items():
                sub = _filter(g, f"{prefix}{gn}/")
                if sub is not None:
                    kept_groups[gn] = sub
            if (
                not kept_vars
                and not kept_groups
                and prefix  # never prune the root
            ):
                return None
            return group._replace(variables=kept_vars, groups=kept_groups)

        filtered = _filter(self, "") or self._replace(variables={}, groups={})
        return DatasetSchema(
            dimensions=dict(self.dimensions),
            variables=dict(filtered.variables),
            groups=dict(filtered.groups),
            attrs=dict(self.attrs),
            format_version=self.format_version,
        )

    # JSON round-trip --------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialize the schema (root + nested groups) to a JSON dict."""
        doc = super().to_json()
        doc["format_version"] = self.format_version
        return doc

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DatasetSchema:
        """Build a schema from a JSON-compatible dict, upgrading older formats."""
        payload = upgrade(payload)
        dims = {
            d["name"]: Dimension.from_json(d)
            for d in payload.get("dimensions", [])
        }
        vars_ = {
            v["name"]: VariableSchema.from_json(v)
            for v in payload.get("variables", [])
        }
        groups = {
            g["name"]: GroupSchema.from_json(g)
            for g in payload.get("groups", [])
        }
        return cls(
            dimensions=dims,
            variables=vars_,
            groups=groups,
            attrs=payload.get("attrs", {}),
            format_version=payload.get("format_version", FORMAT_VERSION),
        )
