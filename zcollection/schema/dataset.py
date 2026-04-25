"""Dataset-level schema."""

from typing import Any
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..errors import SchemaError
from .attribute import encode_attrs
from .dimension import Dimension
from .variable import VariableSchema
from .versioning import FORMAT_VERSION, upgrade


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Immutable description of a collection's dataset."""

    #: Mapping of dimension name to dimension metadata.
    dimensions: Mapping[str, Dimension]
    #: Mapping of variable name to variable metadata.
    variables: Mapping[str, VariableSchema]
    #: Optional global attributes associated with the dataset.
    attrs: Mapping[str, Any] = field(default_factory=dict)
    #: Format version of this schema; used for compatibility checks and
    #: upgrades.
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        """Freeze the dimension/variable/attribute maps and validate references."""
        # Freeze maps and validate dimension references.
        dims = dict(self.dimensions)
        vars_ = dict(self.variables)
        for var in vars_.values():
            for d in var.dimensions:
                if d not in dims:
                    raise SchemaError(
                        f"variable {var.name!r} references unknown dimension "
                        f"{d!r}; known: {sorted(dims)}"
                    )
        object.__setattr__(self, "dimensions", MappingProxyType(dims))
        object.__setattr__(self, "variables", MappingProxyType(vars_))
        object.__setattr__(
            self, "attrs", MappingProxyType(encode_attrs(dict(self.attrs)))
        )

    # Convenience views ------------------------------------------------

    @property
    def dim_sizes(self) -> dict[str, int | None]:
        """Return a mapping of dimension name to declared size."""
        return {n: d.size for n, d in self.dimensions.items()}

    @property
    def dim_chunks(self) -> dict[str, int | None]:
        """Return a mapping of dimension name to declared chunk size."""
        return {n: d.chunks for n, d in self.dimensions.items()}

    def variables_by_role(
        self, *, immutable: bool | None = None
    ) -> tuple[VariableSchema, ...]:
        """Return the variables, optionally filtered by their immutable flag.

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

    # Mutators that return a new schema --------------------------------

    def with_partition_axis(self, axis: str) -> DatasetSchema:
        """Mark each variable immutable iff it does not span ``axis``."""
        if axis not in self.dimensions:
            raise SchemaError(
                f"partitioning axis {axis!r} is not a known dimension"
            )
        new_vars = {
            name: var.with_immutable(axis not in var.dimensions)
            for name, var in self.variables.items()
        }
        return DatasetSchema(
            dimensions=dict(self.dimensions),
            variables=new_vars,
            attrs=dict(self.attrs),
            format_version=self.format_version,
        )

    def select(self, names: Iterable[str]) -> DatasetSchema:
        """Return a new schema restricted to the named variables.

        Args:
            names: Names of variables to keep.

        Returns:
            A new schema containing only the requested variables.

        Raises:
            SchemaError: If any of ``names`` is not a known variable.

        """
        wanted = set(names)
        missing = wanted - set(self.variables)
        if missing:
            raise SchemaError(f"unknown variables: {sorted(missing)}")
        return DatasetSchema(
            dimensions=dict(self.dimensions),
            variables={n: self.variables[n] for n in wanted},
            attrs=dict(self.attrs),
            format_version=self.format_version,
        )

    # JSON round-trip --------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialize the schema to a JSON-compatible dict."""
        return {
            "format_version": self.format_version,
            "dimensions": [d.to_json() for d in self.dimensions.values()],
            "variables": [v.to_json() for v in self.variables.values()],
            "attrs": dict(self.attrs),
        }

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
        return cls(
            dimensions=dims,
            variables=vars_,
            attrs=payload.get("attrs", {}),
            format_version=payload.get("format_version", FORMAT_VERSION),
        )
