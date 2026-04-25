"""Variable metadata: dtype, dims, fill, codec stack, role."""

from typing import Any
from dataclasses import dataclass, field
from enum import StrEnum

import numpy

from ..codecs import CodecStack, auto_codecs
from ..errors import SchemaError
from .attribute import encode_attrs


class VariableRole(StrEnum):
    """Provenance / purpose of a variable in the schema."""

    USER = "user"  # added by the caller
    FILLER = "filler"  # added by the collection to satisfy schema
    AUXILIARY = "aux"  # internal (catalog, indices, ...)


@dataclass(frozen=True, slots=True)
class VariableSchema:
    """All information needed to create a Zarr v3 array for one variable."""

    name: str
    dtype: numpy.dtype
    dimensions: tuple[str, ...]
    fill_value: Any | None = None
    codecs: CodecStack = field(default_factory=CodecStack)
    attrs: dict[str, Any] = field(default_factory=dict)
    role: VariableRole = VariableRole.USER
    immutable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, numpy.dtype):
            object.__setattr__(self, "dtype", numpy.dtype(self.dtype))
        if not isinstance(self.dimensions, tuple):
            object.__setattr__(self, "dimensions", tuple(self.dimensions))
        if not self.codecs.array_to_bytes:
            object.__setattr__(self, "codecs", auto_codecs(self.dtype))
        object.__setattr__(self, "attrs", encode_attrs(self.attrs))

    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    def with_immutable(self, immutable: bool) -> VariableSchema:
        return VariableSchema(
            name=self.name,
            dtype=self.dtype,
            dimensions=self.dimensions,
            fill_value=self.fill_value,
            codecs=self.codecs,
            attrs=dict(self.attrs),
            role=self.role,
            immutable=immutable,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.str,
            "dimensions": list(self.dimensions),
            "fill_value": _encode_fill(self.fill_value, self.dtype),
            "codecs": self.codecs.to_json(),
            "attrs": dict(self.attrs),
            "role": self.role.value,
            "immutable": self.immutable,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> VariableSchema:
        dtype = numpy.dtype(payload["dtype"])
        return cls(
            name=payload["name"],
            dtype=dtype,
            dimensions=tuple(payload["dimensions"]),
            fill_value=_decode_fill(payload.get("fill_value"), dtype),
            codecs=CodecStack.from_json(payload.get("codecs", {})),
            attrs=dict(payload.get("attrs", {})),
            role=VariableRole(payload.get("role", VariableRole.USER.value)),
            immutable=bool(payload.get("immutable", False)),
        )


def _encode_fill(value: Any, dtype: numpy.dtype) -> Any:
    if value is None:
        return None
    if isinstance(value, numpy.generic):
        value = value.item()
    if isinstance(value, float) and numpy.isnan(value):
        return "NaN"
    if isinstance(value, float) and value == float("inf"):
        return "Infinity"
    if isinstance(value, float) and value == float("-inf"):
        return "-Infinity"
    return value


def _decode_fill(value: Any, dtype: numpy.dtype) -> Any:
    if value is None:
        return None
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    try:
        return numpy.array(value, dtype=dtype).item()
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"could not decode fill_value {value!r} for dtype {dtype}"
        ) from exc
