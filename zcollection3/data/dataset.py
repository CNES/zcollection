"""In-memory Dataset bound to a :class:`DatasetSchema`."""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping


from ..codecs import CodecStack
from ..schema import (
    DatasetSchema,
    SchemaBuilder,
)
from .variable import Variable

if TYPE_CHECKING:
    import xarray


class Dataset:
    """A schema plus the in-memory data for one or more partitions."""

    __slots__ = ("_attrs", "_variables", "schema")

    def __init__(
        self,
        schema: DatasetSchema,
        variables: Mapping[str, Variable] | Iterable[Variable],
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(variables, Mapping):
            items = variables.items()
        else:
            items = ((v.name, v) for v in variables)
        self._variables: OrderedDict[str, Variable] = OrderedDict(items)
        self.schema = schema
        self._attrs: dict[str, Any] = dict(attrs or schema.attrs)
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

    # Mapping-style API ------------------------------------------------

    def __getitem__(self, name: str) -> Variable:
        return self._variables[name]

    def __contains__(self, name: object) -> bool:
        return name in self._variables

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    @property
    def variables(self) -> Mapping[str, Variable]:
        return self._variables

    @property
    def attrs(self) -> dict[str, Any]:
        return self._attrs

    @property
    def dimensions(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for var in self._variables.values():
            for dim, size in zip(var.dimensions, var.shape, strict=False):
                sizes.setdefault(dim, size)
        return sizes

    @property
    def is_lazy(self) -> bool:
        return any(v.is_lazy for v in self._variables.values())

    def select(self, names: Iterable[str]) -> "Dataset":
        wanted = list(names)
        sub_schema = self.schema.select(wanted)
        return Dataset(
            schema=sub_schema,
            variables={n: self._variables[n] for n in wanted},
            attrs=self._attrs,
        )

    # xarray bridge ----------------------------------------------------

    def to_xarray(self) -> "xarray.Dataset":
        import xarray as xr  # noqa: PLC0415 — lazy: xarray is heavy

        data_vars = {}
        for name, var in self._variables.items():
            data_vars[name] = xr.Variable(
                dims=var.dimensions,
                data=var.data,
                attrs=var.attrs,
                fastpath=False,
            )
        return xr.Dataset(data_vars=data_vars, attrs=dict(self._attrs))

    @classmethod
    def from_xarray(cls, ds: "xarray.Dataset") -> "Dataset":
        builder = SchemaBuilder()
        for dim_name, size in ds.sizes.items():
            builder.with_dimension(dim_name, size=int(size))
        for k, v in ds.attrs.items():
            builder.with_attribute(str(k), v)

        # Combine coords + data_vars: both become variables in zcollection.
        all_vars: OrderedDict[str, "xarray.Variable"] = OrderedDict()
        for n, v in ds.coords.items():
            all_vars[str(n)] = v.variable
        for n, v in ds.data_vars.items():
            all_vars[str(n)] = v.variable

        for name, xrv in all_vars.items():
            builder.with_variable(
                name,
                dtype=xrv.dtype,
                dimensions=tuple(str(d) for d in xrv.dims),
                fill_value=xrv.attrs.get("_FillValue"),
                codecs=CodecStack(),
                attrs={k: v for k, v in xrv.attrs.items() if k != "_FillValue"},
            )
        schema = builder.build()
        variables = {
            name: Variable(schema.variables[name], xrv.data)
            for name, xrv in all_vars.items()
        }
        return cls(schema=schema, variables=variables, attrs=dict(ds.attrs))

    def __repr__(self) -> str:
        return (
            f"Dataset(vars={list(self._variables)}, "
            f"dims={self.dimensions}, lazy={self.is_lazy})"
        )
