# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Root :class:`Group` bound to a :class:`DatasetSchema`."""

from typing import TYPE_CHECKING, Any
from collections import OrderedDict
from collections.abc import Iterable, Mapping

from ..codecs import CodecStack
from ..schema import DatasetSchema, GroupSchema, SchemaBuilder
from .group import Group
from .variable import Variable


if TYPE_CHECKING:
    import xarray


def _build_groups_from_schema(
    schema: GroupSchema,
    variables_by_path: dict[str, Variable],
    *,
    parent: Group | None,
    prefix: str,
) -> dict[str, Group]:
    """Recursively build :class:`Group` objects for each child of ``schema``.

    ``variables_by_path`` maps absolute (``"/"``-joined) variable paths to
    :class:`Variable` instances. Only variables present in the mapping are
    attached; missing entries are silently skipped (typical for partitions
    that don't carry every declared variable).
    """
    out: dict[str, Group] = {}
    for name, child_schema in schema.groups.items():
        full_prefix = f"{prefix}{name}/"
        own_vars = {
            vname: variables_by_path[f"{full_prefix}{vname}"]
            for vname in child_schema.variables
            if f"{full_prefix}{vname}" in variables_by_path
        }
        child = Group(
            schema=child_schema,
            variables=own_vars,
            attrs=dict(child_schema.attrs),
            name=name,
            parent=parent,
        )
        child._groups.update(
            _build_groups_from_schema(
                child_schema,
                variables_by_path,
                parent=child,
                prefix=full_prefix,
            )
        )
        out[name] = child
    return out


class Dataset(Group):
    """A schema plus the in-memory data for one or more partitions.

    A :class:`Dataset` is the root :class:`Group` (``name == "/"``,
    ``parent is None``).

    Args:
        schema: The dataset schema (root :class:`GroupSchema`).
        variables: Variables for the root group. Either a mapping of
            ``name -> Variable`` (short names addressing the root group, or
            absolute paths populating nested groups), or an iterable of
            :class:`Variable` instances (which are placed at the root).
        groups: Optional pre-built child groups for the root.
        attrs: Optional dataset-level attributes; defaults to ``schema.attrs``.

    """

    __slots__ = ()

    if TYPE_CHECKING:
        # Narrow the inherited ``Group.schema`` type for ``Dataset`` users so
        # callers see the richer :class:`DatasetSchema` API (``dim_chunks``,
        # ``select``, ``with_partition_axis`` …) without a runtime cast.
        schema: DatasetSchema  # type: ignore[assignment]

    def __init__(
        self,
        schema: DatasetSchema,
        variables: Mapping[str, Variable] | Iterable[Variable] = (),
        groups: Mapping[str, Group] | Iterable[Group] = (),
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the dataset (root group)."""
        # Split variables by destination group: keys without a "/" go to the
        # root; keys containing "/" address nested groups.
        if isinstance(variables, Mapping):
            items: Iterable[tuple[str, Variable]] = variables.items()
        else:
            items = ((v.name, v) for v in variables)
        flat: dict[str, Variable] = dict(items)

        root_vars: OrderedDict[str, Variable] = OrderedDict()
        nested_by_path: dict[str, Variable] = {}
        for key, var in flat.items():
            if "/" in key:
                nested_by_path[key.lstrip("/")] = var
            else:
                root_vars[key] = var

        super().__init__(
            schema=schema,
            variables=root_vars,
            groups=groups,
            attrs=attrs,
            name="/",
            parent=None,
        )

        if nested_by_path:
            # Merge nested-path variables into the appropriate child groups,
            # creating empty intermediate groups as needed.
            built = _build_groups_from_schema(
                schema, nested_by_path, parent=self, prefix=""
            )
            for n, g in built.items():
                if n not in self._groups:
                    self._groups[n] = g

    # Convenience views ------------------------------------------------

    def select(self, names: Iterable[str]) -> Dataset:
        """Return a new dataset containing only the named variables.

        ``names`` may be short names (resolved against the root group) or
        absolute paths (``/grp/var``). Empty groups are pruned.
        """
        wanted = list(names)
        sub_schema = self.schema.select(wanted)
        # Materialise the variables mentioned by ``names`` from the existing
        # tree. Short names address the root group; absolute paths address
        # nested groups.
        new_vars: dict[str, Variable] = {}
        for n in wanted:
            if "/" in n:
                key = n.lstrip("/")
                new_vars[key] = self.get_variable(n)
            else:
                new_vars[n] = self._variables[n]
        return Dataset(
            schema=sub_schema,
            variables=new_vars,
            attrs=self._attrs,
        )

    # xarray bridge ----------------------------------------------------

    def to_xarray(self) -> xarray.Dataset:
        """Convert the root group's variables to an xarray ``Dataset``.

        Nested groups are not represented (xarray has no group concept) and
        are silently dropped.
        """
        import xarray as xr

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
    def from_xarray(cls, ds: xarray.Dataset) -> Dataset:
        """Build a :class:`Dataset` from an xarray ``Dataset`` (flat).

        xarray has no native group concept, so the result is always a flat
        dataset (root group only).
        """
        builder = SchemaBuilder()
        for dim_name, size in ds.sizes.items():
            builder.with_dimension(str(dim_name), size=int(size))
        for k, v in ds.attrs.items():
            builder.with_attribute(str(k), v)

        # Combine coords + data_vars: both become variables in zcollection.
        all_vars: OrderedDict[str, xarray.Variable] = OrderedDict()
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
