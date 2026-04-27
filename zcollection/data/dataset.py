# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Root :class:`Group` bound to a :class:`DatasetSchema`.

A :class:`Dataset` is the root group of a zcollection tree:
``name == "/"``, ``parent is None``, and its
:attr:`Group.schema` is a :class:`DatasetSchema` (which extends
:class:`~zcollection.schema.GroupSchema` with versioning and JSON
round-tripping). On top of plain :class:`Group` semantics it adds:

- A path-aware constructor that routes variables with ``/`` in
  their key into nested groups instead of placing them at the
  root.
- An xarray bridge — :meth:`Dataset.to_xarray` (with optional
  ``group=`` argument) and :meth:`Dataset.from_xarray` (always
  builds a flat dataset, since xarray has no native group concept).
"""

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

    A :class:`Dataset` is the root :class:`Group` of a zcollection
    tree (``name == "/"``, ``parent is None``). Compared to a plain
    :class:`Group`, the :attr:`schema` attribute is narrowed to
    :class:`DatasetSchema` (statically; the runtime slot is the same)
    and the constructor has a path-aware variable router.

    Args:
        schema: The dataset schema (root :class:`GroupSchema`).
        variables: Variables for the dataset. Two forms are accepted:

            - a mapping ``name -> Variable``: keys without ``/`` are
              placed at the root; keys containing ``/`` are routed
              into the matching nested group (intermediate groups
              are auto-built from ``schema``);
            - an iterable of :class:`Variable` instances — each is
              placed at the root using its own ``schema.name``.
        groups: Optional pre-built child groups for the root. They
            are merged with the auto-built nested groups, with
            ``groups`` winning on name conflict for groups it
            already provides.
        attrs: Optional dataset-level attributes; defaults to
            ``schema.attrs``.

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
        """Initialize the dataset (root group).

        Variables in ``variables`` whose key contains ``/`` are
        **routed** into the matching nested group via
        :func:`_build_groups_from_schema` (intermediate groups are
        created on demand from ``schema``). This is the typical way
        a partition dataset is built::

            zc.Dataset(
                schema=schema,
                variables={
                    "time": ...,
                    "data_01/ku/power": ...,  # placed under /data_01/ku
                },
            )

        See the class docstring for the full constructor semantics.
        """
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
        """Return a new dataset restricted to the named variables.

        Both the variables *and* the schema are subsetted: the
        returned dataset's :attr:`schema` is built via
        :meth:`DatasetSchema.select`, which prunes any group that
        ends up empty after the selection.

        ``names`` may be short names (resolved against the root
        group) or absolute paths (``/grp/var``).
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

    def to_xarray(self, group: str | None = None) -> xarray.Dataset:
        """Convert one group's variables to an xarray ``Dataset``.

        xarray has no native group concept, so a single
        :class:`xarray.Dataset` carries exactly one flat namespace of
        variables. Use ``group`` to pick which of this dataset's groups
        is materialised; the rest of the tree is dropped silently.

        Args:
            group: Absolute or relative path of the group to convert.
                ``None`` (default) and ``"/"`` both target the root
                group (backward-compatible behaviour). For a nested
                group, pass its path (e.g. ``"/data_01"`` or
                ``"data_01/ku"``); ``get_group`` semantics apply.

        Returns:
            An :class:`xarray.Dataset` whose ``variables`` are the
            named group's variables and whose ``attrs`` are that
            group's attributes.

        Raises:
            KeyError: If ``group`` does not resolve to a known group
                in this dataset's tree.

        """
        import xarray as xr

        target: Group = (
            self
            if group is None or group in ("/", "")
            else self.get_group(group)
        )
        data_vars = {}
        for name, var in target.variables.items():
            data_vars[name] = xr.Variable(
                dims=var.dimensions,
                data=var.data,
                attrs=var.attrs,
                fastpath=False,
            )
        return xr.Dataset(data_vars=data_vars, attrs=dict(target.attrs))

    @classmethod
    def from_xarray(cls, ds: xarray.Dataset) -> Dataset:
        """Build a :class:`Dataset` from an xarray ``Dataset`` (flat).

        Schema fields are inferred from the input:

        - dimensions come from ``ds.sizes``;
        - dataset attributes from ``ds.attrs``;
        - each variable contributes its ``dtype``, ``dimensions``,
          ``attrs`` (minus ``_FillValue``, which becomes the
          variable's ``fill_value``); coordinates and data variables
          are merged into a single flat namespace.

        xarray has no native group concept, so the resulting
        :class:`Dataset` is always flat (root group only). Round-trip
        a hierarchical Dataset by calling :meth:`to_xarray` per group.

        Args:
            ds: The xarray Dataset to convert.

        Returns:
            A new :class:`Dataset` with an inferred schema and the
            same variable data (no copy — the underlying arrays are
            shared with ``ds``).

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
