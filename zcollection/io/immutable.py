# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Read/write the ``_immutable/`` group.

The immutable group holds variables that do not span the partitioning
axis. They are written *once* at the collection root and merged into the
dataset returned by every partition open, instead of being duplicated in
each partition group. This shaves both PUT count on insert and GET count
on cold open by ``O(N_partitions)`` for large collections.

Hierarchical layout: variables declared inside nested groups are written
under the matching nested path (e.g. an immutable variable at
``/data_01/ku/range`` is written to ``_immutable/data_01/ku/range``).
"""

from typing import TYPE_CHECKING
from collections.abc import Iterable
import logging

import numpy
import zarr
import zarr.api.asynchronous as zarr_async

from ..data import Dataset, Group, Variable
from ..store.layout import IMMUTABLE_DIR
from .partition import _build_array_kwargs, _chunks_for


if TYPE_CHECKING:
    from ..schema import DatasetSchema, GroupSchema
    from ..store import Store

_LOGGER = logging.getLogger(__name__)


def _immutable_groups_in_tree(group: Group) -> Iterable[tuple[str, Group]]:
    """Yield ``(absolute_path, group)`` for every group containing immutable vars."""
    has_immutable = any(v.schema.immutable for v in group.variables.values())
    if has_immutable:
        yield (group.long_name(), group)
    for child in group.iter_groups():
        if any(v.schema.immutable for v in child.variables.values()):
            yield (child.long_name(), child)


def write_immutable_dataset(
    store: Store,
    dataset: Dataset,
    *,
    overwrite: bool = True,
) -> list[str]:
    """Write the dataset's immutable variables under ``_immutable/``.

    Returns the absolute paths of the variables written. The on-disk layout
    mirrors the in-memory group hierarchy.
    """
    written: list[str] = []
    immutable_groups = list(_immutable_groups_in_tree(dataset))
    if not immutable_groups:
        return written

    zstore = store.zarr_store()
    root = zarr.create_group(
        store=zstore,
        path=IMMUTABLE_DIR,
        overwrite=overwrite,
        attributes={},
    )
    dim_chunks = dataset.schema.dim_chunks

    for path, group in immutable_groups:
        zgroup = _ensure_zarr_subgroup(root, path)
        for name, var in group.variables.items():
            if not var.schema.immutable:
                continue
            data = var.to_numpy()
            shape = data.shape
            inner_chunks = _chunks_for(var.schema, shape, dim_chunks)
            kw = _build_array_kwargs(
                var.schema, shape, inner_chunks, data.dtype
            )
            arr = zgroup.create_array(
                name=name,
                shape=shape,
                dtype=data.dtype,
                fill_value=var.fill_value,
                attributes=dict(var.attrs),
                dimension_names=list(var.dimensions),
                overwrite=overwrite,
                **kw,
            )
            arr[...] = data
            full = name if path == "/" else f"{path.rstrip('/')}/{name}"
            written.append(full)

    return written


def _ensure_zarr_subgroup(root: zarr.Group, path: str) -> zarr.Group:
    """Return (creating if needed) a Zarr subgroup at ``path`` under ``root``."""
    node = root
    for segment in [p for p in path.split("/") if p]:
        if segment in node:
            existing = node[segment]
            if not isinstance(existing, zarr.Group):
                raise ValueError(
                    f"expected a Zarr group at {segment!r}, found "
                    f"{type(existing).__name__}"
                )
            node = existing
        else:
            node = node.create_group(name=segment)
    return node


def immutable_group_exists(store: Store) -> bool:
    """Return whether the immutable group exists in ``store``."""
    return store.exists(f"{IMMUTABLE_DIR}/zarr.json")


async def open_immutable_dataset_async(
    store: Store,
    schema: DatasetSchema,
    *,
    variables: Iterable[str] | None = None,
) -> dict[str, Variable]:
    """Open the ``_immutable/`` tree and return its variables.

    Returns an empty dict if the group is missing. The keys of the returned
    mapping are *absolute* variable paths (``/grp/sub/var``) so callers can
    place each variable into the correct group on the dataset side. Names
    declared at the root keep their short form (``"name"``).

    ``variables`` filters by *short* name for backwards compatibility (a
    short name matches any descendant variable with that name).
    """
    if not immutable_group_exists(store):
        return {}

    zstore = store.zarr_store()
    root = await zarr_async.open_group(
        store=zstore,
        path=IMMUTABLE_DIR,
        mode="r",
    )

    wanted = set(variables) if variables is not None else None
    out: dict[str, Variable] = {}

    async def _walk(
        zgroup: object, schema_node: GroupSchema, prefix: str
    ) -> None:
        # Variables in this group
        for name, vschema in schema_node.variables.items():
            if not vschema.immutable:
                continue
            if wanted is not None and name not in wanted:
                continue
            try:
                zarr_arr = await zgroup.getitem(name)  # type: ignore[attr-defined]
            except KeyError:
                continue
            data = await zarr_arr.getitem(Ellipsis)  # type: ignore[arg-type]
            key = name if not prefix else f"{prefix}/{name}"
            out[key] = Variable(vschema, numpy.asarray(data))
        # Recurse into child groups
        for child_name, child_schema in schema_node.groups.items():
            try:
                child_zgroup = await zgroup.getitem(child_name)  # type: ignore[attr-defined]
            except KeyError:
                continue
            child_prefix = (
                child_name if not prefix else f"{prefix}/{child_name}"
            )
            await _walk(child_zgroup, child_schema, child_prefix)

    await _walk(root, schema, "")
    return out
