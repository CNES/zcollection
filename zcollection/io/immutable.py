"""Read/write the ``_immutable/`` group.

The immutable group holds variables that do not span the partitioning
axis. They are written *once* at the collection root and merged into the
dataset returned by every partition open, instead of being duplicated in
each partition group. This shaves both PUT count on insert and GET count
on cold open by ``O(N_partitions)`` for large collections.
"""

from typing import TYPE_CHECKING
from collections.abc import Iterable
import logging

import numpy
import zarr
import zarr.api.asynchronous as zarr_async

from ..data import Dataset, Variable
from ..store.layout import IMMUTABLE_DIR
from .partition import _build_array_kwargs, _chunks_for

if TYPE_CHECKING:
    from ..schema import DatasetSchema
    from ..store import Store

_LOGGER = logging.getLogger(__name__)


def write_immutable_dataset(
    store: Store,
    dataset: Dataset,
    *,
    overwrite: bool = True,
) -> list[str]:
    """Write the dataset's immutable variables under ``_immutable/``.

    Returns the list of variable names written. Variables already declared
    immutable in the schema are written; the rest are silently skipped.
    """
    names = [
        name for name, var in dataset.variables.items() if var.schema.immutable
    ]
    if not names:
        return []

    zstore = store.zarr_store()
    group = zarr.create_group(
        store=zstore,
        path=IMMUTABLE_DIR,
        overwrite=overwrite,
        attributes={},
    )
    dim_chunks = dataset.schema.dim_chunks
    for name in names:
        var = dataset.variables[name]
        data = var.to_numpy()
        shape = data.shape
        inner_chunks = _chunks_for(var.schema, shape, dim_chunks)
        kw = _build_array_kwargs(var.schema, shape, inner_chunks, data.dtype)
        arr = group.create_array(
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

    return names


def immutable_group_exists(store: Store) -> bool:
    return store.exists(f"{IMMUTABLE_DIR}/zarr.json")


async def open_immutable_dataset_async(
    store: Store,
    schema: DatasetSchema,
    *,
    variables: Iterable[str] | None = None,
) -> dict[str, Variable]:
    """Open the ``_immutable/`` group and return its variables.

    Returns an empty dict if the group is missing. ``variables`` filters by
    name; immutable variables not present in the schema are ignored.
    """
    if not immutable_group_exists(store):
        return {}

    wanted = set(variables) if variables is not None else None
    immutable_names = {n for n, v in schema.variables.items() if v.immutable}
    targets = immutable_names if wanted is None else immutable_names & wanted
    if not targets:
        return {}

    zstore = store.zarr_store()
    group = await zarr_async.open_group(
        store=zstore,
        path=IMMUTABLE_DIR,
        mode="r",
    )

    out: dict[str, Variable] = {}
    for name in targets:
        try:
            zarr_arr = await group.getitem(name)
        except KeyError:
            continue
        data = await zarr_arr.getitem(Ellipsis)  # type: ignore[arg-type]
        out[name] = Variable(schema.variables[name], numpy.asarray(data))
    return out
