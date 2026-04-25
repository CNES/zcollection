"""Per-partition Zarr v3 group I/O — async path (Phase 2).

Mirrors :mod:`zcollection3.io.partition` but uses :mod:`zarr.api.asynchronous`
end-to-end so that callers can run many partition reads/writes concurrently
on a single event loop.
"""
from __future__ import annotations

import asyncio
import logging
import warnings
from typing import TYPE_CHECKING, Any, Iterable

import numpy
import zarr.api.asynchronous as zarr_async
from zarr.errors import ZarrUserWarning

from ..codecs import resolve_codec
from ..data import Dataset, Variable
from ..store import join_path
from .partition import _chunks_for

if TYPE_CHECKING:
    from ..schema import DatasetSchema, VariableSchema
    from ..store import Store

_LOGGER = logging.getLogger(__name__)


def _build_codec_kwargs(schema: VariableSchema) -> dict[str, Any]:
    stack = schema.codecs
    if stack.array_to_bytes is None:
        return {}
    return {
        "filters": [resolve_codec(c) for c in stack.array_to_array] or "auto",
        "serializer": resolve_codec(stack.array_to_bytes),
        "compressors": [resolve_codec(c) for c in stack.bytes_to_bytes] or None,
    }


async def write_partition_dataset_async(
    store: Store,
    partition_path: str,
    dataset: Dataset,
    *,
    overwrite: bool = True,
    extra_attrs: dict[str, Any] | None = None,
    concurrency: int = 8,
) -> None:
    """Async write of one partition (group + arrays) using bounded concurrency."""
    _LOGGER.debug("Async writing partition %r", partition_path)

    zstore = store.zarr_store()
    group_attrs = dict(dataset.attrs)
    if extra_attrs:
        group_attrs.update(extra_attrs)

    group = await zarr_async.create_group(
        store=zstore,
        path=partition_path,
        overwrite=overwrite,
        attributes=group_attrs,
    )

    dim_chunks = dataset.schema.dim_chunks
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _write_one(name: str, var: Variable) -> None:
        async with sem:
            data = var.to_numpy()
            shape = data.shape
            chunks = _chunks_for(var.schema, shape, dim_chunks)
            kw = _build_codec_kwargs(var.schema)
            arr = await group.create_array(
                name=name,
                shape=shape,
                chunks=chunks,
                dtype=data.dtype,
                fill_value=var.fill_value,
                attributes=dict(var.attrs),
                dimension_names=list(var.dimensions),
                overwrite=overwrite,
                **kw,
            )
            await arr.setitem(slice(None), data)

    await asyncio.gather(*[
        _write_one(name, var) for name, var in dataset.variables.items()
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZarrUserWarning)
        await zarr_async.consolidate_metadata(zstore, path=partition_path)


async def open_partition_dataset_async(
    store: Store,
    partition_path: str,
    schema: DatasetSchema,
    *,
    variables: Iterable[str] | None = None,
    concurrency: int = 8,
) -> Dataset:
    """Async open of one partition group; reads variable arrays concurrently."""
    zstore = store.zarr_store()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZarrUserWarning)
        group = await zarr_async.open_group(
            store=zstore, path=partition_path, mode="r"
        )

    wanted = set(variables) if variables is not None else None
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _read_one(name: str) -> tuple[str, Variable] | None:
        async with sem:
            try:
                zarr_arr = await group.getitem(name)
            except KeyError:
                return None
            data = await zarr_arr.getitem(Ellipsis)
            return name, Variable(schema.variables[name], numpy.asarray(data))

    names = [
        n for n in schema.variables
        if (wanted is None or n in wanted)
    ]
    results = await asyncio.gather(*[_read_one(n) for n in names])

    out: dict[str, Variable] = {}
    for item in results:
        if item is None:
            continue
        name, var = item
        out[name] = var

    sub_schema = schema.select(out.keys()) if wanted is not None else schema
    attrs = dict(group.attrs)
    return Dataset(schema=sub_schema, variables=out, attrs=attrs)


async def partition_exists_async(store: Store, partition_path: str) -> bool:
    return store.exists(join_path(partition_path, "zarr.json"))
