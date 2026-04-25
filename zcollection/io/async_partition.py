"""Per-partition Zarr v3 group I/O — async path.

Mirrors :mod:`zcollection.io.partition` but uses :mod:`zarr.api.asynchronous`
end-to-end so that callers can run many partition reads/writes concurrently
on a single event loop.
"""

from typing import TYPE_CHECKING, Any
import asyncio
from collections.abc import Iterable
import logging

import numpy
import zarr.api.asynchronous as zarr_async

from ..data import Dataset, Group, Variable
from ..store import join_path
from .partition import _attach_nested, _build_array_kwargs, _chunks_for


if TYPE_CHECKING:
    from ..schema import DatasetSchema, GroupSchema
    from ..store import Store

_LOGGER = logging.getLogger(__name__)


async def _write_variable_async(
    zgroup: Any,
    var: Variable,
    dim_chunks: dict[str, int | None],
    *,
    overwrite: bool,
) -> None:
    """Write one :class:`Variable` as a Zarr array under ``zgroup`` (async)."""
    data = var.to_numpy()
    shape = data.shape
    inner_chunks = _chunks_for(var.schema, shape, dim_chunks)
    kw = _build_array_kwargs(var.schema, shape, inner_chunks, data.dtype)
    arr = await zgroup.create_array(
        name=var.name,
        shape=shape,
        dtype=data.dtype,
        fill_value=var.fill_value,
        attributes=dict(var.attrs),
        dimension_names=list(var.dimensions),
        overwrite=overwrite,
        **kw,
    )
    await arr.setitem(slice(None), data)


async def _write_group_async(
    zgroup: Any,
    group: Group,
    dim_chunks: dict[str, int | None],
    sem: asyncio.Semaphore,
    *,
    overwrite: bool,
) -> None:
    """Recursively write ``group`` (variables + child groups) under ``zgroup``."""

    async def _one_var(var: Variable) -> None:
        if var.schema.immutable:
            return
        async with sem:
            await _write_variable_async(
                zgroup, var, dim_chunks, overwrite=overwrite
            )

    var_tasks = [_one_var(v) for v in group.variables.values()]

    # Child groups must be created *before* recursing.
    child_tasks: list[Any] = []
    for name, child in group.groups.items():
        sub = await zgroup.create_group(
            name=name,
            attributes=dict(child.attrs),
            overwrite=overwrite,
        )
        child_tasks.append(
            _write_group_async(sub, child, dim_chunks, sem, overwrite=overwrite)
        )

    await asyncio.gather(*var_tasks, *child_tasks)


async def write_partition_dataset_async(
    store: Store,
    partition_path: str,
    dataset: Dataset,
    *,
    overwrite: bool = True,
    extra_attrs: dict[str, Any] | None = None,
    concurrency: int = 8,
) -> None:
    """Async write of one partition (root group + nested arrays/groups)."""
    _LOGGER.debug("Async writing partition %r", partition_path)

    zstore = store.zarr_store()
    group_attrs = dict(dataset.attrs)
    if extra_attrs:
        group_attrs.update(extra_attrs)

    root = await zarr_async.create_group(
        store=zstore,
        path=partition_path,
        overwrite=overwrite,
        attributes=group_attrs,
    )

    dim_chunks = dataset.schema.dim_chunks
    sem = asyncio.Semaphore(max(1, concurrency))
    await _write_group_async(
        root, dataset, dim_chunks, sem, overwrite=overwrite
    )


async def _read_group_async(
    zgroup: Any,
    group_schema: GroupSchema,
    sem: asyncio.Semaphore,
    *,
    wanted_paths: set[str] | None,
    prefix: str,
) -> tuple[
    dict[str, Variable],
    dict[str, dict[str, Variable]],
    dict[str, dict[str, Any]],
]:
    async def _one_var(name: str, vschema: Any) -> tuple[str, Variable] | None:
        async with sem:
            try:
                zarr_arr = await zgroup.getitem(name)
            except KeyError:
                return None
            data = await zarr_arr.getitem(Ellipsis)  # type: ignore[arg-type]
            return name, Variable(vschema, numpy.asarray(data))

    var_tasks = []
    for name, vschema in group_schema.variables.items():
        if vschema.immutable:
            continue
        path = f"{prefix}{name}" if prefix else name
        if wanted_paths is not None and path not in wanted_paths:
            continue
        var_tasks.append(_one_var(name, vschema))

    own: dict[str, Variable] = {}
    for item in await asyncio.gather(*var_tasks):
        if item is not None:
            own[item[0]] = item[1]

    nested_vars: dict[str, dict[str, Variable]] = {}
    nested_attrs: dict[str, dict[str, Any]] = {}
    for child_name, child_schema in group_schema.groups.items():
        try:
            child_zgroup = await zgroup.getitem(child_name)
        except KeyError:
            continue
        child_prefix = f"{prefix}{child_name}/" if prefix else f"{child_name}/"
        child_path = child_prefix.rstrip("/")
        nested_attrs[child_path] = dict(child_zgroup.attrs)
        sub_own, sub_nested, sub_attrs = await _read_group_async(
            child_zgroup,
            child_schema,
            sem,
            wanted_paths=wanted_paths,
            prefix=child_prefix,
        )
        nested_vars[child_path] = sub_own
        nested_vars.update(sub_nested)
        nested_attrs.update(sub_attrs)
    return own, nested_vars, nested_attrs


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
    root = await zarr_async.open_group(
        store=zstore,
        path=partition_path,
        mode="r",
    )

    wanted_paths: set[str] | None = None
    if variables is not None:
        wanted_paths = {n.lstrip("/") for n in variables}

    sem = asyncio.Semaphore(max(1, concurrency))
    own_vars, nested_vars, nested_attrs = await _read_group_async(
        root, schema, sem, wanted_paths=wanted_paths, prefix=""
    )

    dataset = Dataset(
        schema=schema,
        variables=own_vars,
        attrs=dict(root.attrs),
    )
    _attach_nested(dataset, schema, nested_vars, nested_attrs)
    return dataset


async def partition_exists_async(store: Store, partition_path: str) -> bool:
    """Return whether a partition exists at ``partition_path`` in ``store``."""
    return store.exists(join_path(partition_path, "zarr.json"))
