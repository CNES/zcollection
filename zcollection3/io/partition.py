"""Per-partition Zarr v3 group I/O (sync, Phase 1)."""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Iterable

import numpy
import zarr
from zarr.errors import ZarrUserWarning

from ..codecs import resolve_codec
from ..data import Dataset, Variable
from ..store import join_path

if TYPE_CHECKING:
    from ..schema import DatasetSchema, VariableSchema
    from ..store import Store

_LOGGER = logging.getLogger(__name__)


def _build_codec_kwargs(schema: VariableSchema) -> dict[str, Any]:
    """Map our CodecStack onto Zarr v3's filters/serializer/compressors trio.

    Returns kwargs to splat into ``Group.create_array``. If the stack has no
    explicit serializer, return an empty dict so Zarr picks defaults.
    """
    stack = schema.codecs
    if stack.array_to_bytes is None:
        return {}
    return {
        "filters": [resolve_codec(c) for c in stack.array_to_array] or "auto",
        "serializer": resolve_codec(stack.array_to_bytes),
        "compressors": [resolve_codec(c) for c in stack.bytes_to_bytes] or None,
    }


def _chunks_for(var: VariableSchema, shape: tuple[int, ...],
                dim_chunks: dict[str, int | None]) -> tuple[int, ...]:
    out: list[int] = []
    for dim, size in zip(var.dimensions, shape, strict=True):
        c = dim_chunks.get(dim)
        out.append(c if c is not None and c > 0 else size)
    return tuple(out)


def write_partition_dataset(
    store: Store,
    partition_path: str,
    dataset: Dataset,
    *,
    overwrite: bool = True,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Write one partition's data as a Zarr v3 group with consolidated metadata."""
    _LOGGER.debug("Writing partition %r", partition_path)

    zstore = store.zarr_store()
    group_attrs = dict(dataset.attrs)
    if extra_attrs:
        group_attrs.update(extra_attrs)

    group = zarr.create_group(
        store=zstore,
        path=partition_path,
        overwrite=overwrite,
        attributes=group_attrs,
    )

    dim_chunks = dataset.schema.dim_chunks
    for name, var in dataset.variables.items():
        data = var.to_numpy()
        shape = data.shape
        chunks = _chunks_for(var.schema, shape, dim_chunks)
        codec_kwargs = _build_codec_kwargs(var.schema)
        attrs = dict(var.attrs)
        # Persist Zarr v3 dimension names natively.
        arr = group.create_array(
            name=name,
            shape=shape,
            chunks=chunks,
            dtype=data.dtype,
            fill_value=var.fill_value,
            attributes=attrs,
            dimension_names=list(var.dimensions),
            overwrite=overwrite,
            **codec_kwargs,
        )
        arr[...] = data

    # Per-partition consolidation is by design: one GET reopens the whole
    # partition. The Zarr v3 spec hasn't blessed it yet, so silence the
    # advisory warning rather than letting it bubble up to users.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZarrUserWarning)
        zarr.consolidate_metadata(zstore, path=partition_path)


def partition_exists(store: Store, partition_path: str) -> bool:
    return store.exists(join_path(partition_path, "zarr.json"))


def open_partition_dataset(
    store: Store,
    partition_path: str,
    schema: DatasetSchema,
    *,
    variables: Iterable[str] | None = None,
) -> Dataset:
    """Open a partition's Zarr v3 group and return a :class:`Dataset`.

    Phase 1 loads variables eagerly into numpy. Lazy / async loading lands
    in Phase 2.
    """
    zstore = store.zarr_store()
    group = zarr.open_group(store=zstore, path=partition_path, mode="r")

    wanted = set(variables) if variables is not None else None
    out: dict[str, Variable] = {}
    for name in schema.variables:
        if wanted is not None and name not in wanted:
            continue
        if name not in group:
            continue  # variable not present in this partition
        zarr_arr = group[name]
        data = numpy.asarray(zarr_arr[...])
        out[name] = Variable(schema.variables[name], data)

    sub_schema = schema.select(out.keys()) if wanted is not None else schema
    return Dataset(
        schema=sub_schema,
        variables=out,
        attrs=dict(group.attrs),
    )
