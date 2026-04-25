"""Per-partition Zarr v3 group I/O (sync, Phase 1)."""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Iterable

import numpy
import zarr
import zarr.codecs as zcodecs
from zarr.errors import ZarrUserWarning

from ..codecs import resolve_codec
from ..codecs.sharding import shard_decision
from ..data import Dataset, Variable
from ..store import join_path

if TYPE_CHECKING:
    from ..schema import DatasetSchema, VariableSchema
    from ..store import Store

_LOGGER = logging.getLogger(__name__)


def _build_array_kwargs(
    schema: VariableSchema,
    shape: tuple[int, ...],
    inner_chunks: tuple[int, ...],
    dtype: numpy.dtype,
) -> dict[str, Any]:
    """Build kwargs for ``create_array``.

    When the variable's :class:`CodecStack` enables sharding, we promote the
    serializer to a :class:`ShardingCodec` whose inner ``chunk_shape`` is the
    inner-chunk size, while the outer ``chunks=`` becomes the shard shape.
    """
    stack = schema.codecs
    if stack.array_to_bytes is None:
        return {"chunks": inner_chunks}

    inner_filters = [resolve_codec(c) for c in stack.array_to_array]
    inner_compressors = [resolve_codec(c) for c in stack.bytes_to_bytes]
    inner_serializer = resolve_codec(stack.array_to_bytes)

    shard_shape = (
        shard_decision(
            inner_chunks=inner_chunks,
            shape=shape,
            dtype=dtype,
            target_shard_bytes=stack.shard_target_bytes,
        )
        if stack.sharded else None
    )

    if shard_shape is None:
        return {
            "chunks": inner_chunks,
            "filters": inner_filters or "auto",
            "serializer": inner_serializer,
            "compressors": inner_compressors or None,
        }

    inner_codecs: list[Any] = [*inner_filters, inner_serializer, *inner_compressors]
    sharding = zcodecs.ShardingCodec(
        chunk_shape=inner_chunks,
        codecs=inner_codecs,
    )
    return {
        "chunks": shard_shape,
        "filters": "auto",
        "serializer": sharding,
        "compressors": None,
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
        if var.schema.immutable:
            continue  # immutable vars live in the root _immutable/ group
        data = var.to_numpy()
        shape = data.shape
        inner_chunks = _chunks_for(var.schema, shape, dim_chunks)
        array_kwargs = _build_array_kwargs(var.schema, shape, inner_chunks, data.dtype)
        attrs = dict(var.attrs)
        # Persist Zarr v3 dimension names natively.
        arr = group.create_array(
            name=name,
            shape=shape,
            dtype=data.dtype,
            fill_value=var.fill_value,
            attributes=attrs,
            dimension_names=list(var.dimensions),
            overwrite=overwrite,
            **array_kwargs,
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
    for name, vschema in schema.variables.items():
        if wanted is not None and name not in wanted:
            continue
        if vschema.immutable:
            continue  # served from the root _immutable/ group
        if name not in group:
            continue  # variable not present in this partition
        zarr_arr = group[name]
        data = numpy.asarray(zarr_arr[...])
        out[name] = Variable(schema.variables[name], data)

    return Dataset(
        schema=schema,
        variables=out,
        attrs=dict(group.attrs),
    )
