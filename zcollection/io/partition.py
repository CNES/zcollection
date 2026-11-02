# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Per-partition Zarr v3 group I/O — sync facade + helpers shared with async.

The sync ``write_partition_dataset`` / ``open_partition_dataset`` entry
points are thin ``run_sync`` wrappers over their async counterparts; the
array-building helpers (``_build_array_kwargs``, ``_chunks_for``) and the
tree-attach helper (``_attach_nested``) live here because the async
module imports them.
"""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterable

import numpy
import zarr.codecs as zcodecs

from ..codecs import resolve_codec
from ..codecs.sharding import shard_decision
from ..data import Dataset, Group, Variable
from ..store import join_path


if TYPE_CHECKING:
    from ..schema import DatasetSchema, GroupSchema, VariableSchema
    from ..store import Store


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
        if stack.sharded
        else None
    )

    if shard_shape is None:
        return {
            "chunks": inner_chunks,
            "filters": inner_filters or "auto",
            "serializer": inner_serializer,
            "compressors": inner_compressors or None,
        }

    inner_codecs: list[Any] = [
        *inner_filters,
        inner_serializer,
        *inner_compressors,
    ]
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


def _chunks_for(
    var: VariableSchema,
    shape: tuple[int, ...],
    dim_chunks: dict[str, int | None],
) -> tuple[int, ...]:
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
    """Write one partition's data as a (possibly nested) Zarr v3 group tree.

    Thin ``run_sync`` wrapper over :func:`write_partition_dataset_async`;
    kept so sync callers (and tests) have a non-``await`` entry point
    without maintaining a parallel recursive walker.
    """
    from ..dask.scheduler import run_sync
    from .async_partition import write_partition_dataset_async

    run_sync(
        write_partition_dataset_async(
            store,
            partition_path,
            dataset,
            overwrite=overwrite,
            extra_attrs=extra_attrs,
        )
    )


def partition_exists(store: Store, partition_path: str) -> bool:
    """Return whether a partition exists at ``partition_path`` in ``store``."""
    return store.exists(join_path(partition_path, "zarr.json"))


def open_partition_dataset(
    store: Store,
    partition_path: str,
    schema: DatasetSchema,
    *,
    variables: Iterable[str] | None = None,
) -> Dataset:
    """Open a partition's Zarr v3 group tree and return a :class:`Dataset`.

    Thin ``run_sync`` wrapper over :func:`open_partition_dataset_async`;
    the async form is the one implementation.
    """
    from ..dask.scheduler import run_sync
    from .async_partition import open_partition_dataset_async

    return run_sync(
        open_partition_dataset_async(
            store, partition_path, schema, variables=variables
        )
    )


def _attach_nested(
    parent: Group,
    parent_schema: GroupSchema,
    nested_vars: dict[str, dict[str, Variable]],
    nested_attrs: dict[str, dict[str, Any]],
    prefix: str = "",
) -> None:
    """Attach descendant groups to ``parent`` from the recursive read result."""
    for child_name, child_schema in parent_schema.groups.items():
        path = f"{prefix}{child_name}" if prefix else child_name
        own = nested_vars.get(path, {})
        attrs = nested_attrs.get(path, dict(child_schema.attrs))
        child = Group(
            schema=child_schema,
            variables=own,
            attrs=attrs,
            name=child_name,
            parent=parent,
        )
        parent._groups[child_name] = child
        _attach_nested(
            child, child_schema, nested_vars, nested_attrs, prefix=f"{path}/"
        )
