"""Per-partition Zarr v3 group I/O — synchronous path."""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterable
import logging

import numpy
import zarr
import zarr.codecs as zcodecs

from ..codecs import resolve_codec
from ..codecs.sharding import shard_decision
from ..data import Dataset, Group, Variable
from ..store import join_path


if TYPE_CHECKING:
    from ..schema import DatasetSchema, GroupSchema, VariableSchema
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


def _write_variable(
    zgroup: zarr.Group,
    var: Variable,
    dim_chunks: dict[str, int | None],
    *,
    overwrite: bool,
) -> None:
    """Write one :class:`Variable` as a Zarr array under ``zgroup``."""
    data = var.to_numpy()
    shape = data.shape
    inner_chunks = _chunks_for(var.schema, shape, dim_chunks)
    array_kwargs = _build_array_kwargs(
        var.schema, shape, inner_chunks, data.dtype
    )
    arr = zgroup.create_array(
        name=var.name,
        shape=shape,
        dtype=data.dtype,
        fill_value=var.fill_value,
        attributes=dict(var.attrs),
        dimension_names=list(var.dimensions),
        overwrite=overwrite,
        **array_kwargs,
    )
    arr[...] = data


def _write_group(
    zgroup: zarr.Group,
    group: Group,
    dim_chunks: dict[str, int | None],
    *,
    overwrite: bool,
) -> None:
    """Recursively write ``group``'s variables and child groups under ``zgroup``."""
    for var in group.variables.values():
        if var.schema.immutable:
            continue  # immutable vars live in the root _immutable/ group
        _write_variable(zgroup, var, dim_chunks, overwrite=overwrite)

    for name, child in group.groups.items():
        sub = zgroup.create_group(
            name=name,
            attributes=dict(child.attrs),
            overwrite=overwrite,
        )
        _write_group(sub, child, dim_chunks, overwrite=overwrite)


def write_partition_dataset(
    store: Store,
    partition_path: str,
    dataset: Dataset,
    *,
    overwrite: bool = True,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Write one partition's data as a (possibly nested) Zarr v3 group tree."""
    _LOGGER.debug("Writing partition %r", partition_path)

    zstore = store.zarr_store()
    group_attrs = dict(dataset.attrs)
    if extra_attrs:
        group_attrs.update(extra_attrs)

    root = zarr.create_group(
        store=zstore,
        path=partition_path,
        overwrite=overwrite,
        attributes=group_attrs,
    )

    dim_chunks = dataset.schema.dim_chunks
    _write_group(root, dataset, dim_chunks, overwrite=overwrite)


def partition_exists(store: Store, partition_path: str) -> bool:
    """Return whether a partition exists at ``partition_path`` in ``store``."""
    return store.exists(join_path(partition_path, "zarr.json"))


def _read_group(
    zgroup: zarr.Group,
    group_schema: GroupSchema,
    *,
    wanted_paths: set[str] | None,
    prefix: str,
) -> tuple[
    dict[str, Variable],
    dict[str, dict[str, Variable]],
    dict[str, dict[str, Any]],
]:
    """Recursively read variables under ``zgroup``.

    Returns:
        - own_vars: variables for this group, keyed by short name.
        - nested_vars: dict of ``{group_path: {var_name: Variable}}`` for all
          descendants of this group.
        - nested_attrs: dict of ``{group_path: {attr: value}}`` for all
          descendants (so the caller can attach attrs when building Group
          instances).

    """
    own: dict[str, Variable] = {}
    for name, vschema in group_schema.variables.items():
        if vschema.immutable:
            continue
        path = f"{prefix}{name}" if prefix else name
        if wanted_paths is not None and path not in wanted_paths:
            continue
        if name not in zgroup:
            continue
        zarr_arr = zgroup[name]
        data = numpy.asarray(zarr_arr[...])  # type: ignore[index]
        own[name] = Variable(vschema, data)

    nested_vars: dict[str, dict[str, Variable]] = {}
    nested_attrs: dict[str, dict[str, Any]] = {}
    for child_name, child_schema in group_schema.groups.items():
        if child_name not in zgroup:
            continue
        child_zgroup = zgroup[child_name]
        if not isinstance(child_zgroup, zarr.Group):
            continue
        child_prefix = f"{prefix}{child_name}/" if prefix else f"{child_name}/"
        child_path = child_prefix.rstrip("/")
        nested_attrs[child_path] = dict(child_zgroup.attrs)
        sub_own, sub_nested, sub_attrs = _read_group(
            child_zgroup,
            child_schema,
            wanted_paths=wanted_paths,
            prefix=child_prefix,
        )
        nested_vars[child_path] = sub_own
        nested_vars.update(sub_nested)
        nested_attrs.update(sub_attrs)
    return own, nested_vars, nested_attrs


def open_partition_dataset(
    store: Store,
    partition_path: str,
    schema: DatasetSchema,
    *,
    variables: Iterable[str] | None = None,
) -> Dataset:
    """Open a partition's Zarr v3 group tree and return a :class:`Dataset`.

    This sync helper loads variables eagerly into numpy; for non-blocking
    loading, use :func:`open_partition_dataset_async`.
    """
    zstore = store.zarr_store()
    root_zgroup = zarr.open_group(store=zstore, path=partition_path, mode="r")

    wanted_paths: set[str] | None = None
    if variables is not None:
        wanted_paths = {n.lstrip("/") for n in variables}

    own_vars, nested_vars, nested_attrs = _read_group(
        root_zgroup, schema, wanted_paths=wanted_paths, prefix=""
    )

    # Build child Group objects bottom-up.
    dataset = Dataset(
        schema=schema,
        variables=own_vars,
        attrs=dict(root_zgroup.attrs),
    )
    _attach_nested(dataset, schema, nested_vars, nested_attrs)
    return dataset


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
