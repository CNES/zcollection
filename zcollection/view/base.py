"""Slim, v3-native View implementation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

from ..config import get as config_get
from ..data import Dataset, Variable
from ..errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    ReadOnlyError,
    ZCollectionError,
)
from ..io import (
    open_partition_dataset_async,
    partition_exists,
    write_partition_dataset_async,
)
from ..schema import DatasetSchema, VariableSchema

if TYPE_CHECKING:
    from ..collection import Collection
    from ..store import Store


VIEW_CONFIG_FILE: str = "_zcollection_view.json"
VIEW_FORMAT_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class ViewReference:
    """Pointer to a view's underlying base collection."""

    uri: str

    def to_json(self) -> dict[str, Any]:
        return {"uri": self.uri}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ViewReference":
        return cls(uri=str(payload["uri"]))


class View:
    """Overlay of extra variables on top of a base :class:`Collection`."""

    def __init__(
        self,
        *,
        store: Store,
        base: Collection,
        view_schema: DatasetSchema,
        reference: ViewReference,
        read_only: bool = False,
    ) -> None:
        self._store = store
        self._base = base
        self._view_schema = view_schema
        self._reference = reference
        self._read_only = read_only

    # --- construction -----------------------------------------------

    @classmethod
    def create(
        cls,
        store: Store,
        *,
        base: Collection,
        variables: Iterable[VariableSchema],
        reference: ViewReference | str,
        overwrite: bool = False,
    ) -> "View":
        if store.exists(VIEW_CONFIG_FILE) and not overwrite:
            raise CollectionExistsError(
                f"a view already exists at {store.root_uri}",
            )
        ref = (
            reference if isinstance(reference, ViewReference)
            else ViewReference(uri=reference)
        )
        view_vars = {v.name: _ensure_view_variable(v, base.schema) for v in variables}
        view_schema = DatasetSchema(
            dimensions=dict(base.schema.dimensions),
            variables=view_vars,
            attrs={},
        )
        payload = {
            "format_version": VIEW_FORMAT_VERSION,
            "reference": ref.to_json(),
            "schema": view_schema.to_json(),
        }
        store.write_bytes(
            VIEW_CONFIG_FILE,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
        return cls(
            store=store, base=base, view_schema=view_schema,
            reference=ref, read_only=False,
        )

    @classmethod
    def open(
        cls,
        store: Store,
        *,
        base: Collection,
        read_only: bool = False,
    ) -> "View":
        raw = store.read_bytes(VIEW_CONFIG_FILE)
        if raw is None:
            raise CollectionNotFoundError(
                f"no view config at {store.root_uri}",
            )
        payload = json.loads(raw.decode("utf-8"))
        ref = ViewReference.from_json(payload["reference"])
        view_schema = DatasetSchema.from_json(payload["schema"])
        return cls(
            store=store, base=base, view_schema=view_schema,
            reference=ref, read_only=read_only,
        )

    # --- properties -------------------------------------------------

    @property
    def store(self) -> Store:
        return self._store

    @property
    def base(self) -> Collection:
        return self._base

    @property
    def view_schema(self) -> DatasetSchema:
        return self._view_schema

    @property
    def reference(self) -> ViewReference:
        return self._reference

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._view_schema.variables)

    @property
    def read_only(self) -> bool:
        return self._read_only

    # --- listing ----------------------------------------------------

    def partitions(self, *, filters: str | None = None) -> Iterator[str]:
        return self._base.partitions(filters=filters)

    # --- query ------------------------------------------------------

    def query(
        self,
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Dataset | None:
        from ..dask.scheduler import run_sync  # noqa: PLC0415

        return run_sync(self.query_async(filters=filters, variables=variables))

    async def query_async(
        self,
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Dataset | None:
        wanted = set(variables) if variables is not None else None
        view_names = set(self._view_schema.variables)
        base_names = set(self._base.schema.variables)
        base_wanted = (
            None if wanted is None else sorted((wanted & base_names) | (wanted - view_names))
        )
        view_wanted = (
            None if wanted is None else sorted(wanted & view_names)
        )

        base_ds = await self._base.query_async(
            filters=filters, variables=base_wanted,
        )
        if base_ds is None:
            return None

        if view_wanted == []:
            return base_ds

        parts = list(self.partitions(filters=filters))
        if not parts:
            return base_ds

        concurrency = max(1, int(config_get("partition.concurrency")))
        sem = asyncio.Semaphore(concurrency)

        async def _load(path: str) -> Dataset | None:
            async with sem:
                if not partition_exists(self._store, path):
                    return None
                return await open_partition_dataset_async(
                    self._store, path, self._view_schema,
                    variables=view_wanted,
                )

        loaded = [d for d in await asyncio.gather(*[_load(p) for p in parts]) if d is not None]
        if not loaded:
            return base_ds

        view_ds = _concat_along(loaded, dim=self._base.partitioning.dimension)
        return _merge_overlay(base_ds, view_ds)

    # --- update -----------------------------------------------------

    def update(
        self,
        fn: Callable[[Dataset], dict[str, Any]],
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> list[str]:
        """Compute view-variable arrays for each base partition and write them.

        ``fn`` receives the merged base+view dataset and must return a dict
        mapping view-variable names to numpy arrays sized along the
        partitioning dimension.
        """
        from ..dask.scheduler import run_sync  # noqa: PLC0415

        return run_sync(self.update_async(fn, filters=filters, variables=variables))

    async def update_async(
        self,
        fn: Callable[[Dataset], dict[str, Any]],
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> list[str]:
        self._require_writable()
        wanted = set(variables) if variables is not None else None
        view_names = set(self._view_schema.variables)
        base_names = set(self._base.schema.variables)
        base_wanted = (
            None if wanted is None else sorted((wanted & base_names) | (wanted - view_names))
        )

        parts = list(self.partitions(filters=filters))
        concurrency = max(1, int(config_get("partition.concurrency")))
        sem = asyncio.Semaphore(concurrency)

        async def _step(path: str) -> str:
            async with sem:
                base_ds = await open_partition_dataset_async(
                    self._base.store, path, self._base.schema,
                    variables=base_wanted,
                )
                produced = fn(base_ds)
                view_vars = {
                    name: Variable(self._view_schema.variables[name], data)
                    for name, data in produced.items()
                    if name in view_names
                }
                if not view_vars:
                    return path
                ds = Dataset(
                    schema=self._view_schema, variables=view_vars,
                )
                await write_partition_dataset_async(
                    self._store, path, ds,
                    overwrite=True, concurrency=concurrency,
                )
                return path

        return list(await asyncio.gather(*[_step(p) for p in parts]))

    # --- internal ---------------------------------------------------

    def _require_writable(self) -> None:
        if self._read_only:
            raise ReadOnlyError(f"view at {self._store.root_uri} is read-only")


# --- helpers --------------------------------------------------------


def _ensure_view_variable(
    var: VariableSchema, base_schema: DatasetSchema,
) -> VariableSchema:
    """Reject a view variable that collides with a base-collection name."""
    if var.name in base_schema.variables:
        raise ZCollectionError(
            f"view variable {var.name!r} collides with a base-collection variable",
        )
    for d in var.dimensions:
        if d not in base_schema.dimensions:
            raise ZCollectionError(
                f"view variable {var.name!r} references unknown dimension {d!r}",
            )
    return var


def _concat_along(parts: list[Dataset], *, dim: str) -> Dataset:
    import numpy  # noqa: PLC0415

    if len(parts) == 1:
        return parts[0]
    schema = parts[0].schema
    out: dict[str, Variable] = {}
    for name in parts[0].variables:
        ref = parts[0][name]
        if dim in ref.dimensions:
            axis = ref.dimensions.index(dim)
            data = numpy.concatenate(
                [p[name].to_numpy() for p in parts], axis=axis,
            )
        else:
            data = ref.to_numpy()
        out[name] = Variable(ref.schema, data)
    return Dataset(schema=schema, variables=out, attrs=parts[0].attrs)


def _merge_overlay(base: Dataset, overlay: Dataset) -> Dataset:
    """Return a dataset with overlay variables added on top of ``base``."""
    merged = dict(base.variables)
    merged.update(overlay.variables)
    return Dataset(schema=base.schema, variables=merged, attrs=base.attrs)
