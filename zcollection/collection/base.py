"""High-level Collection facade — sync + async surfaces."""

from typing import TYPE_CHECKING, Any
import asyncio
from collections.abc import Callable, Iterable, Iterator

import numpy

from . import merge as merge_mod
from ..config import get as config_get
from ..data import Dataset, Variable
from ..errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    PartitionError,
    ReadOnlyError,
)
from ..io import (
    open_immutable_dataset_async,
    open_partition_dataset_async,
    partition_exists,
    read_root_config,
    write_immutable_dataset,
    write_partition_dataset_async,
    write_root_config,
)
from ..partitioning import (
    Catalog,
    compile_filter,
    from_json as partitioning_from_json,
    key_to_dict,
    reconcile_catalog,
)
from ..schema import DatasetSchema
from ..schema.serde import CONFIG_FILE
from ..store.layout import CATALOG_DIR, IMMUTABLE_DIR, join_path

if TYPE_CHECKING:
    from ..partitioning import Partitioning
    from ..store import Store
    from .merge import MergeCallable

_RESERVED_TOP_LEVEL = {
    CATALOG_DIR,
    IMMUTABLE_DIR,
    CONFIG_FILE,
    "zarr.json",
    "_zc_meta",
}


class Collection:
    """A partitioned Zarr v3 collection on a :class:`Store`."""

    def __init__(
        self,
        store: Store,
        *,
        schema: DatasetSchema,
        axis: str,
        partitioning: Partitioning,
        catalog_enabled: bool = False,
        read_only: bool = False,
    ) -> None:
        self._store = store
        self._schema = schema
        self._axis = axis
        self._partitioning = partitioning
        self._catalog_enabled = catalog_enabled
        self._read_only = read_only
        self._catalog = Catalog(store) if catalog_enabled else None

    # --- Construction ------------------------------------------------

    @classmethod
    def create(
        cls,
        store: Store,
        *,
        schema: DatasetSchema,
        axis: str,
        partitioning: Partitioning,
        catalog_enabled: bool = False,
        overwrite: bool = False,
    ) -> Collection:
        if store.exists(CONFIG_FILE) and not overwrite:
            raise CollectionExistsError(
                f"a collection already exists at {store.root_uri}"
            )
        bound = schema.with_partition_axis(axis)
        with store.session():
            write_root_config(
                store,
                schema=bound,
                axis=axis,
                partitioning=partitioning.to_json(),
                catalog_enabled=catalog_enabled,
            )
        return cls(
            store=store,
            schema=bound,
            axis=axis,
            partitioning=partitioning,
            catalog_enabled=catalog_enabled,
        )

    @classmethod
    def open(cls, store: Store, *, read_only: bool = False) -> Collection:
        try:
            doc = read_root_config(store)
        except CollectionNotFoundError:
            raise
        schema = DatasetSchema.from_json(doc["schema"])
        partitioning = partitioning_from_json(doc["partitioning"])
        catalog_enabled = bool(doc.get("catalog", {}).get("enabled", False))
        return cls(
            store=store,
            schema=schema,
            axis=doc["axis"],
            partitioning=partitioning,
            catalog_enabled=catalog_enabled,
            read_only=read_only,
        )

    # --- Public properties -----------------------------------------

    @property
    def store(self) -> Store:
        return self._store

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    @property
    def axis(self) -> str:
        return self._axis

    @property
    def partitioning(self) -> Partitioning:
        return self._partitioning

    @property
    def read_only(self) -> bool:
        return self._read_only

    # --- Listing ---------------------------------------------------

    def partitions(self, *, filters: str | None = None) -> Iterator[str]:
        """Yield relative partition paths in sorted order, optionally filtered."""
        predicate = compile_filter(filters)
        for path in self._enumerate_partitions():
            try:
                key = self._partitioning.decode(path)
            except PartitionError:
                continue
            if predicate(key_to_dict(key)):
                yield path

    def _enumerate_partitions(self) -> list[str]:
        """Return the canonical partition list, preferring the catalog."""
        if self._catalog is not None:
            cached = self._catalog.read_paths()
            if cached is not None:
                return cached
        return sorted(self._walk_partitions())

    def _walk_partitions(self) -> Iterator[str]:
        depth = len(self._partitioning.axis)
        yield from self._walk(prefix="", depth=depth)

    def repair_catalog(self) -> list[str]:
        """Rebuild the catalog by walking the store; returns the new path list.

        Use after a crash, or after manual edits to the partition tree.
        Raises if the collection was not opened with ``catalog_enabled=True``.
        """
        if self._catalog is None:
            raise RuntimeError(
                "repair_catalog() requires a catalog-enabled collection",
            )
        self._require_writable()
        walked = sorted(self._walk_partitions())
        reconcile_catalog(self._catalog, walked)
        return walked

    def _walk(self, *, prefix: str, depth: int) -> Iterator[str]:
        for child in sorted(self._store.list_dir(prefix)):
            if not prefix and child in _RESERVED_TOP_LEVEL:
                continue
            full = join_path(prefix, child)
            if depth == 1:
                if partition_exists(self._store, full):
                    yield full
            else:
                yield from self._walk(prefix=full, depth=depth - 1)

    # --- Insert ----------------------------------------------------

    def insert(
        self,
        dataset: Dataset,
        *,
        overwrite: bool = True,
        merge: MergeCallable | str | None = None,
    ) -> list[str]:
        from ..dask.scheduler import run_sync

        return run_sync(
            self.insert_async(dataset, overwrite=overwrite, merge=merge)
        )

    async def insert_async(
        self,
        dataset: Dataset,
        *,
        overwrite: bool = True,
        merge: MergeCallable | str | None = None,
    ) -> list[str]:
        self._require_writable()
        merge_fn = merge_mod.resolve(merge)
        concurrency = max(1, int(config_get("partition.concurrency")))

        # The incoming dataset typically carries the *unbound* schema
        # (immutable=False everywhere). Rebind to the Collection's bound
        # schema so the writer can identify immutable variables.
        dataset = _rebind_to_schema(dataset, self._schema)

        # Wrap the whole insert in a store session so transactional backends
        # (Icechunk) commit atomically; for non-transactional stores this is
        # a no-op context.
        with self._store.session():
            # Immutable variables are written once at the root; they're
            # identical across partitions by definition.
            if any(v.schema.immutable for v in dataset.variables.values()):
                write_immutable_dataset(self._store, dataset, overwrite=True)

            plan: list[tuple[Any, slice, str]] = [
                (key, sl, self._partitioning.encode(key))
                for key, sl in self._partitioning.split(dataset)
            ]
            sem = asyncio.Semaphore(concurrency)

            async def _write(key, sl, path) -> str:
                async with sem:
                    sub = _slice_dataset(
                        dataset,
                        dim=self._partitioning.dimension,
                        sl=sl,
                    )
                    if merge is not None and partition_exists(
                        self._store, path
                    ):
                        existing = await open_partition_dataset_async(
                            self._store,
                            path,
                            self._schema,
                        )
                        sub = merge_fn(
                            existing,
                            sub,
                            axis=self._axis,
                            partitioning_dim=self._partitioning.dimension,
                        )
                    extra = {"_zc_partition_key": [list(t) for t in key]}
                    await write_partition_dataset_async(
                        self._store,
                        path,
                        sub,
                        overwrite=overwrite,
                        extra_attrs=extra,
                        concurrency=concurrency,
                    )
                    return path

            written = list(await asyncio.gather(*[_write(*t) for t in plan]))
            if self._catalog is not None and written:
                self._catalog.add(written)
            return written

    # --- Query -----------------------------------------------------

    def query(
        self,
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Dataset | None:
        from ..dask.scheduler import run_sync

        return run_sync(self.query_async(filters=filters, variables=variables))

    async def query_async(
        self,
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Dataset | None:
        wanted = list(variables) if variables is not None else None
        parts = list(self.partitions(filters=filters))
        if not parts:
            return None
        concurrency = max(1, int(config_get("partition.concurrency")))
        sem = asyncio.Semaphore(concurrency)

        async def _load(path: str) -> Dataset:
            async with sem:
                return await open_partition_dataset_async(
                    self._store,
                    path,
                    self._schema,
                    variables=wanted,
                )

        loaded_task = asyncio.gather(*[_load(p) for p in parts])
        immutable_task = open_immutable_dataset_async(
            self._store,
            self._schema,
            variables=wanted,
        )
        loaded, immutable = await asyncio.gather(loaded_task, immutable_task)
        merged = _concat_datasets(
            list(loaded),
            dim=self._partitioning.dimension,
        )
        return _attach_immutable(merged, immutable)

    # --- Drop ------------------------------------------------------

    def drop_partitions(self, *, filters: str | None = None) -> list[str]:
        self._require_writable()
        dropped: list[str] = []
        with self._store.session():
            for path in list(self.partitions(filters=filters)):
                self._store.delete_prefix(path)
                dropped.append(path)
            if self._catalog is not None and dropped:
                self._catalog.remove(dropped)
        return dropped

    # --- Map / Update ----------------------------------------------

    def map(
        self,
        fn: Callable[[Dataset], Any],
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Apply ``fn`` to each partition's dataset and return ``{path: result}``."""
        from ..dask.scheduler import run_sync

        return run_sync(
            self.map_async(fn, filters=filters, variables=variables)
        )

    async def map_async(
        self,
        fn: Callable[[Dataset], Any],
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        wanted = list(variables) if variables is not None else None
        parts = list(self.partitions(filters=filters))
        concurrency = max(1, int(config_get("partition.concurrency")))
        sem = asyncio.Semaphore(concurrency)
        immutable = await open_immutable_dataset_async(
            self._store,
            self._schema,
            variables=wanted,
        )

        async def _apply(path: str) -> tuple[str, Any]:
            async with sem:
                ds = await open_partition_dataset_async(
                    self._store,
                    path,
                    self._schema,
                    variables=wanted,
                )
                return path, fn(_attach_immutable(ds, immutable))

        results = await asyncio.gather(*[_apply(p) for p in parts])
        return dict(results)

    def update(
        self,
        fn: Callable[[Dataset], Dataset],
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> list[str]:
        """Read each partition, apply ``fn`` returning a new Dataset, write back."""
        from ..dask.scheduler import run_sync

        return run_sync(
            self.update_async(fn, filters=filters, variables=variables)
        )

    async def update_async(
        self,
        fn: Callable[[Dataset], Dataset],
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> list[str]:
        self._require_writable()
        wanted = list(variables) if variables is not None else None
        parts = list(self.partitions(filters=filters))
        concurrency = max(1, int(config_get("partition.concurrency")))
        sem = asyncio.Semaphore(concurrency)
        immutable = await open_immutable_dataset_async(
            self._store,
            self._schema,
            variables=wanted,
        )

        async def _update(path: str) -> str:
            async with sem:
                ds = await open_partition_dataset_async(
                    self._store,
                    path,
                    self._schema,
                    variables=wanted,
                )
                new_ds = fn(_attach_immutable(ds, immutable))
                await write_partition_dataset_async(
                    self._store,
                    path,
                    new_ds,
                    overwrite=True,
                    concurrency=concurrency,
                )
                return path

        with self._store.session():
            return list(await asyncio.gather(*[_update(p) for p in parts]))

    # --- Internal --------------------------------------------------

    def _require_writable(self) -> None:
        if self._read_only:
            raise ReadOnlyError(
                f"collection at {self._store.root_uri} is read-only"
            )


def _rebind_to_schema(dataset: Dataset, schema: DatasetSchema) -> Dataset:
    """Replace each variable's schema reference with ``schema``'s entry.

    Lets a Dataset built from an unbound schema participate in operations
    that depend on bound metadata (notably ``immutable``).
    """
    rebound: dict[str, Variable] = {}
    for name, var in dataset.variables.items():
        target = schema.variables.get(name)
        rebound[name] = Variable(target, var.to_numpy()) if target else var
    return Dataset(schema=schema, variables=rebound, attrs=dataset.attrs)


def _attach_immutable(
    dataset: Dataset,
    immutable: dict[str, Variable],
) -> Dataset:
    """Return ``dataset`` with the immutable vars merged in (dataset wins)."""
    if not immutable:
        return dataset
    merged = dict(immutable)
    merged.update(dataset.variables)
    return Dataset(
        schema=dataset.schema,
        variables=merged,
        attrs=dataset.attrs,
    )


def _slice_dataset(dataset: Dataset, *, dim: str, sl: slice) -> Dataset:
    """Slice every variable that spans ``dim`` by ``sl``; copy others."""
    new_vars: dict[str, Variable] = {}
    for name, var in dataset.variables.items():
        if dim in var.dimensions:
            axis = var.dimensions.index(dim)
            slicer = [slice(None)] * var.ndim
            slicer[axis] = sl
            data = var.to_numpy()[tuple(slicer)]
        else:
            data = var.to_numpy()
        new_vars[name] = Variable(var.schema, data)
    return Dataset(
        schema=dataset.schema, variables=new_vars, attrs=dataset.attrs
    )


def _concat_datasets(parts: list[Dataset], *, dim: str) -> Dataset:
    if len(parts) == 1:
        return parts[0]
    schema = parts[0].schema
    names = list(parts[0].variables)
    merged: dict[str, Variable] = {}
    for name in names:
        ref = parts[0][name]
        if dim in ref.dimensions:
            axis = ref.dimensions.index(dim)
            arrs = [p[name].to_numpy() for p in parts]
            data = numpy.concatenate(arrs, axis=axis)
        else:
            data = ref.to_numpy()
        merged[name] = Variable(ref.schema, data)
    return Dataset(schema=schema, variables=merged, attrs=parts[0].attrs)
