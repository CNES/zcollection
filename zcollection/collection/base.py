# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""High-level Collection facade — sync + async surfaces."""

from typing import TYPE_CHECKING, Any
import asyncio
from collections.abc import Callable, Iterable, Iterator

import numpy

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
from . import merge as merge_mod


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
        """Initialize a Collection.

        Args:
            store: Backing store for the collection.
            schema: Bound dataset schema describing variables.
            axis: Name of the partition axis.
            partitioning: Partitioning strategy.
            catalog_enabled: Whether to maintain a catalog of partitions.
            read_only: Whether the collection should refuse mutations.

        """
        #: The backing store for the collection.
        self._store = store
        #: The bound dataset schema describing variables.
        self._schema = schema
        #: The name of the partition axis.
        self._axis = axis
        #: The partitioning strategy.
        self._partitioning = partitioning
        #: Whether the catalog is enabled.
        self._catalog_enabled = catalog_enabled
        #: Whether the collection is read-only.
        self._read_only = read_only
        #: The catalog instance, if enabled.
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
        """Create a new collection on ``store`` and return its handle.

        The collection's root config (``_zcollection.json``) is written
        immediately. Variables that are not partitioned along ``axis`` are
        treated as immutable and lifted into the ``_immutable/`` group at
        first insert.

        Args:
            store: The backing :class:`~zcollection.store.Store`.
            schema: The dataset schema. It is rebound to ``axis`` so that
                variables that don't depend on ``axis`` become immutable.
            axis: Name of the partition axis (must be a dimension of
                ``schema``).
            partitioning: Partitioning strategy (e.g. ``Date``, ``Sequence``,
                ``GroupedSequence``).
            catalog_enabled: If ``True``, maintain a sharded ``_catalog/``
                group listing partition paths so that ``partitions()`` and
                cold opens skip the O(N) directory walk.
            overwrite: If ``True``, replace any existing collection root at
                this location. If ``False`` (default) and a root exists,
                :class:`~zcollection.errors.CollectionExistsError` is
                raised.

        Returns:
            The newly-created :class:`~zcollection.collection.base.Collection`, ready to ``insert``.

        Raises:
            ~zcollection.errors.CollectionExistsError: If a collection already exists at
                ``store.root_uri`` and ``overwrite=False``.

        """
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
        """Open an existing collection from ``store``.

        Reads the root config (axis, partitioning, schema, catalog flag)
        and returns a handle pointing at it.

        Args:
            store: The store backing the collection.
            read_only: If ``True``, mutating methods (``insert``,
                ``drop_partitions``, ``update``, ``repair_catalog``) raise
                :class:`~zcollection.errors.ReadOnlyError` instead of
                writing.

        Returns:
            A :class:`~zcollection.collection.base.Collection` bound to the existing root.

        Raises:
            ~zcollection.errors.CollectionNotFoundError: If no collection exists at
                ``store.root_uri``.

        """
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
        """Return the backing store."""
        return self._store

    @property
    def schema(self) -> DatasetSchema:
        """Return the bound dataset schema."""
        return self._schema

    @property
    def axis(self) -> str:
        """Return the name of the partition axis."""
        return self._axis

    @property
    def partitioning(self) -> Partitioning:
        """Return the partitioning strategy."""
        return self._partitioning

    @property
    def read_only(self) -> bool:
        """Return whether the collection is read-only."""
        return self._read_only

    # --- Listing ---------------------------------------------------

    def partitions(self, *, filters: str | None = None) -> Iterator[str]:
        """Yield relative partition paths in sorted order, optionally filtered.

        Args:
            filters: An optional partition-key expression (e.g.
                ``"year == 2024 and month >= 3"``, ``"cycle in [1, 2]"``).
                Only paths whose decoded key satisfies the expression are
                yielded. Comparable types are integers, strings, and
                anything that ``Partitioning.decode`` produces. ``None``
                yields every partition.

        Yields:
            Relative partition paths (e.g. ``"year=2024/month=03"``),
            sorted lexicographically.

        """
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
        """Insert ``dataset`` into the collection.

        The dataset is split by the collection's partitioning. Each slice
        is written under the matching partition path. Immutable variables
        (those that don't depend on the partition axis) are written once
        at the root and shared across partitions.

        On a transactional store (e.g.
        :class:`~zcollection.store.IcechunkStore`) the entire insert is
        wrapped in a single commit — a crash mid-insert leaves no partial
        state.

        Args:
            dataset: The data to insert. Variables can be backed by numpy,
                Dask, or Zarr ``AsyncArray``.
            overwrite: If a partition already exists, whether to overwrite
                the chunks (``True``, default) or fail. Has no effect when
                ``merge`` is set, since merging implies an overwrite of
                the merged result.
            merge: Strategy for combining the inserted data with an
                existing partition. Either a built-in name
                (``"replace"`` / ``"concat"`` / ``"time_series"`` /
                ``"upsert"``), a :class:`~zcollection.merge.MergeCallable`,
                or ``None`` (default — partitions are written as-is and
                existing chunks are overwritten by ``overwrite=True``).

        Returns:
            The list of partition paths that were written, in the order
            they were produced.

        Raises:
            ~zcollection.errors.ReadOnlyError: If the collection was opened with
                ``read_only=True``.
            KeyError: If ``merge`` is a string that doesn't match a
                built-in strategy.
            PartitionError: If ``dataset`` is missing variables required
                by the partitioning.

        """
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
        """Async variant of :meth:`insert`."""
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
            if any(
                v.schema.immutable for v in dataset.all_variables().values()
            ):
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
        """Read matching partitions and return the concatenated dataset.

        Partitions are loaded concurrently up to
        ``zcollection.config["partition.concurrency"]``. The result has
        the immutable group's variables merged in (the partition's data
        wins on name conflict).

        Args:
            filters: A partition-key predicate; see :meth:`partitions`.
                ``None`` reads every partition.
            variables: Iterable of variable names to load. ``None``
                (default) loads every variable. Loading a subset is the
                primary way to keep cold S3 reads cheap.

        Returns:
            The concatenated :class:`~zcollection.Dataset` along the
            partitioning dimension, or ``None`` if no partition matched
            ``filters``.

        """
        from ..dask.scheduler import run_sync

        return run_sync(self.query_async(filters=filters, variables=variables))

    async def query_async(
        self,
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Dataset | None:
        """Async variant of :meth:`query`."""
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
        """Delete matching partitions.

        Wrapped in a store session so transactional backends commit the
        whole drop atomically.

        Args:
            filters: A partition-key predicate; see :meth:`partitions`.
                If ``None``, **every** partition is dropped — pass an
                explicit filter when you don't mean that.

        Returns:
            The list of partition paths that were removed, in the order
            they were processed.

        Raises:
            ~zcollection.errors.ReadOnlyError: If the collection was opened with
                ``read_only=True``.

        """
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
        """Apply ``fn`` to each matching partition and collect the results.

        ``fn`` receives the partition :class:`~zcollection.Dataset` with
        the immutable group merged in. Read-only — the partition is not
        written back. Use :meth:`update` to persist transformed data.

        Args:
            fn: Callable applied to each partition dataset; its return
                value is stored against the partition path.
            filters: Partition-key predicate; see :meth:`partitions`.
            variables: Optional whitelist of variables to load.

        Returns:
            A mapping ``{partition_path: fn(dataset)}``.

        """
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
        """Async variant of :meth:`map`."""
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
        """Read each matching partition, transform it, and write it back.

        ``fn`` is called per partition with the merged dataset (immutable
        group attached). It must return a :class:`~zcollection.Dataset`
        with the same partitioning dimension and length, ready to
        replace the partition's contents.

        Args:
            fn: Pure function ``Dataset -> Dataset`` applied to each
                matching partition.
            filters: Partition-key predicate; see :meth:`partitions`.
            variables: Optional whitelist of variables to load before
                calling ``fn``. Variables that are not loaded remain
                untouched on disk.

        Returns:
            The list of partition paths that were written, in the order
            they were processed.

        Raises:
            ~zcollection.errors.ReadOnlyError: If the collection was opened with
                ``read_only=True``.

        """
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
        """Async variant of :meth:`update`."""
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

    Walks the full group tree so nested-group variables are rebound too.
    """
    all_vars = dataset.all_variables()
    all_schema_vars = schema.all_variables()
    rebound: dict[str, Variable] = {}
    for path, var in all_vars.items():
        target = all_schema_vars.get(path)
        rebound[path] = Variable(target, var.to_numpy()) if target else var
    return Dataset(schema=schema, variables=rebound, attrs=dataset.attrs)


def _attach_immutable(
    dataset: Dataset,
    immutable: dict[str, Variable],
) -> Dataset:
    """Return ``dataset`` with the immutable vars merged in (dataset wins).

    Keys of ``immutable`` may be short names (root scope) or absolute paths
    addressing nested groups; ``Dataset.__init__`` handles both.
    """
    if not immutable:
        return dataset
    merged: dict[str, Variable] = dict(immutable)
    merged.update(dataset.all_variables())
    return Dataset(
        schema=dataset.schema,
        variables=merged,
        attrs=dataset.attrs,
    )


def _slice_dataset(dataset: Dataset, *, dim: str, sl: slice) -> Dataset:
    """Slice every variable (root or nested) that spans ``dim`` by ``sl``."""
    new_vars: dict[str, Variable] = {}
    for path, var in dataset.all_variables().items():
        if dim in var.dimensions:
            axis = var.dimensions.index(dim)
            slicer = [slice(None)] * var.ndim
            slicer[axis] = sl
            data = var.to_numpy()[tuple(slicer)]
        else:
            data = var.to_numpy()
        new_vars[path] = Variable(var.schema, data)
    return Dataset(
        schema=dataset.schema, variables=new_vars, attrs=dataset.attrs
    )


def _concat_datasets(parts: list[Dataset], *, dim: str) -> Dataset:
    """Concatenate a list of datasets along ``dim``, recursing into groups."""
    if len(parts) == 1:
        return parts[0]
    schema = parts[0].schema
    paths = list(parts[0].all_variables())
    parts_vars = [p.all_variables() for p in parts]
    merged: dict[str, Variable] = {}
    for path in paths:
        ref = parts_vars[0][path]
        if dim in ref.dimensions:
            axis = ref.dimensions.index(dim)
            arrs = [pv[path].to_numpy() for pv in parts_vars]
            data = numpy.concatenate(arrs, axis=axis)
        else:
            data = ref.to_numpy()
        merged[path] = Variable(ref.schema, data)
    return Dataset(schema=schema, variables=merged, attrs=parts[0].attrs)
