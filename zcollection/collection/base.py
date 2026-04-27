# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""High-level :class:`Collection` facade.

A :class:`Collection` is the project's central abstraction: a dataset
split along **one** unbounded *partition axis* and persisted as a tree
of Zarr v3 groups under a :class:`~zcollection.store.Store`. Every
public mutation (``insert``, ``update``, ``drop_partitions``) and every
public read (``query``, ``map``, ``partitions``) lives on this class.

Method-pairing convention: each blocking method has an ``_async``
sibling with the same signature and identical semantics; the blocking
form is a thin :func:`run_sync` wrapper around the async one. Use the
sync API by default; switch to ``_async`` when you are already inside
an event loop.

Two on-disk concepts are worth knowing about:

- The **immutable group** (``_immutable/`` at the collection root) holds
  variables flagged ``immutable`` in the schema — variables whose
  dimensions don't include the partition axis. They are written once
  at the root and merged back into the dataset returned by every read.
- The optional **catalog** (``_catalog/state.json``) is a single JSON
  document listing the partition paths in sorted order, used to skip
  the O(N) directory walk on cold opens. Built at insert time when
  ``catalog_enabled=True``.

The user-facing entry points are :func:`zcollection.create_collection`
and :func:`zcollection.open_collection`; this module's
:meth:`Collection.create` / :meth:`Collection.open` classmethods are
their direct backers.
"""

from typing import TYPE_CHECKING, Any
import asyncio
from collections.abc import Callable, Iterable, Iterator

import numpy

from ..config import get as config_get
from ..data import Dataset, Variable
from ..errors import CollectionExistsError, PartitionError, ReadOnlyError
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
from .merge import resolve as resolve_merge_strategy


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
    """A partitioned Zarr v3 collection on a :class:`~zcollection.store.Store`.

    A :class:`Collection` materialises a :class:`DatasetSchema` over a
    set of partitions on disk. Each partition is a self-contained Zarr
    v3 group whose key in the partitioning dimension is encoded into
    its path (e.g. ``year=2024/month=03``). Variables that don't span
    the partition axis are flagged *immutable*, written once under
    ``_immutable/``, and merged back into every read.

    Typical lifecycle:

    1. :func:`zcollection.create_collection` (or
       :meth:`Collection.create`) writes the root config and returns a
       writable handle.
    2. :meth:`insert` (and any merge strategy) populates partitions.
    3. :func:`zcollection.open_collection` (or :meth:`Collection.open`)
       reopens an existing root, optionally read-only.
    4. :meth:`query`, :meth:`map`, :meth:`update`, and
       :meth:`drop_partitions` operate on subsets selected by partition
       filters; each has an ``_async`` sibling with identical
       semantics.

    Direct use of :meth:`__init__` is reserved for the library; user
    code should always go through :meth:`create` / :meth:`open` or the
    factory functions in :mod:`zcollection.api`.
    """

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
        """Bind already-validated metadata to a backing store.

        Internal constructor. Prefer :meth:`Collection.create` /
        :meth:`Collection.open` (or the factories
        :func:`zcollection.create_collection` /
        :func:`zcollection.open_collection`) — they are responsible for
        rebinding the schema to the partition axis and for reading the
        on-disk root config. This constructor performs no validation
        beyond what the dataclasses themselves enforce.

        Args:
            store: The backing :class:`~zcollection.store.Store`.
            schema: A schema already passed through
                :meth:`~zcollection.DatasetSchema.with_partition_axis`
                — every variable's ``immutable`` flag is set
                accordingly.
            axis: Name of the partition axis (a dimension of
                ``schema``).
            partitioning: Partitioning strategy (already constructed).
            catalog_enabled: When ``True``, the partition catalog is
                read and maintained on every mutation.
            read_only: When ``True``, mutating methods raise
                :class:`~zcollection.errors.ReadOnlyError`.

        """
        # The backing store for the collection.
        self._store = store
        # The schema, with immutable flags set per the partition axis.
        self._schema = schema
        # Name of the partition axis (also a dimension of the schema).
        self._axis = axis
        # Partitioning strategy bound to this collection.
        self._partitioning = partitioning
        # Whether to maintain the optional partition catalog.
        self._catalog_enabled = catalog_enabled
        # Whether mutating calls should refuse and raise.
        self._read_only = read_only
        # The catalog reader/writer, lazily None when disabled.
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
        immediately. Variables that don't span ``axis`` are flagged
        immutable and lifted into the ``_immutable/`` group at first
        insert.

        Args:
            store: The backing :class:`~zcollection.store.Store`.
            schema: The dataset schema. It is passed through
                :meth:`~zcollection.DatasetSchema.with_partition_axis`,
                which flags variables that don't span ``axis`` as
                immutable and rejects schemas that can't be soundly
                partitioned (e.g. an unbounded non-axis dimension).
            axis: Name of the partition axis (must be a root
                dimension of ``schema``).
            partitioning: Partitioning strategy (e.g.
                :class:`~zcollection.partitioning.Date`,
                :class:`~zcollection.partitioning.Sequence`,
                :class:`~zcollection.partitioning.GroupedSequence`).
            catalog_enabled: If ``True``, maintain a single
                ``_catalog/state.json`` document listing the partition
                paths so ``partitions()`` and cold opens skip the O(N)
                directory walk.
            overwrite: If ``True``, replace any existing collection
                root at this location. If ``False`` (default) and a
                root exists,
                :class:`~zcollection.errors.CollectionExistsError` is
                raised.

        Returns:
            The newly-created :class:`~zcollection.Collection`, ready
            to ``insert``.

        Raises:
            ~zcollection.errors.CollectionExistsError: If a collection
                already exists at ``store.root_uri`` and
                ``overwrite=False``.
            ~zcollection.errors.SchemaError: If the schema cannot be
                bound to ``axis`` (unknown axis, or any variable spans
                an unbounded dimension other than ``axis``).

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

        Reads the root config (axis, partitioning, schema, catalog
        flag) and returns a handle pointing at it.

        Args:
            store: The store backing the collection.
            read_only: If ``True``, mutating methods (:meth:`insert`,
                :meth:`drop_partitions`, :meth:`update`,
                :meth:`repair_catalog`) raise
                :class:`~zcollection.errors.ReadOnlyError` instead of
                writing.

        Returns:
            A :class:`~zcollection.Collection` bound to the existing
            root.

        Raises:
            ~zcollection.errors.CollectionNotFoundError: If no
                collection exists at ``store.root_uri``.

        """
        doc = read_root_config(store)
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
            filters: An optional partition-key expression. The
                expression language is the small typed subset
                described in
                :func:`~zcollection.partitioning.compile_filter`:
                comparisons, ``and``/``or``/``not``, ``in`` /
                ``not in``, integer/string literals, and partition-key
                names. Examples:
                ``"year == 2024 and month >= 3"``,
                ``"cycle in (1, 2)"``. Comparable types are whatever
                ``Partitioning.decode`` produces (integers and strings
                in the built-in partitionings). ``None`` yields every
                partition.

        Yields:
            Relative partition paths (e.g. ``"year=2024/month=03"``),
            sorted lexicographically.

        Raises:
            ~zcollection.errors.ExpressionError: If ``filters`` has
                invalid syntax or uses disallowed AST nodes (raised at
                compile time), or if it references a partition-key
                name that doesn't exist (raised the first time the
                predicate is evaluated against a partition).

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
        """Rebuild the catalog by walking the store; return the new path list.

        Use after a crash, or after manual edits to the partition tree.

        Returns:
            The freshly walked list of partition paths, in sorted
            order, after the catalog has been rewritten.

        Raises:
            RuntimeError: If the collection was not opened with
                ``catalog_enabled=True`` (no catalog to repair).
            ~zcollection.errors.ReadOnlyError: If the collection was
                opened read-only.

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

        The dataset is split by the collection's partitioning. Each
        slice is written under the matching partition path. Variables
        flagged immutable in the schema (those that don't span the
        partition axis) are written once under ``_immutable/`` and are
        merged into the dataset returned by every subsequent
        :meth:`query` / :meth:`map` / :meth:`update`.

        On a transactional store (e.g.
        :class:`~zcollection.store.IcechunkStore`) the entire insert is
        wrapped in a single commit — a crash mid-insert leaves no
        partial state.

        Args:
            dataset: The data to insert. Variables can be backed by
                numpy, Dask, or Zarr ``AsyncArray``.
            overwrite: When a partition already exists, whether to
                replace its chunks (``True``, default) or fail.
                **When `merge` is set, ``overwrite=True`` is
                effectively required**: the merged dataset must replace
                the existing partition, so passing ``overwrite=False``
                with a non-``None`` ``merge`` will raise once the
                writer hits the existing group.
            merge: Strategy for combining the inserted data with an
                existing partition. Either a built-in alias
                (``"replace"`` / ``"concat"`` / ``"time_series"`` /
                ``"upsert"``), a
                :class:`~zcollection.collection.merge.MergeCallable`,
                or ``None`` (default — the inserted slice is written
                as-is and replaces the existing partition's chunks).
                For tolerance-aware nearest-neighbour matching, build
                a callable with
                :func:`zcollection.collection.merge.upsert_within`
                — it isn't registered as a string alias because it
                needs the tolerance argument.

        Returns:
            The list of partition paths that were written, in the
            order they were produced.

        Raises:
            ~zcollection.errors.ReadOnlyError: If the collection was
                opened with ``read_only=True``.
            KeyError: If ``merge`` is a string that doesn't match a
                built-in strategy.
            ~zcollection.errors.PartitionError: If ``dataset`` is
                missing variables required by the partitioning.

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
        merge_fn = resolve_merge_strategy(merge)
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
        ``zcollection.config["partition.concurrency"]``. The variables
        flagged ``immutable`` in the schema (read once from
        ``_immutable/``) are merged into the result; on name conflict
        the partition's data wins.

        Args:
            filters: A partition-key predicate; see :meth:`partitions`.
                ``None`` reads every partition.
            variables: Iterable of variable names to load. ``None``
                (default) loads every variable. Loading a subset is
                the primary way to keep cold S3 reads cheap.

        Returns:
            The concatenated :class:`~zcollection.Dataset` along the
            partitioning dimension, or ``None`` if no partition
            matched ``filters``.

        Raises:
            ~zcollection.errors.ExpressionError: If ``filters`` is
                malformed; see :meth:`partitions`.

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

        Wrapped in a store session so transactional backends commit
        the whole drop atomically.

        Unlike :meth:`insert`, :meth:`query`, :meth:`map` and
        :meth:`update`, this method has no ``_async`` sibling: deletes
        are sequential by design (the store-session commit is one
        transaction).

        Args:
            filters: A partition-key predicate; see :meth:`partitions`.
                If ``None``, **every** partition is dropped — pass an
                explicit filter when you don't mean that.

        Returns:
            The list of partition paths that were removed, in the
            order they were processed.

        Raises:
            ~zcollection.errors.ReadOnlyError: If the collection was
                opened with ``read_only=True``.
            ~zcollection.errors.ExpressionError: If ``filters`` is
                malformed; see :meth:`partitions`.

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

        ``fn`` is called per partition with the dataset returned by
        :meth:`query` (immutable variables merged in). It must return a
        :class:`~zcollection.Dataset` carrying every variable the
        partition should keep — the call **rewrites the partition
        wholesale** with whatever ``fn`` returns.

        .. warning::

           ``update`` is *not* a partial-write API. Each partition
           group is recreated with ``overwrite=True`` before the new
           dataset is written, so any variable absent from
           ``fn``'s return value is **dropped from disk**. This holds
           regardless of ``variables``: if you load only a subset and
           return only a subset, the unloaded variables are *also*
           lost. To update one variable without disturbing the
           others, return a Dataset that still carries the rest
           (e.g. start from the input ``ds`` and replace just the
           target variable).

           ``fn`` should also preserve the partition's length along
           the partitioning dimension; ``update`` does not refresh
           the catalog from per-partition geometry.

        Args:
            fn: Function ``Dataset -> Dataset`` applied to each
                matching partition.
            filters: Partition-key predicate; see :meth:`partitions`.
            variables: Optional whitelist of variables to load before
                calling ``fn``. Reduces I/O at read time, but does
                *not* protect unloaded variables from being dropped on
                write — see the warning above.

        Returns:
            The list of partition paths that were written, in the
            order they were processed.

        Raises:
            ~zcollection.errors.ReadOnlyError: If the collection was
                opened with ``read_only=True``.
            ~zcollection.errors.ExpressionError: If ``filters`` is
                malformed; see :meth:`partitions`.

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
    """Slice every variable (root or nested) that spans ``dim`` by ``sl``.

    Variables whose dimensions don't include ``dim`` are passed through
    unchanged (they are static across partitions by the
    partitioned-or-immutable contract enforced at schema bind time).
    """
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
    """Concatenate a list of datasets along ``dim``, recursing into groups.

    Preconditions: ``parts`` must be non-empty and every dataset must
    expose the same schema and the same set of variable paths. A list
    of length 1 is returned unchanged. For variables that don't span
    ``dim``, the first partition's value is reused (the static /
    immutable-by-construction case).
    """
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
