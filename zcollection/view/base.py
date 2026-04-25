"""Slim, v3-native View implementation."""

from typing import TYPE_CHECKING, Any
import asyncio
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import json

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
        """Return the reference as a JSON-serialisable dictionary."""
        return {"uri": self.uri}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ViewReference:
        """Build a ``ViewReference`` from its JSON payload."""
        return cls(uri=str(payload["uri"]))


class View:
    """Overlay of extra variables on top of a base :class:`~zcollection.collection.base.Collection`."""

    def __init__(
        self,
        *,
        store: Store,
        base: Collection,
        view_schema: DatasetSchema,
        reference: ViewReference,
        read_only: bool = False,
    ) -> None:
        """Initialize a View.

        Args:
            store: Backing store for the view's overlay variables.
            base: Underlying base collection.
            view_schema: Schema describing the overlay variables.
            reference: Pointer to the base collection.
            read_only: Whether the view should refuse mutations.

        """
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
    ) -> View:
        """Create a new view backed by ``store`` and overlaying ``base``.

        Args:
            store: Backing store for the view's overlay variables. Must
                be different from ``base.store``.
            base: The underlying read-only base collection.
            variables: Schemas for the *new* variables the view adds.
                Each must share at least the partitioning dimension with
                the base collection.
            reference: Either a :class:`ViewReference` or a string URI
                identifying the base collection.
            overwrite: If ``True``, replace any existing view at this
                location.

        Returns:
            A writable :class:`~zcollection.view.base.View` ready to ``update``.

        Raises:
            ~zcollection.errors.CollectionExistsError: If a view already exists at
                ``store.root_uri`` and ``overwrite=False``.
            ZCollectionError: If a view variable's dimensions are
                inconsistent with the base schema.

        """
        if store.exists(VIEW_CONFIG_FILE) and not overwrite:
            raise CollectionExistsError(
                f"a view already exists at {store.root_uri}",
            )
        ref = (
            reference
            if isinstance(reference, ViewReference)
            else ViewReference(uri=reference)
        )
        view_vars = {
            v.name: _ensure_view_variable(v, base.schema) for v in variables
        }
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
            store=store,
            base=base,
            view_schema=view_schema,
            reference=ref,
            read_only=False,
        )

    @classmethod
    def open(
        cls,
        store: Store,
        *,
        base: Collection,
        read_only: bool = False,
    ) -> View:
        """Open an existing view from ``store``.

        Args:
            store: The store backing the view's overlay variables.
            base: The base collection that this view extends. The caller
                is responsible for ensuring it matches ``reference``.
            read_only: If ``True``, mutating methods (``update``) raise
                :class:`~zcollection.errors.ReadOnlyError`.

        Returns:
            A :class:`~zcollection.view.base.View` bound to the existing overlay.

        Raises:
            ~zcollection.errors.CollectionNotFoundError: If no view config exists at
                ``store.root_uri``.

        """
        raw = store.read_bytes(VIEW_CONFIG_FILE)
        if raw is None:
            raise CollectionNotFoundError(
                f"no view config at {store.root_uri}",
            )
        payload = json.loads(raw.decode("utf-8"))
        ref = ViewReference.from_json(payload["reference"])
        view_schema = DatasetSchema.from_json(payload["schema"])
        return cls(
            store=store,
            base=base,
            view_schema=view_schema,
            reference=ref,
            read_only=read_only,
        )

    # --- properties -------------------------------------------------

    @property
    def store(self) -> Store:
        """Return the backing store for the view overlay."""
        return self._store

    @property
    def base(self) -> Collection:
        """Return the underlying base collection."""
        return self._base

    @property
    def view_schema(self) -> DatasetSchema:
        """Return the view's overlay schema."""
        return self._view_schema

    @property
    def reference(self) -> ViewReference:
        """Return the reference to the base collection."""
        return self._reference

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the names of the view's overlay variables."""
        return tuple(self._view_schema.variables)

    @property
    def read_only(self) -> bool:
        """Return whether the view is read-only."""
        return self._read_only

    # --- listing ----------------------------------------------------

    def partitions(self, *, filters: str | None = None) -> Iterator[str]:
        """Yield partition paths from the base collection, optionally filtered."""
        return self._base.partitions(filters=filters)

    # --- query ------------------------------------------------------

    def query(
        self,
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Dataset | None:
        """Return the merged base+overlay dataset for matching partitions.

        Variables are sourced from whichever side owns them. On a name
        collision the overlay (view) wins.

        Args:
            filters: Partition-key predicate forwarded to the base
                collection's :meth:`~Collection.partitions`.
            variables: Optional whitelist mixing base and overlay names.
                ``None`` returns all base + overlay variables.

        Returns:
            The merged :class:`~zcollection.Dataset`, or ``None`` if no
            base partition matched ``filters``. If matching partitions
            exist but no overlay has been written for them yet, only the
            base is returned.

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
        wanted = set(variables) if variables is not None else None
        view_names = set(self._view_schema.variables)
        base_names = set(self._base.schema.variables)
        base_wanted = (
            None
            if wanted is None
            else sorted((wanted & base_names) | (wanted - view_names))
        )
        view_wanted = None if wanted is None else sorted(wanted & view_names)

        base_ds = await self._base.query_async(
            filters=filters,
            variables=base_wanted,
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
                    self._store,
                    path,
                    self._view_schema,
                    variables=view_wanted,
                )

        loaded = [
            d
            for d in await asyncio.gather(*[_load(p) for p in parts])
            if d is not None
        ]
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

        ``fn`` runs once per matching base partition. It receives the
        merged base+view dataset and must return a mapping from
        view-variable name to a numpy array sized along the partitioning
        dimension. Returned keys must be a subset of
        :attr:`view_schema`'s variables; missing keys are ignored,
        unknown keys raise.

        Args:
            fn: Pure function ``Dataset -> {name: numpy.ndarray}``.
            filters: Partition-key predicate (same syntax as
                :meth:`Collection.partitions`).
            variables: Optional whitelist of variables to load before
                calling ``fn``. ``None`` loads everything available.

        Returns:
            The list of partition paths that were written, in the order
            they were processed.

        Raises:
            ~zcollection.errors.ReadOnlyError: If the view was opened with
                ``read_only=True``.

        """
        from ..dask.scheduler import run_sync

        return run_sync(
            self.update_async(fn, filters=filters, variables=variables)
        )

    async def update_async(
        self,
        fn: Callable[[Dataset], dict[str, Any]],
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> list[str]:
        """Async variant of :meth:`update`."""
        self._require_writable()
        wanted = set(variables) if variables is not None else None
        view_names = set(self._view_schema.variables)
        base_names = set(self._base.schema.variables)
        base_wanted = (
            None
            if wanted is None
            else sorted((wanted & base_names) | (wanted - view_names))
        )

        parts = list(self.partitions(filters=filters))
        concurrency = max(1, int(config_get("partition.concurrency")))
        sem = asyncio.Semaphore(concurrency)

        async def _step(path: str) -> str:
            async with sem:
                base_ds = await open_partition_dataset_async(
                    self._base.store,
                    path,
                    self._base.schema,
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
                    schema=self._view_schema,
                    variables=view_vars,
                )
                await write_partition_dataset_async(
                    self._store,
                    path,
                    ds,
                    overwrite=True,
                    concurrency=concurrency,
                )
                return path

        return list(await asyncio.gather(*[_step(p) for p in parts]))

    # --- internal ---------------------------------------------------

    def _require_writable(self) -> None:
        if self._read_only:
            raise ReadOnlyError(f"view at {self._store.root_uri} is read-only")


# --- helpers --------------------------------------------------------


def _ensure_view_variable(
    var: VariableSchema,
    base_schema: DatasetSchema,
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
    import numpy

    if len(parts) == 1:
        return parts[0]
    schema = parts[0].schema
    out: dict[str, Variable] = {}
    for name in parts[0].variables:
        ref = parts[0][name]
        if dim in ref.dimensions:
            axis = ref.dimensions.index(dim)
            data = numpy.concatenate(
                [p[name].to_numpy() for p in parts],
                axis=axis,
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
