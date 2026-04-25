"""High-level Collection facade — Phase 1 sync surface."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator

import numpy

from ..data import Dataset, Variable
from ..errors import (
    CollectionExistsError,
    CollectionNotFoundError,
    PartitionError,
    ReadOnlyError,
)
from ..io import (
    open_partition_dataset,
    partition_exists,
    read_root_config,
    write_partition_dataset,
    write_root_config,
)
from ..partitioning import (
    compile_filter,
    from_json as partitioning_from_json,
    key_to_dict,
)
from ..schema import DatasetSchema
from ..schema.serde import CONFIG_FILE
from ..store.layout import CATALOG_DIR, IMMUTABLE_DIR, join_path

if TYPE_CHECKING:
    from ..partitioning import Partitioning
    from ..store import Store


_RESERVED_TOP_LEVEL = {CATALOG_DIR, IMMUTABLE_DIR, CONFIG_FILE, "zarr.json"}


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
    ) -> "Collection":
        if store.exists(CONFIG_FILE) and not overwrite:
            raise CollectionExistsError(
                f"a collection already exists at {store.root_uri}"
            )
        bound = schema.with_partition_axis(axis)
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
    def open(cls, store: Store, *, read_only: bool = False) -> "Collection":
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
        depth = len(self._partitioning.axis)
        predicate = compile_filter(filters)
        for path in sorted(self._walk(prefix="", depth=depth)):
            try:
                key = self._partitioning.decode(path)
            except PartitionError:
                continue
            if predicate(key_to_dict(key)):
                yield path

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

    def insert(self, dataset: Dataset, *, overwrite: bool = True) -> list[str]:
        self._require_writable()
        written: list[str] = []
        for key, sl in self._partitioning.split(dataset):
            sub = _slice_dataset(dataset, dim=self._partitioning.dimension, sl=sl)
            path = self._partitioning.encode(key)
            extra = {"_zc_partition_key": [list(t) for t in key]}
            write_partition_dataset(
                self._store, path, sub, overwrite=overwrite, extra_attrs=extra
            )
            written.append(path)
        return written

    # --- Query -----------------------------------------------------

    def query(
        self,
        *,
        filters: str | None = None,
        variables: Iterable[str] | None = None,
    ) -> Dataset | None:

        wanted = list(variables) if variables is not None else None
        parts = list(self.partitions(filters=filters))
        if not parts:
            return None
        loaded = [
            open_partition_dataset(self._store, p, self._schema, variables=wanted)
            for p in parts
        ]
        return _concat_datasets(loaded, dim=self._partitioning.dimension)

    # --- Drop ------------------------------------------------------

    def drop_partitions(self, *, filters: str | None = None) -> list[str]:
        self._require_writable()
        dropped: list[str] = []
        for path in list(self.partitions(filters=filters)):
            self._store.delete_prefix(path)
            dropped.append(path)
        return dropped

    # --- Internal --------------------------------------------------

    def _require_writable(self) -> None:
        if self._read_only:
            raise ReadOnlyError(f"collection at {self._store.root_uri} is read-only")


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
    return Dataset(schema=dataset.schema, variables=new_vars, attrs=dataset.attrs)


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
