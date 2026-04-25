"""Phase 4 — partition catalog, _immutable group, views, parquet indexer."""

import json

import numpy
import pytest

import zcollection as zc
from zcollection.indexing import Indexer
from zcollection.partitioning import Catalog
from zcollection.partitioning.catalog import CATALOG_FILE, _checksum
from zcollection.store.layout import IMMUTABLE_DIR
from zcollection.view import View, ViewReference

# --- catalog --------------------------------------------------------


def test_catalog_round_trip(tmp_path):
    store = zc.LocalStore(tmp_path / "col")
    cat = Catalog(store)
    assert cat.read() is None

    cat.write(["b", "a", "c"])
    state = cat.read()
    assert state is not None
    assert state.paths == ("a", "b", "c")
    assert state.checksum == _checksum(["a", "b", "c"])
    assert state.matches(["c", "a", "b"])
    assert not state.matches(["a", "b"])


def test_catalog_add_and_remove(tmp_path):
    store = zc.LocalStore(tmp_path / "col")
    cat = Catalog(store)
    cat.add(["p1", "p2"])
    cat.add(["p1", "p3"])
    assert cat.read_paths() == ["p1", "p2", "p3"]
    cat.remove(["p2"])
    assert cat.read_paths() == ["p1", "p3"]


def test_catalog_drop(tmp_path):
    store = zc.LocalStore(tmp_path / "col")
    cat = Catalog(store)
    cat.write(["a"])
    assert cat.exists()
    cat.drop()
    assert not cat.exists()


def test_collection_uses_catalog_when_enabled(
    tmp_path, schema, dataset, partitioning
):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        catalog_enabled=True,
        overwrite=True,
    )
    col.insert(dataset)
    cat = Catalog(store)
    paths = cat.read_paths()
    assert paths == ["num=0", "num=1", "num=2"]
    # partitions() must reuse the catalog (verified indirectly: same listing
    # after we corrupt the catalog).
    cat.write(["num=0"])
    reopened = zc.open_collection(store, mode="r")
    assert list(reopened.partitions()) == ["num=0"]


def test_repair_catalog_recovers_from_crash(
    tmp_path,
    schema,
    dataset,
    partitioning,
    monkeypatch,
):
    """Simulate a crash between partition-writes and catalog-update.

    The first ``insert`` raises *after* writing partitions but before the
    catalog is updated. On reopen, the stale catalog still says zero
    partitions; ``repair_catalog`` reconciles by walking the store.
    """
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        catalog_enabled=True,
        overwrite=True,
    )
    Catalog(store).write([])  # baseline empty catalog

    # Patch Catalog.add to simulate a crash before the catalog write lands.
    real_add = Catalog.add

    def _crashing_add(self, paths):
        raise RuntimeError("simulated crash before catalog update")

    monkeypatch.setattr(Catalog, "add", _crashing_add)
    with pytest.raises(RuntimeError, match="simulated crash"):
        col.insert(dataset)

    # Restore the real method.
    monkeypatch.setattr(Catalog, "add", real_add)

    # Stale catalog: empty list, but partitions exist on disk.
    reopened = zc.open_collection(store, mode="rw")
    assert list(reopened.partitions()) == []  # catalog is authoritative
    walked = reopened.repair_catalog()
    assert walked == ["num=0", "num=1", "num=2"]
    assert list(reopened.partitions()) == walked


def test_catalog_drop_partitions_updates_catalog(
    tmp_path,
    schema,
    dataset,
    partitioning,
):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        catalog_enabled=True,
        overwrite=True,
    )
    col.insert(dataset)
    col.drop_partitions(filters="num == 1")
    assert Catalog(store).read_paths() == ["num=0", "num=2"]


def test_catalog_corrupted_payload_treated_as_missing(tmp_path):
    store = zc.LocalStore(tmp_path / "col")
    store.write_bytes(CATALOG_FILE, b"not-json")
    assert Catalog(store).read() is None
    assert Catalog(store).read_paths() is None


# --- _immutable/ group ---------------------------------------------


def test_immutable_variable_persists_at_root(
    tmp_path,
    schema,
    dataset,
    partitioning,
):
    """The ``static`` variable (dim x only) lives at ``_immutable/static``."""
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)
    # The immutable group exists at root.
    assert (tmp_path / "col" / IMMUTABLE_DIR / "static" / "zarr.json").exists()
    # Per-partition groups must NOT contain the immutable variable.
    assert not (tmp_path / "col" / "num=0" / "static" / "zarr.json").exists()


def test_immutable_variable_returned_by_query(
    tmp_path,
    schema,
    dataset,
    partitioning,
):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)
    out = zc.open_collection(store, mode="r").query()
    assert "static" in out.variables
    assert numpy.array_equal(
        out["static"].to_numpy(),
        dataset["static"].to_numpy(),
    )
    # And mutable variables still merge correctly.
    assert numpy.array_equal(
        sorted(out["num"].to_numpy()),
        sorted(dataset["num"].to_numpy()),
    )


# --- view -----------------------------------------------------------


def test_view_create_query_update(tmp_path, schema, dataset, partitioning):
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    base.insert(dataset)

    view_store = zc.LocalStore(tmp_path / "view")
    derived = zc.VariableSchema(
        name="value_squared",
        dtype=numpy.dtype("float32"),
        dimensions=("num", "x"),
    )
    view = View.create(
        view_store,
        base=base,
        variables=[derived],
        reference=ViewReference(uri=f"file://{tmp_path / 'col'}"),
    )

    def _square(ds: zc.Dataset) -> dict[str, numpy.ndarray]:
        return {"value_squared": ds["value"].to_numpy() ** 2}

    written = view.update(_square)
    assert sorted(written) == ["num=0", "num=1", "num=2"]

    out = view.query()
    assert "value_squared" in out.variables
    assert "value" in out.variables  # base var still present
    assert numpy.allclose(
        out["value_squared"].to_numpy(),
        out["value"].to_numpy() ** 2,
    )


def test_view_persists_and_reopens(tmp_path, schema, dataset, partitioning):
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    base.insert(dataset)
    view_store = zc.LocalStore(tmp_path / "view")
    derived = zc.VariableSchema(
        name="bias",
        dtype=numpy.dtype("float32"),
        dimensions=("num", "x"),
    )
    View.create(
        view_store,
        base=base,
        variables=[derived],
        reference=f"file://{tmp_path / 'col'}",
    )
    reopened = View.open(view_store, base=base)
    assert reopened.variables == ("bias",)
    assert reopened.reference.uri.endswith("col")


def test_view_rejects_colliding_variable(
    tmp_path, schema, dataset, partitioning
):
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    view_store = zc.LocalStore(tmp_path / "view")
    colliding = zc.VariableSchema(
        name="value",
        dtype=numpy.dtype("float32"),
        dimensions=("num", "x"),
    )
    with pytest.raises(zc.ZCollectionError, match="collides"):
        View.create(
            view_store,
            base=base,
            variables=[colliding],
            reference=f"file://{tmp_path / 'col'}",
        )


# --- indexer --------------------------------------------------------


def test_indexer_build_and_lookup(tmp_path, schema, dataset, partitioning):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    def _build(ds: zc.Dataset) -> dict[str, numpy.ndarray]:
        nums = ds["num"].to_numpy()
        # one row covering [0, len) for this partition's only key.
        return {
            "num": numpy.array([int(nums[0])], dtype="int64"),
            "_start": numpy.array([0], dtype="int64"),
            "_stop": numpy.array([nums.size], dtype="int64"),
        }

    idx = Indexer.build(col, builder=_build)
    assert len(idx) == 3
    assert "num" in idx.key_columns

    # Persist round-trip.
    out_path = tmp_path / "idx.parquet"
    idx.write(str(out_path))
    reloaded = Indexer.read(str(out_path))
    assert len(reloaded) == 3

    hits = reloaded.lookup(num=2)
    assert list(hits.keys()) == ["num=2"]
    assert hits["num=2"] == [(0, 3)]

    multi = reloaded.lookup(num=[0, 2])
    assert set(multi.keys()) == {"num=0", "num=2"}


def test_indexer_unknown_column_raises(tmp_path, schema, dataset, partitioning):
    store = zc.LocalStore(tmp_path / "col")
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    def _build(ds: zc.Dataset) -> dict[str, numpy.ndarray]:
        return {
            "num": numpy.array([int(ds["num"].to_numpy()[0])], dtype="int64"),
            "_start": numpy.array([0], dtype="int64"),
            "_stop": numpy.array([ds["num"].to_numpy().size], dtype="int64"),
        }

    idx = Indexer.build(col, builder=_build)
    with pytest.raises(KeyError, match="unknown index columns"):
        idx.lookup(other=1)


def test_view_config_is_json(tmp_path, schema, dataset, partitioning):
    """Sanity: the view config is well-formed JSON we can hand-decode."""
    base_store = zc.LocalStore(tmp_path / "col")
    base = zc.create_collection(
        base_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    view_store = zc.LocalStore(tmp_path / "view")
    derived = zc.VariableSchema(
        name="d",
        dtype=numpy.dtype("float32"),
        dimensions=("num",),
    )
    View.create(
        view_store,
        base=base,
        variables=[derived],
        reference=f"file://{tmp_path / 'col'}",
    )
    raw = (tmp_path / "view" / "_zcollection_view.json").read_bytes()
    payload = json.loads(raw)
    assert payload["format_version"] == 1
    assert payload["reference"]["uri"].endswith("col")
