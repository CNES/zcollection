"""IcechunkStore: round-trip + transactional commit/rollback."""

import numpy
import pytest

import zcollection as zc
from zcollection.collection import base as _cbase
from zcollection.store import open_store


icechunk = pytest.importorskip("icechunk")
from zcollection.store.icechunk_store import IcechunkStore  # noqa: E402


@pytest.fixture
def ic_store(tmp_path):
    """Provide a fresh IcechunkStore backed by a temp repo."""
    return IcechunkStore(str(tmp_path / "repo"))


def test_create_open_round_trip(ic_store, schema, dataset, partitioning):
    """IcechunkStore round-trips create/insert/query with bit-equal data."""
    col = zc.create_collection(
        ic_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    reopened = zc.open_collection(ic_store, mode="r")
    out = reopened.query()
    assert out is not None
    numpy.testing.assert_array_equal(
        out["value"].to_numpy(),
        dataset["value"].to_numpy(),
    )
    numpy.testing.assert_array_equal(
        out["static"].to_numpy(),
        dataset["static"].to_numpy(),
    )


def test_drop_partitions_commits(ic_store, schema, dataset, partitioning):
    """``drop_partitions`` on Icechunk commits the deletion."""
    col = zc.create_collection(
        ic_store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)
    assert sorted(col.partitions()) == ["num=0", "num=1", "num=2"]
    col.drop_partitions(filters="num == 1")
    assert sorted(col.partitions()) == ["num=0", "num=2"]


def test_failed_insert_rolls_back(
    tmp_path,
    schema,
    dataset,
    partitioning,
    monkeypatch,
):
    """SIGKILL-equivalent: any exception inside insert_async must leave the repo in its prior committed state.

    Not a single partition persisted.
    """
    store = IcechunkStore(str(tmp_path / "repo"))
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )

    # Make a mid-insert partition write blow up to simulate a crash.
    # Patch the binding inside collection.base where insert_async resolved
    # the import.
    cbase = _cbase
    real_write = cbase.write_partition_dataset_async
    calls: list[str] = []

    async def _exploding(store_, path, ds, **kw):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("simulated mid-insert crash")
        return await real_write(store_, path, ds, **kw)

    monkeypatch.setattr(cbase, "write_partition_dataset_async", _exploding)

    with pytest.raises(RuntimeError, match="simulated mid-insert crash"):
        col.insert(dataset)

    # Reopen via a fresh IcechunkStore so we read only committed state.
    fresh = IcechunkStore(str(tmp_path / "repo"))
    reopened = zc.open_collection(fresh, mode="r")
    assert list(reopened.partitions()) == [], (
        "transaction rollback should have erased every partition write"
    )


def test_successful_insert_persists_after_reopen(
    tmp_path,
    schema,
    dataset,
    partitioning,
):
    """A successful insert is visible after reopening the Icechunk repo."""
    store = IcechunkStore(str(tmp_path / "repo"))
    col = zc.create_collection(
        store,
        schema=schema,
        axis="num",
        partitioning=partitioning,
        overwrite=True,
    )
    col.insert(dataset)

    fresh = IcechunkStore(str(tmp_path / "repo"))
    reopened = zc.open_collection(fresh, mode="r")
    assert sorted(reopened.partitions()) == ["num=0", "num=1", "num=2"]


def test_factory_routes_icechunk_url(tmp_path):
    """``open_store`` routes ``icechunk://`` URLs to IcechunkStore."""
    s = open_store(f"icechunk://{tmp_path / 'repo'}")
    assert isinstance(s, IcechunkStore)
