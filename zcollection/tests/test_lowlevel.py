"""Targeted tests for the small support modules and primitives.

Covers ``config``, ``schema.attribute``, ``store.layout``,
``partitioning.expression``, ``partitioning.{Sequence,Date,GroupedSequence}``
JSON / encode / decode round-trips, and the in-memory + local stores.
These pieces are easy to test in isolation but fall through the cracks
of the partition-roundtrip suite.
"""

import asyncio
import json

import numpy
import pytest

import zcollection as zc
from zcollection import config as zc_config
from zcollection.errors import ExpressionError, PartitionError, StoreError
from zcollection.partitioning import (
    Date,
    GroupedSequence,
    Sequence,
    compile_filter,
    from_json as partitioning_from_json,
    key_to_dict,
)
from zcollection.partitioning.base import keys_from_columns, runs_from_inverse
from zcollection.schema.attribute import encode_attrs, encode_value
from zcollection.store import LocalStore, MemoryStore, open_store
from zcollection.store.layout import join_path, parent_path, relative_path


# %%
# zcollection.config -------------------------------------------------------


def test_config_get_known_key_returns_default() -> None:
    """``get`` returns the registered default."""
    assert zc_config.get("partition.concurrency") == 8
    assert zc_config.get("codec.profile") == "cloud-balanced"


def test_config_set_unknown_key_raises() -> None:
    """``set`` rejects unknown keys with a clear message."""
    with pytest.raises(KeyError, match="unknown"):
        zc_config.set(does_not_exist=1)


def test_config_override_restores_previous_value() -> None:
    """``override`` restores the prior value on context exit."""
    before = zc_config.get("partition.concurrency")
    with zc_config.override(**{"partition.concurrency": 1}):
        assert zc_config.get("partition.concurrency") == 1
    assert zc_config.get("partition.concurrency") == before


def test_config_override_restores_on_exception() -> None:
    """The override context unwinds even when the body raises."""
    before = zc_config.get("partition.concurrency")
    with (
        pytest.raises(RuntimeError),
        zc_config.override(**{"partition.concurrency": 99}),
    ):
        raise RuntimeError("boom")
    assert zc_config.get("partition.concurrency") == before


def test_config_set_then_reset() -> None:
    """``set`` updates the runtime value; restore manually."""
    before = zc_config.get("catalog.enabled")
    try:
        zc_config.set(**{"catalog.enabled": False})
        assert zc_config.get("catalog.enabled") is False
    finally:
        zc_config.set(**{"catalog.enabled": before})


def test_configure_zarr_no_arg_is_a_noop() -> None:
    """Calling ``configure_zarr`` without args should not blow up."""
    zc_config.configure_zarr()
    zc_config.configure_zarr(async_concurrency=4)


# %%
# schema.attribute.encode_value -------------------------------------------


def test_encode_value_passes_native_through() -> None:
    """Plain Python scalars are returned unchanged."""
    assert encode_value(1) == 1
    assert encode_value("hello") == "hello"
    assert encode_value(True) is True
    assert encode_value(None) is None


def test_encode_value_converts_numpy_scalar() -> None:
    """Numpy generics are unwrapped to Python scalars (JSON-clean)."""
    assert encode_value(numpy.int32(7)) == 7
    assert isinstance(encode_value(numpy.float64(1.5)), float)


def test_encode_value_lists_arrays_and_dicts() -> None:
    """Lists / tuples / arrays / dicts are recursively encoded."""
    arr = numpy.array([1, 2, 3], dtype="int32")
    assert encode_value(arr) == [1, 2, 3]
    assert encode_value((numpy.int8(1), 2)) == [1, 2]
    assert encode_value({"x": numpy.float32(1.0), 1: "v"}) == {
        "x": 1.0,
        "1": "v",
    }


def test_encode_attrs_handles_none_and_dict() -> None:
    """``encode_attrs`` returns ``{}`` for ``None`` and recursively encodes."""
    assert encode_attrs(None) == {}
    assert encode_attrs({}) == {}
    out = encode_attrs({"a": numpy.int64(1), "b": numpy.array([1, 2])})
    assert out == {"a": 1, "b": [1, 2]}
    # JSON-clean: must be serializable.
    json.dumps(out)


# %%
# store.layout helpers ----------------------------------------------------


def test_join_path_strips_and_skips_empty() -> None:
    """``join_path`` strips slashes and skips empty parts."""
    assert join_path("a", "b", "c") == "a/b/c"
    assert join_path("/a/", "/b/", "c/") == "a/b/c"
    assert join_path("", "a", None or "") == "a"
    assert join_path() == ""


def test_parent_path_root_and_nested() -> None:
    """``parent_path`` returns the parent or empty string at the root."""
    assert parent_path("a/b/c") == "a/b"
    assert parent_path("a") == ""
    assert parent_path("") == ""
    assert parent_path("a/b/c/") == "a/b"


def test_relative_path_against_base() -> None:
    """``relative_path`` strips ``base`` if ``path`` lives under it."""
    assert relative_path("a/b/c", "a") == "b/c"
    assert relative_path("a/b", "a/b") == ""
    assert relative_path("a/b", "x") == "a/b"
    assert relative_path("a/b", "") == "a/b"


# %%
# partitioning.compile_filter ---------------------------------------------


def test_filter_none_or_empty_is_tautology() -> None:
    """No expression → predicate that's always true."""
    assert compile_filter(None)({}) is True
    assert compile_filter("")({}) is True


@pytest.mark.parametrize(
    ("expr", "ctx", "expected"),
    [
        ("year == 2024", {"year": 2024}, True),
        ("year == 2024", {"year": 2023}, False),
        ("year >= 2020 and year <= 2024", {"year": 2022}, True),
        ("month in (1, 2, 3)", {"month": 2}, True),
        ("month not in (1, 2, 3)", {"month": 4}, True),
        ("not (year == 2024)", {"year": 2024}, False),
        ("year == 2020 or month == 12", {"year": 2024, "month": 12}, True),
        ("1 <= month <= 6", {"month": 5}, True),
        ("1 <= month <= 6", {"month": 7}, False),
    ],
)
def test_filter_truthy_paths(
    expr: str, ctx: dict[str, int], *, expected: bool
) -> None:
    """The compiled filter behaves like the equivalent Python expression."""
    assert compile_filter(expr)(ctx) is expected


def test_filter_rejects_invalid_syntax() -> None:
    """Bad Python at compile time raises ExpressionError."""
    with pytest.raises(ExpressionError, match="invalid filter"):
        compile_filter("year ==")


def test_filter_rejects_disallowed_node() -> None:
    """Function calls / arithmetic outside the whitelist are rejected."""
    with pytest.raises(ExpressionError, match="disallowed"):
        compile_filter("len(x) == 0")
    with pytest.raises(ExpressionError, match="disallowed"):
        compile_filter("year + 1 == 2024")


def test_filter_unknown_partition_key_raises_at_eval_time() -> None:
    """Compilation succeeds but evaluating with a missing key fails."""
    pred = compile_filter("missing == 1")
    with pytest.raises(ExpressionError, match="unknown partition key"):
        pred({})


def test_key_to_dict_round_trip() -> None:
    """``key_to_dict`` is the inverse of ``tuple(d.items())``."""
    key = (("year", 2024), ("month", 3))
    assert key_to_dict(key) == {"year": 2024, "month": 3}


# %%
# partitioning.base helpers ------------------------------------------------


def test_keys_from_columns_groups_by_unique_tuple() -> None:
    """``keys_from_columns`` returns one row per distinct tuple."""
    a = numpy.array([0, 0, 1, 1, 1, 2], dtype="int64")
    b = numpy.array([10, 10, 20, 20, 20, 30], dtype="int64")
    unique, inverse = keys_from_columns({"a": a, "b": b})
    assert unique.shape[0] == 3
    assert list(inverse) == [0, 0, 1, 1, 1, 2]


def test_runs_from_inverse_yields_contiguous_runs() -> None:
    """``runs_from_inverse`` walks contiguous runs of the same group id."""
    inverse = numpy.array([0, 0, 1, 1, 1, 2, 0])
    runs = list(runs_from_inverse(inverse))
    # 4 runs because the trailing 0 starts a new run.
    assert [(g, (s.start, s.stop)) for g, s in runs] == [
        (0, (0, 2)),
        (1, (2, 5)),
        (2, (5, 6)),
        (0, (6, 7)),
    ]


# %%
# partitioning.Sequence ---------------------------------------------------


@pytest.fixture
def seq_dataset() -> zc.Dataset:
    """Build a 1-D dataset suitable for Sequence and GroupedSequence."""
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=8)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("cycle", dtype="int64", dimensions=("time",))
        .with_variable("pass_", dtype="int64", dimensions=("time",))
        .build()
    )
    n = 8
    return zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(
                schema.variables["time"], numpy.arange(n, dtype="int64")
            ),
            "cycle": zc.Variable(
                schema.variables["cycle"],
                numpy.array([1, 1, 1, 2, 2, 2, 3, 3], dtype="int64"),
            ),
            "pass_": zc.Variable(
                schema.variables["pass_"],
                numpy.array([1, 1, 2, 1, 2, 2, 3, 3], dtype="int64"),
            ),
        },
    )


def test_sequence_split_yields_unique_runs(seq_dataset: zc.Dataset) -> None:
    """Each unique (cycle, pass) tuple yields one slice."""
    sp = Sequence(("cycle", "pass_"), dimension="time")
    keys = [(k, (s.start, s.stop)) for k, s in sp.split(seq_dataset)]
    assert keys == [
        ((("cycle", 1), ("pass_", 1)), (0, 2)),
        ((("cycle", 1), ("pass_", 2)), (2, 3)),
        ((("cycle", 2), ("pass_", 1)), (3, 4)),
        ((("cycle", 2), ("pass_", 2)), (4, 6)),
        ((("cycle", 3), ("pass_", 3)), (6, 8)),
    ]


def test_sequence_encode_decode_roundtrip() -> None:
    """``encode``/``decode`` are inverses for integer keys."""
    sp = Sequence(("cycle", "pass_"), dimension="time")
    key = (("cycle", 31), ("pass_", 100))
    path = sp.encode(key)
    assert path == "cycle=31/pass_=100"
    assert sp.decode(path) == key


def test_sequence_decode_rejects_bad_segment() -> None:
    """A segment without ``=`` raises PartitionError."""
    sp = Sequence(("x",), dimension="time")
    with pytest.raises(PartitionError, match="invalid partition path"):
        sp.decode("not_a_kv_segment")


def test_sequence_split_rejects_missing_variable(
    seq_dataset: zc.Dataset,
) -> None:
    """A required partition variable missing from the dataset raises."""
    sp = Sequence(("missing",), dimension="time")
    with pytest.raises(PartitionError, match="missing"):
        list(sp.split(seq_dataset))


def test_sequence_split_rejects_wrong_dim(seq_dataset: zc.Dataset) -> None:
    """A partition variable with the wrong dim raises."""
    sp = Sequence(("cycle",), dimension="other")
    with pytest.raises(PartitionError, match="must be 1-D"):
        list(sp.split(seq_dataset))


def test_sequence_split_rejects_non_integer(
    seq_dataset: zc.Dataset,
) -> None:
    """Sequence keys must be integer-typed."""
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=4)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("flag", dtype="float32", dimensions=("time",))
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(
                schema.variables["time"], numpy.arange(4, dtype="int64")
            ),
            "flag": zc.Variable(
                schema.variables["flag"],
                numpy.zeros(4, dtype="float32"),
            ),
        },
    )
    sp = Sequence(("flag",), dimension="time")
    with pytest.raises(PartitionError, match="integer"):
        list(sp.split(ds))


def test_sequence_requires_at_least_one_variable() -> None:
    """``Sequence(())`` is rejected at construction time."""
    with pytest.raises(PartitionError):
        Sequence((), dimension="time")


def test_sequence_to_from_json_roundtrip() -> None:
    """JSON round-trip preserves both the variables and the dimension."""
    sp = Sequence(("cycle", "pass_"), dimension="time")
    payload = sp.to_json()
    rebuilt = Sequence.from_json(payload)
    assert rebuilt.axis == sp.axis
    assert rebuilt.dimension == sp.dimension


def test_partitioning_from_json_dispatches_by_name() -> None:
    """The factory dispatches Sequence / GroupedSequence / Date by name."""
    seq = partitioning_from_json(Sequence(("a",), dimension="d").to_json())
    assert isinstance(seq, Sequence)
    grp = partitioning_from_json(
        GroupedSequence(("a", "b"), dimension="d", size=10).to_json()
    )
    assert isinstance(grp, GroupedSequence)
    date = partitioning_from_json(
        Date(("time",), resolution="D", dimension="time").to_json()
    )
    assert isinstance(date, Date)


# %%
# partitioning.GroupedSequence -------------------------------------------


def test_grouped_sequence_buckets_last_variable(
    seq_dataset: zc.Dataset,
) -> None:
    """The last variable's values are floor-divided into buckets of ``size``."""
    sp = GroupedSequence(("cycle", "pass_"), dimension="time", size=2)
    keys = [k for k, _ in sp.split(seq_dataset)]
    # pass_ values: 1,1,2,1,2,2,3,3 → bucketed to 0,0,2,0,2,2,2,2.
    seen = {tuple(v for _, v in key) for key in keys}
    assert seen.issubset({(1, 0), (1, 2), (2, 0), (2, 2), (3, 2)})


def test_grouped_sequence_rejects_size_below_two() -> None:
    """``size < 2`` is meaningless and rejected with a clear message."""
    with pytest.raises(PartitionError, match=">= 2"):
        GroupedSequence(("a",), dimension="d", size=1)


def test_grouped_sequence_rejects_non_integer_last_var() -> None:
    """The last variable must be integer-typed (bucket arithmetic)."""
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=4)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("flag", dtype="float32", dimensions=("time",))
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(
                schema.variables["time"], numpy.arange(4, dtype="int64")
            ),
            "flag": zc.Variable(
                schema.variables["flag"], numpy.zeros(4, dtype="float32")
            ),
        },
    )
    sp = GroupedSequence(("flag",), dimension="time", size=2)
    with pytest.raises(PartitionError, match="integer"):
        list(sp.split(ds))


def test_grouped_sequence_to_from_json_roundtrip() -> None:
    """JSON round-trip preserves size and start."""
    sp = GroupedSequence(("a", "b"), dimension="d", size=10, start=5)
    rebuilt = GroupedSequence.from_json(sp.to_json())
    assert (rebuilt.axis, rebuilt.dimension, rebuilt.size, rebuilt.start) == (
        sp.axis,
        sp.dimension,
        sp.size,
        sp.start,
    )


# %%
# partitioning.Date --------------------------------------------------------


def test_date_partitioning_keys_are_components() -> None:
    """Date splits a datetime64 array into per-resolution buckets."""
    times = numpy.array(
        [
            "2024-01-15T00:00:00",
            "2024-01-15T12:00:00",
            "2024-02-01T00:00:00",
            "2024-02-15T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=4)
        .with_variable("time", dtype="datetime64[ns]", dimensions=("time",))
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={"time": zc.Variable(schema.variables["time"], times)},
    )
    date = Date(("time",), resolution="M", dimension="time")
    keys = [k for k, _ in date.split(ds)]
    assert keys == [
        (("year", 2024), ("month", 1)),
        (("year", 2024), ("month", 2)),
    ]


def test_date_rejects_unknown_resolution() -> None:
    """An unknown resolution code raises PartitionError."""
    with pytest.raises(PartitionError, match="resolution"):
        Date(("time",), resolution="bogus", dimension="time")


def test_date_rejects_non_datetime() -> None:
    """A non-datetime variable is rejected at split time."""
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=4)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "time": zc.Variable(
                schema.variables["time"], numpy.arange(4, dtype="int64")
            )
        },
    )
    date = Date(("time",), resolution="D", dimension="time")
    with pytest.raises(PartitionError, match="datetime64"):
        list(date.split(ds))


def test_date_rejects_two_variables() -> None:
    """Date partitioning takes exactly one variable."""
    with pytest.raises(PartitionError, match="exactly one"):
        Date(("a", "b"), resolution="D", dimension="time")


def test_date_to_from_json_roundtrip() -> None:
    """JSON round-trip preserves the resolution + dimension."""
    date = Date(("time",), resolution="D", dimension="time")
    rebuilt = Date.from_json(date.to_json())
    assert rebuilt.resolution == "D"
    assert rebuilt.dimension == "time"


# %%
# Stores -----------------------------------------------------------------


def test_open_store_dispatches_by_scheme(tmp_path) -> None:
    """``open_store`` returns the right concrete class for known schemes."""
    local = open_store(f"file://{tmp_path / 'a'}")
    assert isinstance(local, LocalStore)
    mem = open_store("memory://anything")
    assert isinstance(mem, MemoryStore)
    # Plain path with no scheme is treated as file://.
    plain = open_store(str(tmp_path / "b"))
    assert isinstance(plain, LocalStore)


def test_open_store_rejects_unknown_scheme() -> None:
    """Unknown schemes raise StoreError mentioning the URL."""
    with pytest.raises(StoreError, match="unrecognised"):
        open_store("ftp://example.com/path")


def test_local_store_read_only_blocks_writes(tmp_path) -> None:
    """Opening a LocalStore in read-only mode blocks writes and deletes."""
    rw = LocalStore(str(tmp_path / "store"))
    rw.write_bytes("a/b", b"hello")
    ro = LocalStore(str(tmp_path / "store"), read_only=True)
    assert ro.read_bytes("a/b") == b"hello"
    with pytest.raises(PermissionError):
        ro.write_bytes("a/b", b"world")
    with pytest.raises(PermissionError):
        ro.delete_prefix("a")


def test_local_store_list_dir_yields_subdirs(tmp_path) -> None:
    """``list_dir`` returns sub-directories only; ``list_prefix`` everything."""
    s = LocalStore(str(tmp_path / "store"))
    s.write_bytes("a/file.txt", b"x")
    s.write_bytes("b/c/leaf", b"y")
    s.write_bytes("c.txt", b"z")
    assert sorted(s.list_dir("")) == ["a", "b"]  # sub-dirs only
    assert sorted(s.list_prefix("")) == ["a", "b", "c.txt"]


def test_local_store_delete_prefix_removes_tree(tmp_path) -> None:
    """``delete_prefix`` removes a directory recursively."""
    s = LocalStore(str(tmp_path / "store"))
    s.write_bytes("a/b/c.txt", b"x")
    s.write_bytes("a/b/d.txt", b"y")
    assert s.exists("a/b")
    s.delete_prefix("a")
    assert not s.exists("a")


def test_local_store_read_missing_returns_none(tmp_path) -> None:
    """Reading an absent key yields None (not an exception)."""
    s = LocalStore(str(tmp_path / "store"))
    assert s.read_bytes("missing") is None
    # list_prefix on a missing dir yields nothing rather than raising.
    assert list(s.list_prefix("nope")) == []
    assert list(s.list_dir("nope")) == []


def test_local_store_repr_and_root_uri(tmp_path) -> None:
    """``LocalStore`` exposes a usable ``root_uri`` and a ``repr``."""
    s = LocalStore(str(tmp_path / "store"))
    assert s.root_uri.startswith("file://")
    assert "LocalStore" in repr(s)


def test_memory_store_extras_and_zarr_namespace() -> None:
    """``MemoryStore`` keeps extras and zarr keys independently."""
    s = MemoryStore()
    s.write_bytes("note.txt", b"hello")
    assert s.read_bytes("note.txt") == b"hello"
    assert s.exists("note.txt")
    assert "note.txt" in list(s.list_prefix(""))
    s.delete_prefix("note.txt")
    assert s.read_bytes("note.txt") is None


def test_memory_store_root_uri_is_unique_per_instance() -> None:
    """Each MemoryStore has a distinct ``memory://...`` URI."""
    a = MemoryStore()
    b = MemoryStore()
    assert a.root_uri != b.root_uri
    assert a.root_uri.startswith("memory://")
    assert repr(a) == a.root_uri


def test_memory_store_async_run_inside_running_loop() -> None:
    """The internal ``_run_sync`` works even when an event loop is running."""

    async def _exercise() -> int:
        s = MemoryStore()
        # Trigger the zarr-backed list_dir path which needs an event loop.
        s.write_bytes("x/y", b"v")
        # ``list_dir`` calls _run_sync internally; this must not deadlock.
        names = list(s.list_dir(""))
        return len(names)

    assert asyncio.run(_exercise()) >= 1


# %%
# Cross-module: JSON serialization is the supported round-trip ----------


def test_dataset_schema_json_is_canonical_round_trip() -> None:
    """``to_json`` / ``from_json`` is the schema's canonical persistence."""
    schema = (
        zc.Schema()
        .with_dimension("time", chunks=8)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .build()
    )
    blob = json.dumps(schema.to_json())
    rebuilt = zc.DatasetSchema.from_json(json.loads(blob))
    assert isinstance(rebuilt, zc.DatasetSchema)
    assert set(rebuilt.variables) == set(schema.variables)
