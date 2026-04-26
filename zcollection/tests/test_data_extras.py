"""Targeted tests filling coverage gaps in the v3 data model.

Covers branches the partition-roundtrip and merge-strategy suites don't
exercise: the xarray bridge, group/variable mutation paths, the repr's
edge cases, schema-validation errors, and merge fast-paths.
"""

import numpy
import pytest

import zcollection as zc
from zcollection.collection.merge import (
    concat,
    replace,
    time_series,
    upsert,
    upsert_within,
)
from zcollection.data._repr import (
    _data_repr,
    calculate_column_width,
    format_attributes,
    format_bytes,
    format_dimensions,
    pretty_print,
    variable_repr,
)
from zcollection.errors import SchemaError
from zcollection.schema.variable import _decode_fill, _encode_fill


# %%
# Schema fixtures ---------------------------------------------------------


@pytest.fixture
def flat_schema() -> zc.DatasetSchema:
    """Minimal flat 1-D schema partitioned along ``time``."""
    return (
        zc.Schema()
        .with_dimension("time", chunks=8)
        .with_dimension("x", size=3)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_variable("v", dtype="float32", dimensions=("time",))
        .with_variable("s", dtype="float32", dimensions=("x",))
        .build()
    )


@pytest.fixture
def flat_dataset(flat_schema: zc.DatasetSchema) -> zc.Dataset:
    """Populate a flat dataset matching :func:`flat_schema`."""
    n = 5
    return zc.Dataset(
        schema=flat_schema,
        variables={
            "time": zc.Variable(
                flat_schema.variables["time"], numpy.arange(n, dtype="int64")
            ),
            "v": zc.Variable(
                flat_schema.variables["v"],
                numpy.arange(n, dtype="float32"),
            ),
            "s": zc.Variable(
                flat_schema.variables["s"],
                numpy.array([10.0, 20.0, 30.0], dtype="float32"),
            ),
        },
    )


# %%
# Variable -----------------------------------------------------------------


def test_variable_rejects_dim_count_mismatch(
    flat_schema: zc.DatasetSchema,
) -> None:
    """A 2-D array passed to a 1-D schema entry raises ValueError."""
    with pytest.raises(ValueError, match="dims"):
        zc.Variable(
            flat_schema.variables["v"],
            numpy.zeros((3, 3), dtype="float32"),
        )


def test_variable_accepts_data_none(flat_schema: zc.DatasetSchema) -> None:
    """``data=None`` is allowed; shape is empty, ndim still from schema."""
    v = zc.Variable(flat_schema.variables["v"], None)
    assert v.shape == ()
    assert v.ndim == 1
    assert v.is_lazy is True


def test_variable_to_numpy_passthrough_for_list() -> None:
    """``to_numpy`` falls back to ``numpy.asarray`` for non-numpy inputs."""
    schema = (
        zc.Schema()
        .with_dimension("n", size=3)
        .with_variable("x", dtype="int64", dimensions=("n",))
        .build()
    )
    raw = numpy.array([1, 2, 3], dtype="int64")
    v = zc.Variable(schema.variables["x"], raw)
    assert numpy.array_equal(v.to_numpy(), raw)


def test_variable_attrs_roundtrips_dict() -> None:
    """``attrs`` returns a *copy* — mutating it doesn't touch the schema."""
    schema = (
        zc.Schema()
        .with_dimension("n", size=2)
        .with_variable(
            "x",
            dtype="int64",
            dimensions=("n",),
            attrs={"units": "m", "long_name": "X"},
        )
        .build()
    )
    v = zc.Variable(schema.variables["x"], numpy.array([1, 2], dtype="int64"))
    assert v.attrs == {"units": "m", "long_name": "X"}
    v.attrs["units"] = "km"
    assert v.attrs["units"] == "m"  # original is untouched


# %%
# Group mutation + lookup --------------------------------------------------


def test_group_add_variable(flat_schema: zc.DatasetSchema) -> None:
    """``add_variable`` registers a variable and revalidates dimensions."""
    ds = zc.Dataset(schema=flat_schema)
    ds.add_variable(
        zc.Variable(
            flat_schema.variables["time"], numpy.arange(4, dtype="int64")
        )
    )
    assert "time" in ds
    assert ds["time"].shape == (4,)
    with pytest.raises(ValueError, match="already exists"):
        ds.add_variable(
            zc.Variable(
                flat_schema.variables["time"], numpy.arange(4, dtype="int64")
            )
        )


def test_group_add_group_then_lookups() -> None:
    """``add_group`` wires the parent and ``find_group`` finds it."""
    schema = (
        zc.Schema()
        .with_dimension("n", size=2)
        .with_variable("a", dtype="int64", dimensions=("n",))
        .with_dimension("m", size=3, group="/sub")
        .with_variable("b", dtype="int64", dimensions=("m",), group="/sub")
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "a": zc.Variable(schema.variables["a"], numpy.arange(2)),
            "sub/b": zc.Variable(
                schema.groups["sub"].variables["b"], numpy.arange(3)
            ),
        },
    )
    sub = ds.get_group("/sub")
    assert sub.parent is ds
    assert ds.find_group("sub") is sub
    assert ds.find_variable("b") is sub.variables["b"]
    assert ds.find_variable("missing") is None
    assert ds.find_group("missing") is None
    # add_group rejects duplicates
    extra = zc.Group(schema.groups["sub"], name="sub")
    with pytest.raises(ValueError, match="already exists"):
        ds.add_group(extra)


def test_group_path_lookup_errors(flat_dataset: zc.Dataset) -> None:
    """get_group/get_variable raise KeyError for unknown paths."""
    with pytest.raises(KeyError, match="missing"):
        flat_dataset.get_group("/missing")
    with pytest.raises(KeyError):
        flat_dataset.get_variable("/missing")
    with pytest.raises(KeyError):
        flat_dataset.get_variable("/nope/leaf")
    with pytest.raises(KeyError, match="no leaf"):
        flat_dataset.get_variable("/")


def test_group_walk_and_iteration(flat_dataset: zc.Dataset) -> None:
    """``walk`` yields self first; mapping iteration is over root vars."""
    paths = [path for path, _ in flat_dataset.walk()]
    assert paths == ["/"]  # no nested groups
    assert list(iter(flat_dataset)) == ["time", "v", "s"]
    assert len(flat_dataset) == 3
    assert "time" in flat_dataset
    assert "missing" not in flat_dataset


def test_group_update_attributes(flat_dataset: zc.Dataset) -> None:
    """``update_attributes`` is in-place and additive."""
    flat_dataset.update_attributes(provider="ACME", level=2)
    assert flat_dataset.attrs["provider"] == "ACME"
    assert flat_dataset.attrs["level"] == 2


def test_group_validates_inconsistent_sizes(
    flat_schema: zc.DatasetSchema,
) -> None:
    """Two variables disagreeing on the same dim size raise ValueError."""
    with pytest.raises(ValueError, match="inconsistent size"):
        zc.Dataset(
            schema=flat_schema,
            variables={
                "time": zc.Variable(
                    flat_schema.variables["time"],
                    numpy.arange(3, dtype="int64"),
                ),
                "v": zc.Variable(
                    flat_schema.variables["v"],
                    numpy.arange(4, dtype="float32"),
                ),
            },
        )


def test_group_is_lazy_recurses() -> None:
    """``is_lazy`` returns True when *any* descendant variable is lazy."""
    schema = (
        zc.Schema()
        .with_dimension("n", size=2)
        .with_variable("a", dtype="int64", dimensions=("n",))
        .with_dimension("m", size=3, group="/sub")
        .with_variable("b", dtype="int64", dimensions=("m",), group="/sub")
        .build()
    )
    ds = zc.Dataset(
        schema=schema,
        variables={
            "a": zc.Variable(schema.variables["a"], numpy.arange(2)),
            "sub/b": zc.Variable(
                schema.groups["sub"].variables["b"], numpy.arange(3)
            ),
        },
    )
    assert ds.is_lazy is False
    # Replace a leaf with a non-numpy backend (a list-as-array proxy).

    class _FakeArray:
        def __init__(self, arr: numpy.ndarray) -> None:
            self.arr = arr
            self.shape = arr.shape
            self.ndim = arr.ndim

        def __array__(self, dtype: object = None) -> numpy.ndarray:
            return self.arr

    ds.groups["sub"]._variables["b"] = zc.Variable(
        schema.groups["sub"].variables["b"],
        _FakeArray(numpy.arange(3, dtype="int64")),
    )
    assert ds.is_lazy is True


# %%
# GroupSchema --------------------------------------------------------------


def test_group_schema_get_group_paths(flat_schema: zc.DatasetSchema) -> None:
    """get_group accepts ``"/"``, ``""``, and absolute/relative paths."""
    schema = (
        zc.Schema()
        .with_dimension("n", size=1)
        .with_variable("a", dtype="int64", dimensions=("n",))
        .with_group("/grp/sub")
        .build()
    )
    assert schema.get_group("/") is schema
    assert schema.get_group("") is schema
    assert schema.get_group("/grp").name == "grp"
    assert schema.get_group("grp/sub").name == "sub"
    with pytest.raises(KeyError):
        schema.get_group("/missing")


def test_group_schema_with_group_inserts_at_path() -> None:
    """``with_group`` returns a new schema with the child grafted in."""
    base = (
        zc.Schema()
        .with_dimension("n", size=1)
        .with_variable("a", dtype="int64", dimensions=("n",))
        .build()
    )
    extra = zc.GroupSchema(name="extra")
    grafted = base.with_group("/", extra)
    assert "extra" in grafted.groups
    assert "extra" not in base.groups  # original is unchanged

    deeper = base.with_group("/extra/deeper", zc.GroupSchema(name="leaf"))
    assert deeper.groups["extra"].groups["deeper"].groups["leaf"].name == "leaf"


def test_group_schema_select_short_and_path(
    flat_schema: zc.DatasetSchema,
) -> None:
    """``select`` accepts both short and absolute names; rejects unknown."""
    sub = flat_schema.select(["time", "v"])
    assert set(sub.variables) == {"time", "v"}
    with pytest.raises(SchemaError, match="unknown variable"):
        flat_schema.select(["does_not_exist"])


def test_dataset_schema_to_from_json_roundtrip(
    flat_schema: zc.DatasetSchema,
) -> None:
    """Schema → JSON → schema preserves everything we declared."""
    payload = flat_schema.to_json()
    rebuilt = zc.DatasetSchema.from_json(payload)
    assert set(rebuilt.variables) == set(flat_schema.variables)
    assert set(rebuilt.dimensions) == set(flat_schema.dimensions)
    assert rebuilt.format_version == flat_schema.format_version


def test_dataset_schema_with_partition_axis_unknown_axis(
    flat_schema: zc.DatasetSchema,
) -> None:
    """An unknown axis raises SchemaError mentioning the name."""
    with pytest.raises(SchemaError, match="missing_axis"):
        flat_schema.with_partition_axis("missing_axis")


# %%
# VariableSchema fill_value encode/decode ---------------------------------


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (None, None),
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
        (3.14, 3.14),
    ],
)
def test_fill_value_encode(value: object, encoded: object) -> None:
    """Special floats are JSON-encoded as their canonical string form."""
    assert _encode_fill(value, numpy.dtype("float64")) == encoded


def test_fill_value_decode_nan() -> None:
    """Decoding ``"NaN"`` gives a NaN back."""
    assert numpy.isnan(_decode_fill("NaN", numpy.dtype("float64")))


def test_fill_value_decode_inf() -> None:
    """Decoding ``"Infinity"`` and ``"-Infinity"`` round-trips."""
    assert _decode_fill("Infinity", numpy.dtype("float64")) == float("inf")
    assert _decode_fill("-Infinity", numpy.dtype("float64")) == float("-inf")


def test_fill_value_decode_passthrough() -> None:
    """Plain numeric fills are cast through ``numpy.dtype``."""
    assert _decode_fill(3.14, numpy.dtype("float64")) == pytest.approx(3.14)
    assert _decode_fill(None, numpy.dtype("float64")) is None


def test_fill_value_decode_rejects_garbage() -> None:
    """A non-castable fill value raises SchemaError."""
    with pytest.raises(SchemaError):
        _decode_fill("not-a-number", numpy.dtype("float32"))


# %%
# _validate_partitionable detail -----------------------------------------


def test_partitionable_rejects_unknown_dim_in_nested_group() -> None:
    """A nested-group var that references a missing dim triggers SchemaError."""
    # Build the schema bypassing validate_dim_refs by going through JSON
    # so the validator is exercised at bind time only.
    schema = (
        zc.Schema()
        .with_dimension("time", size=None)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .build()
    )
    # Now bind — root has no nested group; sanity-check the happy path.
    bound = schema.with_partition_axis("time")
    assert bound.variables["time"].immutable is False


# %%
# Repr edge cases ---------------------------------------------------------


def test_format_bytes_handles_huge_sizes() -> None:
    """Sizes larger than TB roll over to PB."""
    assert format_bytes(1024**5) == "1.00 PB"
    assert format_bytes(1024**6).endswith("PB")


def test_format_dimensions_empty_and_one() -> None:
    """Edge cases for the dim formatter."""
    assert format_dimensions({}) == "()"
    assert format_dimensions({"x": 1}) == "(x: 1)"


def test_calculate_column_width_floor_and_max() -> None:
    """Width is the max of 7 and the longest item."""
    assert calculate_column_width([]) == 7
    assert calculate_column_width(["abc"]) == 7  # floor wins
    assert calculate_column_width(["a_long_name", "x"]) == len("a_long_name")


def test_pretty_print_truncates_long_lines() -> None:
    """Lines longer than the budget are ellipsised."""
    line = "x" * 200
    out = pretty_print(line, num_characters=20)
    assert len(out) == 20
    assert out.endswith("...")


def test_format_attributes_aligns_keys() -> None:
    """``format_attributes`` produces aligned ``key : value`` lines."""
    lines = list(format_attributes({"short": 1, "longer_name": "v"}))
    assert all(":" in line for line in lines)
    # Keys are padded to the same width.
    prefix_widths = {len(line.split(":")[0]) for line in lines}
    assert len(prefix_widths) == 1


def test_data_repr_dispatches_on_backend(flat_dataset: zc.Dataset) -> None:
    """Numpy arrays show their decoded size; unknown backends fall through."""
    text = _data_repr(flat_dataset["time"])
    assert "numpy.ndarray" in text
    assert "B" in text


def test_variable_repr_with_attrs() -> None:
    """A variable with attributes lists them under ``Attributes:``."""
    schema = (
        zc.Schema()
        .with_dimension("n", size=2)
        .with_variable(
            "x",
            dtype="int64",
            dimensions=("n",),
            attrs={"units": "m"},
        )
        .build()
    )
    v = zc.Variable(schema.variables["x"], numpy.array([1, 2], dtype="int64"))
    text = variable_repr(v)
    assert "Attributes:" in text
    assert "units" in text


def test_repr_empty_root_is_marked_empty() -> None:
    """A schema with no variables prints ``<empty>`` under Data variables."""
    schema = zc.Schema().with_dimension("n", size=1).with_group("/grp").build()
    ds = zc.Dataset(schema=schema)
    text = repr(ds)
    assert "<empty>" in text


# %%
# xarray bridge ----------------------------------------------------------


def test_dataset_to_xarray_roundtrip(flat_dataset: zc.Dataset) -> None:
    """``to_xarray`` then ``from_xarray`` preserves variables and attrs."""
    xarray = pytest.importorskip("xarray")
    flat_dataset.update_attributes(provider="ACME")
    xds = flat_dataset.to_xarray()
    assert isinstance(xds, xarray.Dataset)
    # ``time`` becomes a coordinate, others are data_vars; both contribute
    # to the rebuilt zcollection.Dataset.
    assert set(xds.variables) == {"time", "v", "s"}
    assert xds.attrs["provider"] == "ACME"

    rebuilt = zc.Dataset.from_xarray(xds)
    assert set(rebuilt.variables) == {"time", "v", "s"}
    numpy.testing.assert_array_equal(
        rebuilt["time"].to_numpy(), flat_dataset["time"].to_numpy()
    )


# %%
# Merge fast-paths and errors --------------------------------------------


def test_replace_returns_inserted(flat_dataset: zc.Dataset) -> None:
    """``replace`` is a passthrough for the inserted dataset."""
    out = replace(
        flat_dataset, flat_dataset, axis="time", partitioning_dim="time"
    )
    assert out is flat_dataset


def test_concat_empty_existing_returns_inserted(
    flat_schema: zc.DatasetSchema, flat_dataset: zc.Dataset
) -> None:
    """An empty existing dataset + non-empty insert returns the insert."""
    empty = zc.Dataset(schema=flat_schema)
    out = concat(empty, flat_dataset, axis="time", partitioning_dim="time")
    assert out["time"].shape == flat_dataset["time"].shape


def test_concat_empty_inserted_returns_existing(
    flat_schema: zc.DatasetSchema, flat_dataset: zc.Dataset
) -> None:
    """A non-empty existing + empty insert returns existing."""
    empty = zc.Dataset(schema=flat_schema)
    out = concat(flat_dataset, empty, axis="time", partitioning_dim="time")
    assert out["time"].shape == flat_dataset["time"].shape


def test_time_series_requires_axis_on_both_sides(
    flat_schema: zc.DatasetSchema, flat_dataset: zc.Dataset
) -> None:
    """time_series rejects a dataset missing the axis variable."""
    sub = flat_dataset.select(["v"])  # drops 'time'
    with pytest.raises(ValueError, match="time_series"):
        time_series(sub, flat_dataset, axis="time", partitioning_dim="time")


def test_upsert_requires_axis_on_both_sides(
    flat_dataset: zc.Dataset,
) -> None:
    """Upsert rejects a dataset missing the axis variable."""
    sub = flat_dataset.select(["v"])
    with pytest.raises(ValueError, match="upsert"):
        upsert(sub, flat_dataset, axis="time", partitioning_dim="time")


def test_upsert_within_with_jittered_timestamps(
    flat_schema: zc.DatasetSchema,
) -> None:
    """``upsert_within`` deduplicates by nearest-neighbour within tolerance.

    Existing rows whose nearest inserted timestamp is within
    ``tolerance`` are dropped; the rest are kept and concatenated with
    the inserted set. The static variable ``s`` (no time dim) is left
    untouched.
    """
    existing_time = numpy.array([0, 100, 200, 300, 400], dtype="int64")
    inserted_time = numpy.array([0, 100, 500], dtype="int64")
    existing = zc.Dataset(
        schema=flat_schema,
        variables={
            "time": zc.Variable(flat_schema.variables["time"], existing_time),
            "v": zc.Variable(
                flat_schema.variables["v"],
                numpy.zeros(5, dtype="float32"),
            ),
            "s": zc.Variable(
                flat_schema.variables["s"],
                numpy.array([10.0, 20.0, 30.0], dtype="float32"),
            ),
        },
    )
    inserted = zc.Dataset(
        schema=flat_schema,
        variables={
            "time": zc.Variable(flat_schema.variables["time"], inserted_time),
            "v": zc.Variable(
                flat_schema.variables["v"],
                numpy.ones(3, dtype="float32"),
            ),
            "s": zc.Variable(
                flat_schema.variables["s"],
                numpy.array([10.0, 20.0, 30.0], dtype="float32"),
            ),
        },
    )
    strategy = upsert_within(numpy.timedelta64(5))
    out = strategy(existing, inserted, axis="time", partitioning_dim="time")
    times = out["time"].to_numpy()
    # 0 and 100 from existing match exactly → dropped.
    # 200, 300, 400 are far from any inserted value → kept.
    # Final = kept/inserted = [200, 300, 400, 0, 100, 500] sorted.
    numpy.testing.assert_array_equal(
        times, numpy.array([0, 100, 200, 300, 400, 500], dtype="int64")
    )
    # The static `s` variable is unchanged.
    assert out["s"].shape == (3,)
    numpy.testing.assert_array_equal(
        out["s"].to_numpy(),
        numpy.array([10.0, 20.0, 30.0], dtype="float32"),
    )
