"""Tests for hierarchical :class:`Group` support and the size-aware repr."""

import numpy
import pytest

import zcollection as zc
from zcollection.data._repr import format_bytes, format_dimensions


# Schema helpers -----------------------------------------------------------


@pytest.fixture
def hier_schema() -> zc.DatasetSchema:
    """Build a small two-level hierarchy: root + /data_01/ku."""
    return (
        zc.Schema()
        .with_dimension("time", size=None, chunks=4)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_group("/data_01", attrs={"product": "L2"})
        .with_dimension("range", size=4, chunks=4, group="/data_01/ku")
        .with_attribute("band", "Ku", group="/data_01/ku")
        .with_variable(
            "power",
            dtype="float32",
            dimensions=("time", "range"),
            group="/data_01/ku",
        )
        .build()
    )


@pytest.fixture
def hier_dataset(hier_schema: zc.DatasetSchema) -> zc.Dataset:
    """Provide a populated hierarchical dataset matching ``hier_schema``."""
    ku = hier_schema.groups["data_01"].groups["ku"]
    return zc.Dataset(
        schema=hier_schema,
        variables={
            "time": zc.Variable(
                hier_schema.variables["time"],
                numpy.arange(5, dtype="int64"),
            ),
            "data_01/ku/power": zc.Variable(
                ku.variables["power"],
                numpy.arange(20, dtype="float32").reshape(5, 4),
            ),
        },
    )


# GroupSchema --------------------------------------------------------------


def test_schema_builder_creates_nested_groups(
    hier_schema: zc.DatasetSchema,
) -> None:
    """SchemaBuilder declares nested groups, dimensions, attrs, and variables."""
    assert "data_01" in hier_schema.groups
    data_01 = hier_schema.groups["data_01"]
    assert data_01.attrs["product"] == "L2"
    ku = data_01.groups["ku"]
    assert ku.attrs["band"] == "Ku"
    assert "power" in ku.variables
    assert "range" in ku.dimensions


def test_schema_all_variables_keys_by_path(
    hier_schema: zc.DatasetSchema,
) -> None:
    """``all_variables`` keys descendant variables by absolute path."""
    all_ = hier_schema.all_variables()
    assert set(all_) == {"time", "data_01/ku/power"}


def test_schema_with_partition_axis_marks_static_nested_immutable() -> None:
    """Nested-group vars whose dims are all fixed-size are immutable.

    The contract: every variable must be either partitioned (spans
    ``axis``) or immutable (all dims have a fixed declared size). When
    that's the case, ``with_partition_axis`` recursively tags nested
    static groups as immutable so they can be lifted to ``_immutable/``.
    """
    schema = (
        zc.Schema()
        .with_dimension("time", size=None)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_dimension("x", size=3, group="/grp")
        .with_variable(
            "static", dtype="float32", dimensions=("x",), group="/grp"
        )
        .build()
        .with_partition_axis("time")
    )
    assert schema.variables["time"].immutable is False
    assert schema.groups["grp"].variables["static"].immutable is True


def test_schema_with_partition_axis_rejects_unbounded_non_axis_dim() -> None:
    """A variable with an unbounded non-axis dim makes the schema unsound.

    A 20 Hz time series alongside a 1 Hz partition axis is the canonical
    case: ``data_01_time`` is unbounded (size=None) and isn't the
    partition axis. The collection has no rule to slice it, so binding
    must fail with :class:`SchemaError`.
    """
    schema = (
        zc.Schema()
        .with_dimension("time", size=None)
        .with_variable("time", dtype="int64", dimensions=("time",))
        .with_dimension("data_01_time", size=None, group="/data_01")
        .with_variable(
            "time",
            dtype="datetime64[ns]",
            dimensions=("data_01_time",),
            group="/data_01",
        )
        .build()
    )
    with pytest.raises(zc.SchemaError, match="data_01_time"):
        schema.with_partition_axis("time")


def test_schema_json_round_trip(hier_schema: zc.DatasetSchema) -> None:
    """``to_json``/``from_json`` preserve the full group hierarchy."""
    payload = hier_schema.to_json()
    rebuilt = zc.DatasetSchema.from_json(payload)
    assert set(rebuilt.all_variables()) == set(hier_schema.all_variables())
    assert rebuilt.groups["data_01"].groups["ku"].attrs["band"] == "Ku"


def test_schema_select_path(hier_schema: zc.DatasetSchema) -> None:
    """``select`` accepts absolute paths and prunes empty groups."""
    sub = hier_schema.select(["/data_01/ku/power"])
    assert "time" not in sub.variables
    assert "power" in sub.groups["data_01"].groups["ku"].variables


# Dataset / Group ---------------------------------------------------------


def test_dataset_path_lookup(hier_dataset: zc.Dataset) -> None:
    """``get_variable`` and ``__getitem__`` accept absolute paths."""
    var = hier_dataset.get_variable("/data_01/ku/power")
    assert var.shape == (5, 4)
    assert hier_dataset["/data_01/ku/power"].shape == (5, 4)


def test_group_long_name_and_parent(hier_dataset: zc.Dataset) -> None:
    """``long_name`` returns the absolute path; parents are wired correctly."""
    ku = hier_dataset.get_group("/data_01/ku")
    assert ku.long_name() == "/data_01/ku"
    assert ku.parent is not None
    assert ku.parent.long_name() == "/data_01"
    assert ku.parent.parent is hier_dataset
    assert hier_dataset.is_root() is True
    assert ku.is_root() is False


def test_dimension_inheritance(hier_dataset: zc.Dataset) -> None:
    """A child group sees dimensions declared on an ancestor."""
    ku = hier_dataset.get_group("/data_01/ku")
    # 'time' is declared on the root only; ku must inherit it.
    inherited = ku.find_dimension("time")
    assert inherited is not None
    assert inherited.chunks == 4


def test_nbytes_recursive(hier_dataset: zc.Dataset) -> None:
    """``nbytes`` sums variables across the whole tree (5 i8 + 20 f4 = 120 B)."""
    assert hier_dataset.nbytes == 40 + 80
    assert hier_dataset.get_group("/data_01").nbytes == 80
    assert hier_dataset.get_group("/data_01/ku").nbytes == 80


def test_find_variable_across_tree(hier_dataset: zc.Dataset) -> None:
    """``find_variable`` locates a variable by short name in any descendant."""
    var = hier_dataset.find_variable("power")
    assert var is not None
    assert var.shape == (5, 4)


def test_all_variables_keys(hier_dataset: zc.Dataset) -> None:
    """``Group.all_variables`` keys nested vars by absolute path."""
    keys = set(hier_dataset.all_variables())
    assert keys == {"time", "data_01/ku/power"}


def test_dataset_select_keeps_path(hier_dataset: zc.Dataset) -> None:
    """``Dataset.select`` keeps a nested variable accessible by path."""
    sub = hier_dataset.select(["/data_01/ku/power"])
    assert sub.get_variable("/data_01/ku/power").shape == (5, 4)


# Repr --------------------------------------------------------------------


def test_format_bytes() -> None:
    """``format_bytes`` returns a human-readable size with kB stepping."""
    assert format_bytes(0) == "0 B"
    assert format_bytes(40) == "40 B"
    assert format_bytes(2 * 1024) == "2.00 kB"
    assert format_bytes(5 * 1024 * 1024) == "5.00 MB"


def test_format_dimensions() -> None:
    """``format_dimensions`` renders a dim mapping as ``(name: size, ...)``."""
    assert format_dimensions({"time": 5, "x": 4}) == "(time: 5, x: 4)"
    assert format_dimensions({}) == "()"


def test_repr_contains_size_and_groups(hier_dataset: zc.Dataset) -> None:
    """The dataset repr lists size, dimensions, variables, and groups."""
    text = repr(hier_dataset)
    assert "Size: 120 B" in text
    assert "Dimensions: (time: 5)" in text
    assert "Data variables:" in text
    assert "Groups:" in text
    assert "data_01" in text


def test_variable_repr_has_size(hier_dataset: zc.Dataset) -> None:
    """The variable repr exposes shape, dtype, and a synthetic byte size."""
    text = repr(hier_dataset["time"])
    assert "Size: 40 B" in text
    assert "int64" in text
    assert "(time: 5)" in text


# Hierarchical Zarr roundtrip --------------------------------------------


def test_hier_roundtrip_in_memory(
    hier_dataset: zc.Dataset, hier_schema: zc.DatasetSchema, tmp_path
) -> None:
    """Write a hierarchical dataset and reopen it from a Zarr store."""
    import zarr

    from zcollection.io.partition import (
        open_partition_dataset,
        write_partition_dataset,
    )

    store = zc.LocalStore(str(tmp_path / "store"))
    write_partition_dataset(store, "p0", hier_dataset)

    # Verify native Zarr nesting on disk.
    zstore = store.zarr_store()
    root = zarr.open_group(store=zstore, path="p0", mode="r")
    assert "data_01" in root
    assert "ku" in root["data_01"]
    assert "power" in root["data_01"]["ku"]

    reopened = open_partition_dataset(store, "p0", hier_schema)
    assert "time" in reopened.variables
    assert reopened.get_variable("/data_01/ku/power").shape == (5, 4)
    numpy.testing.assert_array_equal(
        reopened.get_variable("/data_01/ku/power").to_numpy(),
        numpy.arange(20, dtype="float32").reshape(5, 4),
    )
