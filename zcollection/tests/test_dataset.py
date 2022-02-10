# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing datasets
================
"""
import pickle

import dask.array
import numpy
import pytest
import xarray
import zarr

from .. import dataset, meta
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint enable=unused-import


def test_maybe_truncate():
    """Test the truncation of a string to a given length"""
    data = list(range(1000))
    # pylint: disable=protected-access
    assert dataset._maybe_truncate(data, 10) == "[0, 1, ..."
    assert dataset._maybe_truncate(data, len(str(data))) == str(data)
    # pylint: enable=protected-access


def create_test_variable(name="var1", fill_value=0):
    """Create a test variable"""
    return dataset.Variable(name=name,
                            data=numpy.arange(10, dtype="int64").reshape(5, 2),
                            dimensions=("x", "y"),
                            attrs=(dataset.Attribute(name="attr", value=1), ),
                            compressor=zarr.Blosc(cname="zstd", clevel=1),
                            fill_value=fill_value,
                            filters=(zarr.Delta("int64", "int32"),
                                     zarr.Delta("int32", "int32")))


def create_test_dataset():
    """Create a test dataset"""
    return dataset.Dataset(attrs=(dataset.Attribute(name="attr", value=1), ),
                           variables=(create_test_variable(),
                                      create_test_variable("var2")))


def test_variable(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test variable creation"""
    var = create_test_variable()
    assert var.name == "var1"
    assert var.dtype == numpy.dtype("int64")
    assert var.shape == (5, 2)
    assert var.dimensions == ("x", "y")
    assert var.attrs == (dataset.Attribute(name="attr", value=1), )
    assert var.compressor.cname == "zstd"  # type: ignore
    assert var.compressor.clevel == 1  # type: ignore
    assert var.fill_value == 0
    assert var.size == 10
    assert var.nbytes == 80
    assert var.filters == (
        zarr.Delta("int64", "int32"),
        zarr.Delta("int32", "int32"),
    )
    assert numpy.all(var.values == numpy.arange(10).reshape(5, 2))
    assert numpy.all(var.values == var.values)
    assert tuple(var.dimension_index()) == (("x", 0), ("y", 1))
    assert isinstance(var.metadata(), meta.Variable)
    assert isinstance(str(var), str)
    assert isinstance(repr(var), str)

    var.data = numpy.ones((10, 4), dtype="int64")
    assert var.data.shape == (10, 4)
    assert isinstance(var.data, dask.array.Array)
    assert numpy.all(var.values == 1)

    with pytest.raises(ValueError):
        var.data = numpy.ones((10, 4, 2), dtype="int64")


def test_variable_duplicate(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test of the duplication of variables."""
    var = create_test_variable()
    other = var.duplicate(var.array * 2)
    assert other.name == "var1"
    assert other.dtype == numpy.dtype("int64")
    assert other.shape == (5, 2)
    assert other.dimensions == ("x", "y")
    assert other.attrs == (dataset.Attribute(name="attr", value=1), )
    assert other.compressor.cname == "zstd"  # type: ignore
    assert other.compressor.clevel == 1  # type: ignore
    assert other.fill_value == 0
    assert other.filters == (
        zarr.Delta("int64", "int32"),
        zarr.Delta("int32", "int32"),
    )
    assert numpy.all(var.values == other.values / 2)  # type: ignore
    assert var.have_same_properties(other)

    with pytest.raises(ValueError):
        var.duplicate(numpy.ones((10, 4, 2), dtype="int64"))


def test_variable_concat(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test concatenation of variables."""
    var_a = create_test_variable()
    var_b = create_test_variable()
    var_c = create_test_variable()

    vard = var_a.concat((var_b, var_c), "x")
    assert numpy.all(vard.values == numpy.concatenate(
        (var_a.values, var_b.values, var_c.values), axis=0))

    vard = var_a.concat(var_b, "x")
    assert numpy.all(
        vard.values == numpy.concatenate((var_a.values, var_b.values), axis=0))

    with pytest.raises(ValueError):
        var_a.concat([], "y")


def test_variable_datetime64_to_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test conversion to xarray"""
    dates = numpy.arange(
        numpy.datetime64("2000-01-01", "ms"),
        numpy.datetime64("2000-02-01", "ms"),
        numpy.timedelta64("1", "h"),
    )
    var = dataset.Variable(
        name="time",
        data=dates,
        dimensions=("num_lines", ),
        attrs=(dataset.Attribute(name="attr", value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta("int64", "int64"), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ("num_lines", )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype == "datetime64[ns]"


def test_variable_timedelta64_to_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test conversion to xarray"""
    delta = numpy.diff(
        numpy.arange(
            numpy.datetime64("2000-01-01", "ms"),
            numpy.datetime64("2000-02-01", "ms"),
            numpy.timedelta64("1", "h"),
        ))

    var = dataset.Variable(
        name="timedelta",
        data=delta,
        dimensions=("num_lines", ),
        attrs=(dataset.Attribute(name="attr", value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta("int64", "int64"), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ("num_lines", )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype.kind == "m"


def test_variable_dimension_less(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Concatenate two dimensionless variables.
    """
    data = numpy.array([0, 1], dtype=numpy.int32)
    args = ("nv", data, ("nv", ), (dataset.Attribute("comment", "vertex"),
                                   dataset.Attribute("units", "1")))
    n_vertex = dataset.Variable(*args)
    assert n_vertex.fill_value is None
    metadata = n_vertex.metadata()
    assert metadata.fill_value is None
    assert meta.Variable.from_config(metadata.get_config()) == metadata

    other = dataset.Variable(*args)

    concatenated = n_vertex.concat((other, ), "time")
    assert numpy.all(concatenated.values == n_vertex.values)
    assert concatenated.metadata() == n_vertex.metadata()


def test_variable_masked_values(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the masked where function."""
    var = create_test_variable(fill_value=None)
    other = var.masked_values(1).values
    assert isinstance(other, numpy.ma.MaskedArray)
    assert numpy.all(other == numpy.ma.masked_values(var.values, 1))


def test_variable_masked_where(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the masked where function."""
    var = create_test_variable(fill_value=None)
    other = var.masked_where(var.data % 2 == 0)
    x_values, y_values = other.values, var.values
    assert isinstance(x_values, numpy.ma.MaskedArray)
    assert numpy.all(x_values == numpy.ma.masked_where(y_values %
                                                       2 == 0, y_values))
    other = other.masked_where(other.data % 2 == 1)
    assert numpy.all(other.values.mask)  # type: ignore


def test_variable_logical_operators(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the logical operators."""
    var = create_test_variable(fill_value=None)
    var = var.duplicate(var.data + numpy.pi)
    other = var.duplicate(var.data * 5)
    result = (other > var).compute()
    assert numpy.all(result == (other.values > var.values))
    result = (var < other).compute()
    assert numpy.all(result == (var.values < other.values))
    result = (var.__eq__(var)).compute()
    assert numpy.all(result == (var.values == var.values))
    result = (var != other).compute()
    assert numpy.all(result == (var.values != other.values))
    result = (var >= other).compute()
    assert numpy.all(result == (var.values >= other.values))
    result = (var <= other).compute()
    assert numpy.all(result == (var.values <= other.values))


def test_variable_arithmetic_operators(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the arithmetic operators."""
    var = create_test_variable(fill_value=None)
    var = var.duplicate(var.data + numpy.pi)
    other = var.duplicate(var.data * 5)
    result = (var + other).compute()
    assert numpy.all(result == (var.values + other.values))
    result = (var - other).compute()
    assert numpy.all(result == (var.values - other.values))
    result = (var * other).compute()
    assert numpy.all(result == (var.values * other.values))
    result = (var / other).compute()
    assert numpy.all(result == (var.values / other.values))
    result = (var**other).compute()
    assert numpy.all(result == (var.values**other.values))
    result = (var % other).compute()
    assert numpy.all(result == (var.values % other.values))
    result = (var // var).compute()
    assert numpy.all(result == (var.values // var.values))
    result = (var > 1.5).compute()
    assert numpy.all(result == (var.values > 1.5))
    result = (var < 1.5).compute()
    assert numpy.all(result == (var.values < 1.5))
    result = (var >= 1.5).compute()
    assert numpy.all(result == (var.values >= 1.5))
    result = (var <= 1.5).compute()
    assert numpy.all(result == (var.values <= 1.5))
    result = (var == 1.5).compute()
    assert numpy.all(result == (var.values == 1.5))
    result = (var != 1.5).compute()
    assert numpy.all(result == (var.values != 1.5))
    result = (var + 1.5).compute()
    assert numpy.all(result == (var.values + 1.5))
    result = (var - 1.5).compute()
    assert numpy.all(result == (var.values - 1.5))
    result = (var * 1.5).compute()
    assert numpy.all(result == (var.values * 1.5))
    result = (var / 1.5).compute()
    assert numpy.all(result == (var.values / 1.5))
    result = (var**1.5).compute()
    assert numpy.all(result == (var.values**1.5))
    result = (var % 1.5).compute()
    assert numpy.all(result == (var.values % 1.5))
    result = (var // 1.5).compute()
    assert numpy.all(result == (var.values // 1.5))


def test_variable_binary_operators(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the binary operators."""
    var = create_test_variable()
    other = var.duplicate(var.data * 3)
    result = (var & other).compute()
    assert numpy.all(result == (var.values & other.values))
    result = (var | other).compute()
    assert numpy.all(result == (var.values | other.values))
    result = (var ^ other).compute()
    assert numpy.all(result == (var.values ^ other.values))
    result = (var ^ 3).compute()
    assert numpy.all(result == (var.values ^ 3))
    result = (var & 3).compute()
    assert numpy.all(result == (var.values & 3))
    result = (var | 3).compute()
    assert numpy.all(result == (var.values | 3))


def test_variable_getitem(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    var = create_test_variable()
    values = var.values
    result = var[0].compute()
    assert numpy.all(result == values[0])
    result = var[0:2].compute()
    assert numpy.all(result == values[0:2])
    result = var[0:2, 0].compute()
    assert numpy.all(result == values[0:2, 0])
    result = var[0:2, 0:2].compute()
    assert numpy.all(result == values[0:2, 0:2])


def test_dataset(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset creation"""
    ds = create_test_dataset()
    assert ds.dimensions == dict(x=5, y=2)
    assert ds.attrs == (dataset.Attribute(name="attr", value=1), )
    assert isinstance(str(ds), str)
    assert isinstance(repr(ds), str)
    assert ds.nbytes == 5 * 2 * 8 * 2
    var1 = create_test_variable()
    assert numpy.all(ds.variables["var1"].values == var1.values)
    assert ds.variables["var1"].have_same_properties(var1)
    assert numpy.all(var1.values == ds["var1"].values)
    assert var1.metadata() == ds["var1"].metadata()
    with pytest.raises(KeyError):
        var1 = ds["varX"]
    var2 = create_test_variable("var2")
    assert numpy.all(ds.variables["var2"].values == var2.values)
    assert id(ds["var2"]) == id(ds.variables["var2"])
    assert ds.variables["var2"].have_same_properties(var2)
    assert isinstance(ds.metadata(), meta.Dataset)
    other = ds.compute()
    assert isinstance(other, dataset.Dataset)


def test_dataset_dimensions_conflict(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset creation with dimensions conflict"""
    with pytest.raises(ValueError):
        dataset.Dataset([
            dataset.Variable(name="var1",
                             data=numpy.arange(10,
                                               dtype="int64").reshape(5, 2),
                             dimensions=("x", "y"),
                             attrs=(dataset.Attribute(name="attr", value=1), ),
                             compressor=zarr.Blosc(cname="zstd", clevel=1),
                             fill_value=0,
                             filters=(zarr.Delta("int64", "int32"),
                                      zarr.Delta("int32", "int32"))),
            dataset.Variable(name="var2",
                             data=numpy.arange(20, dtype="int64"),
                             dimensions=("x"),
                             attrs=(dataset.Attribute(name="attr", value=1), ),
                             compressor=zarr.Blosc(cname="zstd", clevel=1),
                             fill_value=0,
                             filters=(zarr.Delta("int64", "int32"),
                                      zarr.Delta("int32", "int32"))),
        ])


def test_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test xarray creation"""
    ds = create_test_dataset()
    xr1 = ds.to_xarray()
    assert isinstance(xr1, xarray.Dataset)

    xr2 = dataset.Dataset.from_xarray(xr1).to_xarray()
    assert xr1 == xr2


def test_dataset_isel(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset selection"""
    ds = create_test_dataset()
    selected_ds = ds.isel(slices=dict(x=slice(0, 2)))
    assert selected_ds.dimensions == dict(x=2, y=2)
    assert selected_ds.attrs == (dataset.Attribute(name="attr", value=1), )
    assert numpy.all(
        selected_ds.variables["var1"].values == numpy.arange(4).reshape(2, 2))
    selected_ds = ds.isel(slices=dict(y=slice(0, 1)))
    assert selected_ds.dimensions == dict(x=5, y=1)
    assert numpy.all(selected_ds.variables["var1"].values == numpy.arange(
        0, 10, 2).reshape(5, 1))

    # Cannot slice on something which is not a dimension
    with pytest.raises(ValueError, match="invalid dimension"):
        ds.isel(slices=dict(z=slice(0, 1)))

    with pytest.raises(ValueError, match="invalid dimension"):
        ds.isel(slices=dict(var1=slice(0, 1)))


def test_dataset_delete(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset deletion"""
    ds = create_test_dataset()

    other = ds.delete([1], "y")
    assert numpy.all(
        other.variables["var1"].values == numpy.arange(0, 10, 2).reshape(5, 1))
    assert numpy.all(
        other.variables["var2"].values == numpy.arange(0, 10, 2).reshape(5, 1))

    other = ds.delete([0], "x")
    assert numpy.all(other.variables["var1"].values == numpy.arange(
        10).reshape(5, 2)[1:, :])


def test_dataset_concat(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test concatenation of datasets."""
    ds1 = create_test_dataset()
    ds2 = create_test_dataset()
    ds3 = ds1.concat(ds2, "y")

    matrix = numpy.arange(10).reshape(5, 2)
    assert numpy.all(ds3.variables["var1"].values == numpy.concatenate(
        (matrix, matrix), axis=1))

    ds3 = ds1.concat(ds2, "x")
    assert numpy.all(ds3.variables["var1"].values == numpy.concatenate(
        (matrix, matrix), axis=0))

    with pytest.raises(ValueError):
        ds1.concat([], "z")


def test_variable_pickle(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test pickling of variables."""
    variable = create_test_variable()
    other = pickle.loads(pickle.dumps(variable))
    assert numpy.all(variable.values == other.values)
    assert variable.attrs == other.attrs
    assert variable.compressor == other.compressor
    assert variable.dimensions == other.dimensions
    assert variable.dtype == other.dtype
    assert variable.fill_value == other.fill_value
    assert variable.filters == other.filters
    assert variable.name == other.name


def test_dataset_pickle(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test pickling of datasets."""
    ds = create_test_dataset()
    other = pickle.loads(pickle.dumps(ds))
    assert ds.attrs == other.attrs
    assert list(ds.variables) == list(other.variables)
    for varname, variable in ds.variables.items():
        assert variable.attrs == other.variables[varname].attrs
        assert variable.compressor == other.variables[varname].compressor
        assert variable.dimensions == other.variables[varname].dimensions
        assert variable.dtype == other.variables[varname].dtype
        assert variable.fill_value == other.variables[varname].fill_value
        assert variable.filters == other.variables[varname].filters
        assert variable.name == other.variables[varname].name


def test_dataset_add_variable(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test for adding a variable.
    """
    ds = create_test_dataset()
    var = meta.Variable("var2", numpy.int64,
                        ("x", "y"), (dataset.Attribute("attr", 1), ),
                        zarr.Blosc(), 255, (zarr.Delta("int64", "int64"), ))
    ds.add_variable(var)

    assert ds.variables["var2"].attrs == (dataset.Attribute(name="attr",
                                                            value=1), )
    assert ds.variables["var2"].dtype == numpy.int64
    assert ds.variables["var2"].dimensions == ("x", "y")
    assert ds.variables["var2"].compressor == zarr.Blosc()
    assert ds.variables["var2"].filters == (zarr.Delta("int64", "int64"), )
    assert ds.variables["var2"].fill_value == 255
    assert ds.variables["var2"].name == "var2"
    assert ds.variables["var2"].shape == (5, 2)
    assert numpy.ma.allequal(
        ds.variables["var2"].values,
        numpy.ma.masked_equal(numpy.full((5, 2), 255, "int64"), 255))

    data = numpy.ones((5, 2), "int64")
    ds.drops_vars("var2")
    assert "var2" not in ds.variables

    var = meta.Variable("var2", numpy.int64,
                        ("x", "y"), (dataset.Attribute("attr", 1), ),
                        zarr.Blosc(), 255, (zarr.Delta("int64", "int64"), ))
    ds.add_variable(var, data)
    assert numpy.ma.allequal(
        ds.variables["var2"].values,
        numpy.ma.masked_equal(numpy.full((5, 2), 1, "int64"), 255))

    data = numpy.ones((10, 2), "int64")
    with pytest.raises(ValueError, match="Conflicting sizes"):
        ds.add_variable(var, data)

    var = meta.Variable("var2", numpy.int64,
                        ("xx", "y"), (dataset.Attribute("attr", 1), ),
                        zarr.Blosc(), 255, (zarr.Delta("int64", "int64"), ))
    with pytest.raises(ValueError, match="has dimension"):
        ds.add_variable(var)


def test_dataset_masked_where(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset creation"""
    ds = create_test_dataset()
    mask = ds["var2"].values == 1
    other = ds.masked_where("var2 == 1")
    assert numpy.all(other.variables["var1"].values == numpy.ma.masked_where(
        mask, ds["var1"].values))
    assert numpy.all(other.variables["var2"].values == numpy.ma.masked_where(
        mask, ds["var2"].values))
    mask = ((ds["var2"].values > 1) & (ds["var2"].values < 5))
    other = ds.masked_where("(var2 > 1) & (var2 < 5)")
    assert numpy.all(other.variables["var1"].values == numpy.ma.masked_where(
        mask, ds["var1"].values))
    assert numpy.all(other.variables["var2"].values == numpy.ma.masked_where(
        mask, ds["var2"].values))
