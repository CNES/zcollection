"""Shared fixtures for zcollection unit tests."""

import numpy
import pytest

import zcollection as zc
from zcollection.partitioning import Sequence


@pytest.fixture
def schema() -> zc.DatasetSchema:
    return (
        zc.Schema()
        .with_dimension("num", size=None, chunks=4)
        .with_dimension("x", size=3, chunks=3)
        .with_variable("num", dtype="int64", dimensions=("num",))
        .with_variable("value", dtype="float32", dimensions=("num", "x"))
        .with_variable("static", dtype="float32", dimensions=("x",))
        .build()
    )


@pytest.fixture
def dataset(schema: zc.DatasetSchema) -> zc.Dataset:
    return zc.Dataset(
        schema=schema,
        variables={
            "num": zc.Variable(
                schema.variables["num"],
                numpy.array([0, 0, 1, 1, 2, 2, 2], dtype="int64"),
            ),
            "value": zc.Variable(
                schema.variables["value"],
                numpy.arange(7 * 3, dtype="float32").reshape(7, 3),
            ),
            "static": zc.Variable(
                schema.variables["static"],
                numpy.array([10.0, 20.0, 30.0], dtype="float32"),
            ),
        },
    )


@pytest.fixture
def partitioning() -> Sequence:
    return Sequence(("num",), dimension="num")


@pytest.fixture(params=["memory", "local"])
def store(request, tmp_path):
    """A fresh Store, parametrised across MemoryStore and LocalStore."""
    if request.param == "memory":
        return zc.MemoryStore()
    return zc.LocalStore(tmp_path / "col")
