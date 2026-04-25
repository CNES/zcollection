"""Shared fixtures for zcollection unit tests."""

import numpy
import pytest

import zcollection as zc
from zcollection.partitioning import Sequence


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--perf`` opt-in flag."""
    parser.addoption(
        "--perf",
        action="store_true",
        default=False,
        help="Run performance tests (requires extra dependencies — "
        "see zcollection/tests/test_perf.py).",
    )
    parser.addoption(
        "--minio-bin",
        action="store",
        default=None,
        help="Path to the minio binary used by S3-backed perf tests. "
        "Falls back to $MINIO_BIN, then to `minio` on PATH.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Declare the ``perf`` marker so pytest does not warn."""
    config.addinivalue_line(
        "markers",
        "perf: opt-in performance benchmarks; enable with --perf.",
    )


@pytest.fixture(scope="session")
def minio_bin(request: pytest.FixtureRequest) -> str | None:
    """Resolve the minio binary path: ``--minio-bin`` > ``$MINIO_BIN`` > PATH."""
    import os
    import shutil

    return (
        request.config.getoption("--minio-bin")
        or os.environ.get("MINIO_BIN")
        or shutil.which("minio")
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.perf`` tests unless ``--perf`` is given."""
    if config.getoption("--perf"):
        return
    skip_perf = pytest.mark.skip(reason="needs --perf to run")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)


@pytest.fixture
def schema() -> zc.DatasetSchema:
    """Provide a small dataset schema fixture for each test."""
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
    """Provide a populated in-memory dataset fixture for each test."""
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
    """Provide a Sequence partitioning over the ``num`` dimension."""
    return Sequence(("num",), dimension="num")


@pytest.fixture(params=["memory", "local"])
def store(request, tmp_path):
    """Provide a fresh Store, parametrised across MemoryStore and LocalStore."""
    if request.param == "memory":
        return zc.MemoryStore()
    return zc.LocalStore(tmp_path / "col")
