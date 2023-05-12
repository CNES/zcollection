# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for :class:`zcollection.compressed_array.CompressedArray`."""
from typing import Any

import dask.array.core
import dask.array.creation
import dask.array.random
import dask.array.reductions
import dask.array.routines
import dask.array.ufunc
import dask.array.utils
import numpy
import pytest

from ..compressed_array import CompressedArray
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster

# pylint: enable=unused-import

# pylint: disable=unnecessary-lambda # We keep the lambdas for readability
#: Functions to test
functions = [
    lambda x: x,
    lambda x: dask.array.ufunc.expm1(x),
    lambda x: 2 * x,
    lambda x: x / 2,
    lambda x: x**2,
    lambda x: x + x,
    lambda x: x * x,
    lambda x: x[0],
    lambda x: x[:, 1],
    lambda x: x[:1, :, 1:3],
    lambda x: x.T,
    lambda x: dask.array.routines.transpose(x, (1, 2, 0)),
    lambda x: dask.array.reductions.nanmean(x),
    lambda x: dask.array.reductions.nanmean(x, axis=1),
    lambda x: dask.array.reductions.nanmax(x),
    lambda x: dask.array.reductions.nanmin(x),
    lambda x: dask.array.reductions.nanprod(x),
    lambda x: dask.array.reductions.nanstd(x),
    lambda x: dask.array.reductions.nanvar(x),
    lambda x: dask.array.reductions.nansum(x),
    lambda x: dask.array.reductions.median(x, axis=0),
    lambda x: dask.array.reductions.nanargmax(x),
    lambda x: dask.array.reductions.nanargmin(x),
    lambda x: dask.array.reductions.nancumprod(x, axis=0),
    lambda x: dask.array.reductions.nancumsum(x, axis=0),
    lambda x: x.sum(),
    lambda x: x.moment(order=0),
    lambda x: x.mean(),
    lambda x: x.mean(axis=1),
    lambda x: x.std(),
    lambda x: x.std(axis=1),
    lambda x: x.var(),
    lambda x: x.var(axis=1),
    lambda x: x.dot(numpy.arange(x.shape[-1])),
    lambda x: x.dot(numpy.eye(x.shape[-1])),
    lambda x: dask.array.routines.tensordot(
        x, numpy.ones(x.shape[:2]), axes=[(0, 1),
                                          (0, 1)]),  # type: ignore[arg-type]
    lambda x: x.sum(axis=0),
    lambda x: x.max(axis=0),
    lambda x: x.min(axis=0),
    lambda x: x.sum(axis=(1, 2)),
    lambda x: x.astype(numpy.complex128),
    lambda x: x.map_blocks(lambda x: x * 2),
    lambda x: x.map_overlap(
        lambda x: x * 2, depth=0, trim=True, boundary='none'),
    lambda x: x.map_overlap(
        lambda x: x * 2, depth=0, trim=False, boundary='none'),
    lambda x: x.round(1),
    lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2])),
    lambda x: abs(x),
    lambda x: x > 0.5,
    lambda x: x.rechunk((4, 4, 4)),
    lambda x: x.rechunk((2, 2, 1)),
    lambda x: numpy.isneginf(x),
    lambda x: numpy.isposinf(x),
]
# pylint: enable=unnecessary-lambda


@pytest.mark.filterwarnings(
    'ignore:Casting complex values to real discards the imaginary part')
@pytest.mark.parametrize('func', functions)
def test_basic(
        func,
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test basic functionality."""
    values: numpy.ndarray = numpy.random.random((2, 3, 4))
    arr: dask.array.core.Array = dask.array.core.from_array(
        CompressedArray(values), chunks='auto')
    compressed_array: numpy.ndarray = func(arr).compute()
    array: numpy.ndarray = func(dask.array.core.from_array(values)).compute()
    assert compressed_array.shape == array.shape
    assert numpy.allclose(compressed_array, array)


def test_metadata(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test metadata."""
    y: dask.array.core.Array = dask.array.random.random((10, 10),
                                                        chunks=(5, 5))
    z = CompressedArray(y.compute())
    y = y.map_blocks(CompressedArray)  # type: ignore[assignment]

    # pylint: disable=protected-access
    assert isinstance(y._meta, numpy.ndarray)
    assert isinstance((y + 1)._meta, numpy.ndarray)
    assert isinstance(y[:5, ::2]._meta, numpy.ndarray)
    assert isinstance(
        y.rechunk((2, 2))._meta,  # type: ignore[arg-type]
        numpy.ndarray)
    assert isinstance((y - z), numpy.ndarray)
    assert isinstance(y.persist()._meta, numpy.ndarray)
    # pylint: enable=protected-access


def test_from_delayed_meta(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test from_delayed with meta."""

    def f() -> CompressedArray:
        return CompressedArray(numpy.eye(3))

    d: Any = dask.delayed(f)()  # type: ignore
    x: dask.array.core.Array = dask.array.core.from_delayed(
        d, shape=(3, 3), meta=CompressedArray(numpy.eye(1)))
    assert numpy.all(x.compute() == f()[...])  # type: ignore


def test_from_array(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test from_array."""
    x = CompressedArray(numpy.eye(10))
    d: dask.array.core.Array = dask.array.core.from_array(
        x, chunks=(5, 5))  # type: ignore[arg-type]

    # pylint: disable=protected-access
    assert isinstance(d._meta, numpy.ndarray)
    # pylint: enable=protected-access
    assert isinstance(d.compute(), numpy.ndarray)
    assert numpy.allclose(d.compute(), x)


def test_map_blocks(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test map_blocks."""
    x: dask.array.core.Array = dask.array.creation.eye(
        10, chunks=5)  # type: ignore[arg-type]
    y: dask.array.core.Array = x.map_blocks(
        CompressedArray)  # type: ignore[arg-type]
    # pylint: disable=protected-access
    assert isinstance(y._meta, numpy.ndarray)
    # pylint: enable=protected-access
    assert numpy.allclose(y.compute(), x.compute())


def test_compressed_masked_array(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
) -> None:
    """Test CompressedMaskedArray."""
    x: dask.array.core.Array = dask.array.creation.eye(
        10, chunks=5)  # type: ignore[arg-type]
    y: dask.array.core.Array = x.map_blocks(
        CompressedArray, fill_value=0)  # type: ignore[arg-type]
    # assert isinstance(y._meta, CompressedArray)
    assert isinstance(y[...].compute(), numpy.ma.MaskedArray)
    assert isinstance(y.compute(), numpy.ma.MaskedArray)
    assert y.mean().compute() == 1
    assert y.min().compute() == 1
    assert y.max().compute() == 1
    assert y.sum().compute() == 10
    assert y.std().compute() == 0
    assert (y * 2).mean().compute() == 2
