# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for :class:`zcollection.compressed_array.CompressedArray`."""
from typing import Any

import dask.array.core
import dask.array.creation
import dask.array.random
import numpy
import pytest
import zarr.abc.codec

from ..compressed_array import CompressedArray


@pytest.mark.parametrize(('value', 'dtype'), [(7, 'int16'), ('a', 'U8')])
def test_numpy_conversion_of_scalar(value, dtype) -> None:
    """Converting a compressed scalar preserves its shape, dtype and value."""
    values = numpy.array(value, dtype=dtype)

    result = numpy.asarray(CompressedArray(values))

    numpy.testing.assert_array_equal(result, values, strict=True)


@pytest.mark.parametrize('indexer', [
    pytest.param(numpy.s_[...], id='all'),
    pytest.param(numpy.s_[0], id='first-axis'),
    pytest.param(numpy.s_[:, 1], id='second-axis'),
    pytest.param(numpy.s_[:1, :, 1:3], id='region'),
])
def test_read_through_dask(indexer) -> None:
    """Reading through Dask selects the values of the compressed array."""
    rng = numpy.random.default_rng(42)
    values: numpy.ndarray = rng.random((2, 3, 4))
    arr: dask.array.core.Array = dask.array.core.from_array(
        CompressedArray(values), chunks='auto')
    numpy.testing.assert_array_equal(arr[indexer].compute(), values[indexer])


def test_metadata() -> None:
    """Test metadata."""
    y: dask.array.core.Array = dask.array.random.random((10, 10),
                                                        chunks=(5, 5))
    z = CompressedArray(y.compute())
    y = y.map_blocks(CompressedArray)  # type: ignore[arg-type,call-arg]

    assert isinstance(y._meta, numpy.ndarray)
    assert isinstance((y + 1)._meta, numpy.ndarray)
    assert isinstance(y[:5, ::2]._meta, numpy.ndarray)
    assert isinstance(
        y.rechunk((2, 2))._meta,  # type: ignore[arg-type]
        numpy.ndarray)
    assert isinstance((y - z), numpy.ndarray)
    assert isinstance(y.persist()._meta, numpy.ndarray)


def test_from_delayed_meta() -> None:
    """Test from_delayed with meta."""

    def f() -> CompressedArray:
        return CompressedArray(numpy.eye(3))

    d: Any = dask.delayed(f)()  # type: ignore[arg-type]
    x: dask.array.core.Array = dask.array.core.from_delayed(
        d, shape=(3, 3), meta=CompressedArray(numpy.eye(1)))
    assert numpy.all(x.compute() == f()[...])  # type: ignore[index]


def test_from_array() -> None:
    """Test from_array."""
    x = CompressedArray(numpy.eye(10))
    d: dask.array.core.Array = dask.array.core.from_array(
        x, chunks=(5, 5))  # type: ignore[arg-type]

    assert isinstance(d._meta, numpy.ndarray)

    assert isinstance(d.compute(), numpy.ndarray)
    assert numpy.allclose(d.compute(), x)


def test_map_blocks() -> None:
    """Test map_blocks."""
    x: dask.array.core.Array = dask.array.creation.eye(
        10, chunks=5)  # type: ignore[arg-type,call-arg]
    y: dask.array.core.Array = x.map_blocks(
        CompressedArray)  # type: ignore[arg-type,call-arg]

    assert isinstance(y._meta, numpy.ndarray)

    assert numpy.allclose(y.compute(), x.compute())


def test_properties() -> None:
    """The properties of the array are the ones Zarr holds for it."""
    array = CompressedArray(numpy.arange(64.0).reshape(8, 8), chunks=(4, 4))

    assert array.ndim == 2
    assert array.shape == (8, 8)
    assert array.size == 64
    assert array.nbytes == 64 * 8
    assert array.dtype == numpy.dtype('float64')
    assert array.chunks == (4, 4)
    assert array.nchunks == 4
    assert all(
        isinstance(item, zarr.abc.codec.Codec) for item in array.compressors)
    assert array.filters == ()
    assert array.fill_value is None


def test_rechunk() -> None:
    """A rechunk keeps the values and the codecs of the array."""
    values = numpy.arange(64.0).reshape(8, 8)
    array = CompressedArray(values, chunks=(4, 4))

    rechunked = array.rechunk((2, 2))

    assert rechunked.chunks == (2, 2)
    assert rechunked.compressors == array.compressors
    assert rechunked.filters == array.filters
    assert numpy.array_equal(numpy.asarray(rechunked), values)


def test_rechunk_scalar() -> None:
    """Rechunking a scalar preserves its shape, dtype and value."""
    values = numpy.array('a', dtype='U8')

    rechunked = CompressedArray(values).rechunk(())

    numpy.testing.assert_array_equal(numpy.asarray(rechunked),
                                     values,
                                     strict=True)


def test_rechunk_keeps_the_value_marking_absence() -> None:
    """The rechunked array masks the values the array it is built from
    masks."""
    array = CompressedArray(numpy.arange(8.0).reshape(2, 4), fill_value=0.0)

    rechunked = array.rechunk((1, 2))

    assert rechunked.fill_value == 0.0
    mask = numpy.ma.getmaskarray(rechunked[:, :])
    assert mask[0, 0]
    assert mask.sum() == 1


def test_compressed_masked_array() -> None:
    """Test CompressedMaskedArray."""
    x: dask.array.core.Array = dask.array.creation.eye(
        10, chunks=5)  # type: ignore[arg-type]
    y: dask.array.core.Array = x.map_blocks(
        CompressedArray, fill_value=0)  # type: ignore[arg-type,call-arg]
    # assert isinstance(y._meta, CompressedArray)
    assert isinstance(y[...].compute(), numpy.ma.MaskedArray)
    assert isinstance(y.compute(), numpy.ma.MaskedArray)
    assert y.mean().compute() == 1
    assert y.min().compute() == 1
    assert y.max().compute() == 1
    assert y.sum().compute() == 10
    assert y.std().compute() == 0
    assert (y * 2).mean().compute() == 2
