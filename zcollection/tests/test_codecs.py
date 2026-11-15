# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Codec translation tests
=======================

Codec translation determines the array metadata and which Zarr
implementations can read it. These tests check the resulting codec names
and configurations.
"""
from __future__ import annotations

from typing import Any, cast

import numcodecs
import numpy
import pytest
import zarr.codecs

from .. import codecs
# pylint: disable=unused-import # Need to import for fixtures
from .fs import local_fs  # noqa: F401

# pylint: enable=unused-import


def names(codec: numcodecs.abc.Codec) -> list[str]:
    """Return the names the translation of a codec is written under."""
    return [
        cast('str',
             item.to_dict()['name']) for item in codecs.translate(codec)
    ]


@pytest.mark.parametrize(('codec', 'expected'), [
    (numcodecs.Blosc(), ['blosc']),
    (numcodecs.Zstd(level=3), ['zstd']),
    (numcodecs.GZip(level=4), ['gzip']),
])
def test_a_registered_codec_is_translated(codec, expected) -> None:
    """Each registered codec uses the name assigned by Zarr."""
    assert names(codec) == expected


@pytest.mark.parametrize(('codec', 'expected'), [
    (numcodecs.Delta(dtype='i8'), ['numcodecs.delta']),
    (numcodecs.Quantize(digits=3, dtype='f8'), ['numcodecs.quantize']),
    (numcodecs.PackBits(), ['numcodecs.packbits']),
    (numcodecs.BitRound(keepbits=5), ['numcodecs.bitround']),
    (numcodecs.FixedScaleOffset(offset=0, scale=1000,
                                dtype='f8'), ['numcodecs.fixedscaleoffset']),
])
def test_an_unregistered_codec_keeps_its_own_name(codec, expected) -> None:
    """A codec translated through the wrapper keeps its numcodecs identifier.

    Zarr writes that identifier prefixed with ``numcodecs.``.
    This name is not part of the Zarr v3 codec registry.
    Reading an array using it requires an implementation that provides
    the codec.
    """
    assert names(codec) == expected


@pytest.mark.parametrize(('shuffle', 'expected'), [
    (numcodecs.Blosc.NOSHUFFLE, 'noshuffle'),
    (numcodecs.Blosc.SHUFFLE, 'shuffle'),
    (numcodecs.Blosc.BITSHUFFLE, 'bitshuffle'),
])
def test_the_shuffle_of_blosc_is_carried(shuffle, expected) -> None:
    """Blosc names a shuffle by an integer, Zarr by a name."""
    codec, = codecs.translate(numcodecs.Blosc(shuffle=shuffle))
    config = cast('dict[str, Any]', codec.to_dict()['configuration'])
    assert config['shuffle'] == expected


def test_the_shuffle_blosc_picks_is_left_to_zarr() -> None:
    """A shuffle Blosc picks from the size of an item is left to Zarr."""
    codec, = codecs.translate(
        numcodecs.Blosc(shuffle=numcodecs.Blosc.AUTOSHUFFLE))
    assert cast('zarr.codecs.BloscCodec', codec).shuffle is not None


def test_a_zarr_codec_is_carried_as_it_is() -> None:
    """A Zarr codec is passed through unchanged."""
    codec = zarr.codecs.BloscCodec(cname='lz4')
    assert codecs.translate(codec) == (codec, )


def test_a_codec_zarr_holds_nothing_for_is_refused() -> None:
    """An unsupported codec is rejected."""

    class Unknown(numcodecs.abc.Codec):
        """A codec type not known to Zarr."""
        codec_id = 'unknown'

        def encode(self, buf):
            """Return the buffer."""
            return buf

        def decode(self, buf, out=None):
            """Return the buffer."""
            return buf

    with pytest.raises(ValueError, match='holds no codec'):
        codecs.translate(Unknown())


def test_the_compressor_zarr_picks_is_left_to_it() -> None:
    """``DEFAULT_COMPRESSOR`` becomes ``'auto'``.

    ``filters(None)`` returns ``()``; ``compressors(None)`` returns ``None``.
    """
    assert codecs.compressors(codecs.DEFAULT_COMPRESSOR) == 'auto'
    assert codecs.compressors(None) is None
    assert codecs.filters(None) == ()
    assert codecs.filters(()) == ()


@pytest.mark.parametrize('astype', ['f8', 'i4'])
def test_a_translated_scaling_holds_the_same_values(
        local_fs,  # noqa: F811
        astype) -> None:
    """Translated scaling produces the same values as numcodecs.

    numcodecs rounds scaled values before writing them.
    """
    values = numpy.linspace(-1.5, 1.5, 1000)
    codec = numcodecs.FixedScaleOffset(offset=0,
                                       scale=1000,
                                       dtype='f8',
                                       astype=astype)
    array = zarr.create_array(
        store=f'{local_fs.root}/scaled',
        shape=values.shape,
        dtype='float64',
        # Zarr checks the role of each codec.
        filters=codecs.filters((codec, )),  # type: ignore[arg-type]
        compressors=[zarr.codecs.BloscCodec()])
    array[...] = values

    numpy.testing.assert_array_equal(array[...],
                                     codec.decode(codec.encode(values)))
