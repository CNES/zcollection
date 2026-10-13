# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the translation of codecs
=================================

What a codec becomes decides what lands in the metadata of an array, and
what another implementation of Zarr can read. The table is pinned here.
"""
from __future__ import annotations

import pathlib

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
    return [item.to_dict()['name'] for item in codecs.translate(codec)]


@pytest.mark.parametrize(('codec', 'expected'), [
    (numcodecs.Blosc(), ['blosc']),
    (numcodecs.Zstd(level=3), ['zstd']),
    (numcodecs.GZip(level=4), ['gzip']),
    (numcodecs.FixedScaleOffset(offset=0, scale=1000, dtype='f8',
                                astype='i4'), ['scale_offset', 'cast_value']),
])
def test_a_registered_codec_is_translated(codec, expected) -> None:
    """A codec Zarr registers a name for is written under that name."""
    assert names(codec) == expected


@pytest.mark.parametrize(('codec', 'expected'), [
    (numcodecs.Delta(dtype='i8'), ['numcodecs.delta']),
    (numcodecs.Quantize(digits=3, dtype='f8'), ['numcodecs.quantize']),
    (numcodecs.PackBits(), ['numcodecs.packbits']),
    (numcodecs.BitRound(keepbits=5), ['numcodecs.bitround']),
])
def test_an_unregistered_codec_keeps_its_own_name(codec, expected) -> None:
    """A codec Zarr registers no name for is written under its own.

    Zarr names such a codec by the identifier numcodecs gives it, prefixed
    with ``numcodecs.``. The name is not in the registry of the Zarr v3
    specification, so an implementation that does not hold the codec cannot
    read the arrays it is applied to.
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
    assert codec.to_dict()['configuration']['shuffle'] == expected


def test_the_shuffle_blosc_picks_is_left_to_zarr() -> None:
    """A shuffle Blosc picks from the size of an item is left to Zarr."""
    codec, = codecs.translate(
        numcodecs.Blosc(shuffle=numcodecs.Blosc.AUTOSHUFFLE))
    assert codec.shuffle is not None


def test_a_cast_that_changes_nothing_is_left_out() -> None:
    """A scaling that keeps the data type is one codec, not two."""
    assert names(numcodecs.FixedScaleOffset(offset=0, scale=1000,
                                            dtype='f8')) == ['scale_offset']


def test_a_zarr_codec_is_carried_as_it_is() -> None:
    """A codec already named by Zarr goes through untouched."""
    codec = zarr.codecs.BloscCodec(cname='lz4')
    assert codecs.translate(codec) == (codec, )


def test_a_codec_zarr_holds_nothing_for_is_refused() -> None:
    """A codec with no counterpart raises."""

    class Unknown(numcodecs.abc.Codec):
        """A codec of no known kind."""
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
    """The marker and the absence of a compressor go through."""
    assert codecs.compressors(codecs.DEFAULT_COMPRESSOR) == 'auto'
    assert codecs.compressors(None) is None
    assert codecs.filters(None) == ()
    assert codecs.filters(()) == ()


def test_the_translation_holds_the_same_bytes(local_fs) -> None:  # noqa: F811
    """The translation of a scaling holds what numcodecs holds.

    ``scale_offset`` followed by ``cast_value`` encodes the values the way
    ``FixedScaleOffset`` does, so a collection migrated to the registered
    codecs takes the same room.
    """
    values = numpy.round(numpy.linspace(-1.5, 1.5, 1000), 3)
    sizes = []
    for label, filters in (
        ('numcodecs', (zarr.codecs.numcodecs.FixedScaleOffset(offset=0,
                                                              scale=1000,
                                                              dtype='f8',
                                                              astype='i4'), )),
        ('registered',
         codecs.filters((numcodecs.FixedScaleOffset(offset=0,
                                                    scale=1000,
                                                    dtype='f8',
                                                    astype='i4'), ))),
    ):
        path = f'{local_fs.root}/{label}'
        array = zarr.create_array(store=path,
                                  shape=values.shape,
                                  dtype='float64',
                                  filters=filters,
                                  compressors=[zarr.codecs.BloscCodec()])
        array[...] = values
        numpy.testing.assert_allclose(array[...], values, atol=1e-9)
        sizes.append(
            sum(item.stat().st_size for item in pathlib.Path(path).rglob('*')
                if item.is_file() and item.name != 'zarr.json'))

    assert sizes[0] == sizes[1]
