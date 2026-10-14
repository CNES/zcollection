# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Translation of codecs
=====================

A variable names the codecs it is held behind with the classes numcodecs
defines, or with the ones Zarr defines. This module translates the first
into the second.

A codec Zarr registers a name for is translated into that codec. The
others are translated into the wrapper Zarr ships for them, which writes
the name numcodecs gives them, prefixed with ``numcodecs.``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numcodecs
import numcodecs.abc
import numpy
import zarr.abc.codec
import zarr.codecs
import zarr.codecs.numcodecs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

#: Marker selecting the codecs Zarr applies when none is named.
DEFAULT_COMPRESSOR = 'auto'

#: :class:`numcodecs.Blosc` names a shuffle by an integer, Zarr by a name.
#: ``None`` lets Zarr pick one from the size of an item.
_SHUFFLE: dict[int, str | None] = {
    numcodecs.Blosc.NOSHUFFLE: 'noshuffle',
    numcodecs.Blosc.SHUFFLE: 'shuffle',
    numcodecs.Blosc.BITSHUFFLE: 'bitshuffle',
    numcodecs.Blosc.AUTOSHUFFLE: None,
}


def _blosc(config: dict[str, Any]) -> tuple[zarr.abc.codec.Codec, ...]:
    """Translate a Blosc codec."""
    return (zarr.codecs.BloscCodec(cname=config['cname'],
                                   clevel=config['clevel'],
                                   shuffle=_SHUFFLE[config['shuffle']],
                                   blocksize=config['blocksize']), )


def _zstd(config: dict[str, Any]) -> tuple[zarr.abc.codec.Codec, ...]:
    """Translate a Zstd codec."""
    return (zarr.codecs.ZstdCodec(level=config['level']), )


def _gzip(config: dict[str, Any]) -> tuple[zarr.abc.codec.Codec, ...]:
    """Translate a GZip codec."""
    return (zarr.codecs.GzipCodec(level=config['level']), )


def _fixed_scale_offset(
        config: dict[str, Any]) -> tuple[zarr.abc.codec.Codec, ...]:
    """Translate a FixedScaleOffset codec.

    Zarr holds the scaling and the change of data type in two codecs.
    """
    scale = zarr.codecs.ScaleOffset(offset=config['offset'],
                                    scale=config['scale'])
    astype = numpy.dtype(config['astype'])
    if astype == numpy.dtype(config['dtype']):
        return (scale, )
    return (scale, zarr.codecs.CastValue(data_type=astype.name))


#: Codecs Zarr registers a name for, by the identifier numcodecs gives them.
_REGISTERED: dict[str, Callable[[dict[str, Any]],
                                tuple[zarr.abc.codec.Codec, ...]]] = {
                                    'blosc': _blosc,
                                    'zstd': _zstd,
                                    'gzip': _gzip,
                                    'fixedscaleoffset': _fixed_scale_offset,
                                }


def _wrapped(codec: numcodecs.abc.Codec) -> tuple[zarr.abc.codec.Codec, ...]:
    """Translate a codec Zarr registers no name for.

    Args:
        codec: The codec to translate.

    Returns:
        The wrapper Zarr ships for it.

    Raises:
        ValueError: If Zarr ships no wrapper for the codec.
    """
    name = type(codec).__name__
    wrapper = getattr(zarr.codecs.numcodecs, name, None)
    if wrapper is None:
        raise ValueError(f'Zarr holds no codec standing for {codec!r}.')
    config = {
        key: value
        for key, value in codec.get_config().items() if key != 'id'
    }
    return (wrapper(**config), )


def translate(
    codec: numcodecs.abc.Codec | zarr.abc.codec.Codec,
) -> tuple[zarr.abc.codec.Codec, ...]:
    """Return the Zarr codecs standing for a codec.

    Args:
        codec: A codec of either vocabulary.

    Returns:
        The Zarr codecs. One codec of numcodecs may stand for several.
    """
    if isinstance(codec, zarr.abc.codec.Codec):
        return (codec, )
    build = _REGISTERED.get(codec.codec_id)
    if build is None:
        return _wrapped(codec)
    return build(codec.get_config())


def compressors(
    compressor: numcodecs.abc.Codec | zarr.abc.codec.Codec | str | None,
) -> Any:
    """Return the codecs a chunk is held behind.

    Args:
        compressor: The compressor of a variable, :data:`DEFAULT_COMPRESSOR`
            to let Zarr pick one, or ``None`` to hold the chunks as they are.

    Returns:
        What :func:`zarr.create_array` takes as its ``compressors``.
    """
    if compressor is None or isinstance(compressor, str):
        return compressor
    return translate(compressor)


def filters(
    codecs: Iterable[numcodecs.abc.Codec | zarr.abc.codec.Codec] | None,
) -> Sequence[zarr.abc.codec.Codec]:
    """Return the codecs applied to the values before they are held.

    Args:
        codecs: The filters of a variable.

    Returns:
        What :func:`zarr.create_array` takes as its ``filters``.
    """
    if not codecs:
        return ()
    return tuple(item for codec in codecs for item in translate(codec))
