# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Translation and configuration of codecs
========================================

A variable names the codecs it is held behind, using the classes
numcodecs defines or the ones Zarr defines. This module translates the
first into the second.

Blosc, Zstd and GZip are translated into the Zarr codec of the same
name. Every other codec of numcodecs is translated into the wrapper Zarr
ships for it, which writes the identifier numcodecs gives it, prefixed
with ``numcodecs.``. A codec Zarr ships no wrapper for is refused. A
codec already named by Zarr goes through untouched.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numcodecs
import numcodecs.abc
import zarr.abc.codec
import zarr.codecs
import zarr.codecs.numcodecs
import zarr.registry

Codec: TypeAlias = numcodecs.abc.Codec | zarr.abc.codec.Codec

if TYPE_CHECKING:
    from typing import Literal
    from collections.abc import Callable, Iterable, Sequence

    #: Builds the Zarr codecs standing for a codec, from its configuration.
    Build = Callable[[dict[str, Any]], tuple[zarr.abc.codec.Codec, ...]]

#: Marker selecting the codecs Zarr applies when none is named.
DEFAULT_COMPRESSOR = 'auto'

#: :class:`numcodecs.Blosc` names a shuffle by an integer, Zarr by a name.
#: ``None`` lets Zarr pick one from the size of an item.
_SHUFFLE: dict[int, Literal['noshuffle', 'shuffle', 'bitshuffle'] | None] = {
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


#: How Blosc, Zstd and GZip are translated, by the identifier numcodecs
#: gives each.
_TRANSLATIONS: dict[str, Build] = {
    'blosc': _blosc,
    'zstd': _zstd,
    'gzip': _gzip,
}


def get_config(codec: Codec) -> dict[str, Any]:
    """Return the configuration supplied by a numcodecs or Zarr codec."""
    if isinstance(codec, zarr.abc.codec.Codec):
        return codec.to_dict()
    return codec.get_config()


def from_config(config: dict[str, Any]) -> Codec:
    """Reconstruct a codec through its numcodecs or Zarr registry."""
    if 'id' in config:
        return numcodecs.get_codec(config)
    return zarr.registry.get_codec_class(config['name']).from_dict(config)


def _wrapped(codec: numcodecs.abc.Codec) -> tuple[zarr.abc.codec.Codec, ...]:
    """Translate a codec through the wrapper Zarr ships for it.

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
        The Zarr codecs standing for it.
    """
    if isinstance(codec, zarr.abc.codec.Codec):
        return (codec, )
    build = _TRANSLATIONS.get(codec.codec_id)
    if build is None:
        return _wrapped(codec)
    return build(codec.get_config())


def compressors(
    compressor: numcodecs.abc.Codec | zarr.abc.codec.Codec | str | None,
) -> Any:
    """Return the codecs a chunk is held behind.

    Args:
        compressor: The compressor of a variable, :data:`DEFAULT_COMPRESSOR`
            to let Zarr pick one, or ``None`` to write the chunks as they
            are.

    Returns:
        What :func:`zarr.create_array` takes as its ``compressors``.
    """
    if compressor is None or isinstance(compressor, str):
        return compressor
    return translate(compressor)


def filters(
    codecs: Iterable[numcodecs.abc.Codec | zarr.abc.codec.Codec] | None,
) -> Sequence[zarr.abc.codec.Codec]:
    """Return the codecs applied to the values before compression.

    Args:
        codecs: The filters of a variable.

    Returns:
        What :func:`zarr.create_array` takes as its ``filters``.
    """
    if not codecs:
        return ()
    return tuple(item for codec in codecs for item in translate(codec))
