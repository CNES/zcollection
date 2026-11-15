# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Codec translation and configuration
===================================

Convert numcodecs codecs into their Zarr equivalents. Blosc, Zstd and GZip
use dedicated translations; other numcodecs codecs use the corresponding
Zarr wrapper. If no wrapper is available, translation raises ``ValueError``.

Zarr codecs are returned unchanged. The wrappers use codec identifiers
prefixed with ``numcodecs.``.
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

    #: Create Zarr codecs from a codec configuration.
    Build = Callable[[dict[str, Any]], tuple[zarr.abc.codec.Codec, ...]]

#: Marker for using Zarr's default codecs.
DEFAULT_COMPRESSOR = 'auto'

#: :class:`numcodecs.Blosc` identifies shuffle modes with integers, while
#: Zarr uses names. ``None`` lets Zarr choose based on the item size.
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


#: Translation functions for Blosc, Zstd and GZip, keyed by their
#: numcodecs identifiers.
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
    """Convert a numcodecs codec using its corresponding Zarr wrapper.

    Args:
        codec: The codec to translate.

    Returns:
        The Zarr wrapper for the codec.

    Raises:
        ValueError: If Zarr has no corresponding wrapper.
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
    """Convert a codec into the corresponding Zarr codecs.

    Args:
        codec: A numcodecs or Zarr codec.

    Returns:
        The Zarr codecs for the given codec.
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
    """Return the compressors to pass to Zarr.

    Args:
        compressor: The compressor for a variable. Use
            :data:`DEFAULT_COMPRESSOR` for Zarr's default or ``None``
            to store chunks without compression.

    Returns:
        The value accepted by :func:`zarr.create_array` as ``compressors``.
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
