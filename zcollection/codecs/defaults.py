# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Codec profiles and the :class:`CodecStack` data class.

A profile is a (named) recipe that produces a :class:`CodecStack` for a
variable. Three profiles ship by default:

- ``local-fast``: no sharding, Zstd L3.
- ``cloud-balanced``: sharded (~128 MiB shards), Zstd L3. *Default*.
- ``cloud-cold``: sharded (~512 MiB shards), Zstd L9.

Codec descriptors are kept as simple dicts so the schema serializes cleanly
without depending on Zarr v3 codec object stability. The ``io`` layer
translates these into ``zarr.codecs`` objects at write time via
:func:`resolve_codec`.
"""

from typing import Any
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy
import zarr.codecs as zcodecs


#: A single codec is described as ``{"name": str, "configuration": dict}``,
#: matching the Zarr v3 spec for codec metadata. This makes round-tripping
#: through ``_zcollection.json`` trivial.
CodecDescriptor = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CodecStack:
    """The persisted codec pipeline for one variable.

    Each codec is held as a JSON-clean ``{"name": ..., "configuration": ...}``
    dict so the schema round-trips through ``_zcollection.json`` without
    depending on Zarr v3 codec object identity. Materialisation into
    concrete ``zarr.codecs`` instances happens lazily in
    :func:`resolve_codec` at write time.
    """

    #: Array-to-array codecs (Zarr v3 *filters*), applied in order to the
    #: chunk's numpy array before serialisation.
    array_to_array: tuple[CodecDescriptor, ...] = ()
    #: The single array-to-bytes codec (Zarr v3 *serializer*) that turns
    #: array elements into a byte string for one chunk. The Zarr v3 spec
    #: requires exactly one codec in this slot.
    array_to_bytes: CodecDescriptor | None = None
    #: Bytes-to-bytes codecs (Zarr v3 *compressors*), applied in order to
    #: the per-chunk byte string. With sharding on, they compress each
    #: inner chunk inside a shard; with sharding off, they compress each
    #: chunk directly.
    bytes_to_bytes: tuple[CodecDescriptor, ...] = ()
    #: When ``True``, the variable's chunks are bundled into shards via
    #: :class:`zarr.codecs.ShardingCodec`. The codecs in
    #: :attr:`bytes_to_bytes` then compress each *inner chunk* inside a
    #: shard. When ``False``, each chunk is compressed directly and shards
    #: are not used.
    sharded: bool = False
    #: Target byte budget for each shard when :attr:`sharded` is ``True``;
    #: ``None`` otherwise. The actual shard shape is picked by
    #: :func:`zcollection.codecs.sharding.shard_decision`, which honours
    #: this as a hint, not a hard cap.
    shard_target_bytes: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the codec stack as a JSON-serialisable dictionary."""
        return {
            "array_to_array": list(self.array_to_array),
            "array_to_bytes": self.array_to_bytes,
            "bytes_to_bytes": list(self.bytes_to_bytes),
            "sharded": self.sharded,
            "shard_target_bytes": self.shard_target_bytes,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CodecStack:
        """Build a codec stack from its JSON representation."""
        return cls(
            array_to_array=tuple(payload.get("array_to_array", [])),
            array_to_bytes=payload.get("array_to_bytes"),
            bytes_to_bytes=tuple(payload.get("bytes_to_bytes", [])),
            sharded=bool(payload.get("sharded", False)),
            shard_target_bytes=payload.get("shard_target_bytes"),
        )


def _bytes_codec() -> CodecDescriptor:
    """Return the standard array-to-bytes serializer (little-endian ``bytes``)."""
    return {"name": "bytes", "configuration": {"endian": "little"}}


def _zstd(level: int = 3) -> CodecDescriptor:
    """Return a Zstd bytes-to-bytes (compressor) descriptor at ``level``."""
    return {
        "name": "zstd",
        "configuration": {"level": level, "checksum": False},
    }


@dataclass(frozen=True, slots=True)
class _Profile:
    """A codec profile specification."""

    #: Profile name.
    name: str
    #: Whether the profile produces sharded stacks.
    sharded: bool
    #: Target byte budget for each shard, consumed by
    #: :class:`~zarr.codecs.ShardingCodec` (the array-to-bytes serializer
    #: in a sharded stack). ``None`` when :attr:`sharded` is ``False``.
    target_shard_bytes: int | None
    #: The single bytes-to-bytes codec (compressor) inserted into the
    #: stack — profiles do not currently compose multiple compressors.
    compressor: CodecDescriptor

    def codecs(self) -> CodecStack:
        """Return the :class:`CodecStack` materialised from this profile."""
        return CodecStack(
            array_to_array=(),
            array_to_bytes=_bytes_codec(),
            bytes_to_bytes=(self.compressor,),
            sharded=self.sharded,
            shard_target_bytes=self.target_shard_bytes
            if self.sharded
            else None,
        )


#: The built-in codec profiles, keyed by name.
PROFILES: dict[str, _Profile] = {
    "local-fast": _Profile(
        name="local-fast",
        sharded=False,
        target_shard_bytes=None,
        compressor=_zstd(3),
    ),
    "cloud-balanced": _Profile(
        name="cloud-balanced",
        sharded=True,
        target_shard_bytes=128 << 20,
        compressor=_zstd(3),
    ),
    "cloud-cold": _Profile(
        name="cloud-cold",
        sharded=True,
        target_shard_bytes=512 << 20,
        compressor=_zstd(9),
    ),
}

#: The default profile name, used when no profile is specified.
DEFAULT_PROFILE: str = "cloud-balanced"


def profile_names() -> tuple[str, ...]:
    """Return the names of all registered codec profiles."""
    return tuple(PROFILES)


def profile(
    name: str | None = None,
    *,
    filters: Iterable[CodecDescriptor] | None = None,
    compressor: CodecDescriptor | None = None,
) -> CodecStack:
    """Build a :class:`CodecStack` from a named profile, with overrides.

    Args:
        name: Profile name. ``None`` uses :data:`DEFAULT_PROFILE`.
        filters: Optional array-to-array codec descriptors that override
            the profile's empty filter list.
        compressor: Optional bytes-to-bytes codec descriptor that
            replaces the profile's entire compressor pipeline with this
            single codec.

    Returns:
        A :class:`CodecStack` materialised from the profile, with the
        overrides applied.

    Raises:
        KeyError: If ``name`` is not a registered profile.

    """
    if name is None:
        name = DEFAULT_PROFILE
    if name not in PROFILES:
        raise KeyError(
            f"unknown codec profile {name!r}; available: {profile_names()!r}"
        )
    spec = PROFILES[name]
    base = spec.codecs()
    return CodecStack(
        array_to_array=tuple(filters) if filters else base.array_to_array,
        array_to_bytes=base.array_to_bytes,
        bytes_to_bytes=(
            (compressor,) if compressor is not None else base.bytes_to_bytes
        ),
        sharded=base.sharded,
        shard_target_bytes=base.shard_target_bytes,
    )


def auto_codecs(
    dtype: numpy.dtype,
    profile_name: str | None = None,
) -> CodecStack:
    """Pick a :class:`CodecStack` for a variable using the named profile.

    The result is currently dtype-agnostic — ``dtype`` is accepted on
    the public surface so future profiles can specialise (e.g. byte-grain
    filters for booleans, transposes for high-rank arrays) without an API
    break. For now it is ignored.

    Args:
        dtype: The variable dtype. Reserved for forward compatibility;
            does not affect the returned stack today.
        profile_name: Profile name. ``None`` uses :data:`DEFAULT_PROFILE`.

    Returns:
        A :class:`CodecStack` materialised from the profile.

    Raises:
        KeyError: If ``profile_name`` is not a registered profile.

    """
    del dtype  # reserved; see docstring.
    name = profile_name or DEFAULT_PROFILE
    if name not in PROFILES:
        raise KeyError(f"unknown codec profile {name!r}")
    return PROFILES[name].codecs()


def shard_target_bytes(profile_name: str | None = None) -> int | None:
    """Return the target shard byte budget for a profile.

    Args:
        profile_name: Profile name. ``None`` uses :data:`DEFAULT_PROFILE`.

    Returns:
        The profile's target shard byte budget, or ``None`` if the
        profile does not shard.

    Raises:
        KeyError: If ``profile_name`` is not a registered profile.

    """
    name = profile_name or DEFAULT_PROFILE
    if name not in PROFILES:
        raise KeyError(f"unknown codec profile {name!r}")
    spec = PROFILES[name]
    return spec.target_shard_bytes if spec.sharded else None


#: Mapping from Zarr v3 codec names to builder functions that take a
#: configuration dict and return a Zarr v3 codec instance.
_CODEC_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "bytes": lambda cfg: zcodecs.BytesCodec(endian=cfg.get("endian", "little")),
    "zstd": lambda cfg: zcodecs.ZstdCodec(
        level=cfg.get("level", 3),
        checksum=cfg.get("checksum", False),
    ),
    "blosc": lambda cfg: zcodecs.BloscCodec(**cfg),
    "gzip": lambda cfg: zcodecs.GzipCodec(**cfg),
    "crc32c": lambda _cfg: zcodecs.Crc32cCodec(),
    "transpose": lambda cfg: zcodecs.TransposeCodec(order=cfg["order"]),
    "vlen-utf8": lambda _cfg: zcodecs.VLenUTF8Codec(),
    "vlen-bytes": lambda _cfg: zcodecs.VLenBytesCodec(),
}


def resolve_codec(descriptor: CodecDescriptor) -> Any:
    """Convert a JSON codec descriptor into a Zarr v3 codec instance.

    Used by the I/O layer at write time to materialise the persisted
    descriptors into concrete ``zarr.codecs`` objects.

    Args:
        descriptor: A codec descriptor dict with keys ``"name"`` and
            ``"configuration"`` (the latter is treated as empty when
            absent or ``None``).

    Returns:
        An instance of the corresponding Zarr v3 codec.

    Raises:
        KeyError: If ``descriptor["name"]`` is not in the codec
            registry. The set of supported names is the keys of
            :data:`_CODEC_BUILDERS` (``bytes``, ``zstd``, ``blosc``,
            ``gzip``, ``crc32c``, ``transpose``, ``vlen-utf8``,
            ``vlen-bytes``).

    """
    name = descriptor["name"]
    cfg = descriptor.get("configuration", {}) or {}
    builder = _CODEC_BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"unsupported codec descriptor: {descriptor!r}")
    return builder(cfg)
