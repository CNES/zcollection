"""Codec profiles and the :class:`CodecStack` data class.

A profile is a (named) recipe that produces a :class:`CodecStack` for a
variable, given its dtype and shape. Phase 1 ships three profiles:

- ``local-fast``: no sharding, Zstd L3.
- ``cloud-balanced``: sharded (~64-256 MiB shards), Zstd L3. *Default*.
- ``cloud-cold``: sharded (~512 MiB shards), Zstd L9.

In Phase 1 we keep the actual codec descriptors as simple dicts so the schema
serialises cleanly without depending on Zarr v3 codec object stability. The
``io`` layer translates these into ``zarr.codecs`` objects at write time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy
import zarr.codecs as zcodecs

#: A single codec is described as ``{"name": str, "configuration": dict}``,
#: matching the Zarr v3 spec for codec metadata. This makes round-tripping
#: through ``_zcollection.json`` trivial.
CodecDescriptor = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CodecStack:
    """The full codec pipeline for one variable."""

    array_to_array: tuple[CodecDescriptor, ...] = ()
    array_to_bytes: CodecDescriptor | None = None
    bytes_to_bytes: tuple[CodecDescriptor, ...] = ()
    sharded: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "array_to_array": list(self.array_to_array),
            "array_to_bytes": self.array_to_bytes,
            "bytes_to_bytes": list(self.bytes_to_bytes),
            "sharded": self.sharded,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CodecStack":
        return cls(
            array_to_array=tuple(payload.get("array_to_array", [])),
            array_to_bytes=payload.get("array_to_bytes"),
            bytes_to_bytes=tuple(payload.get("bytes_to_bytes", [])),
            sharded=bool(payload.get("sharded", False)),
        )


def _bytes_codec() -> CodecDescriptor:
    return {"name": "bytes", "configuration": {"endian": "little"}}


def _zstd(level: int = 3) -> CodecDescriptor:
    return {
        "name": "zstd",
        "configuration": {"level": level, "checksum": False},
    }


@dataclass(frozen=True, slots=True)
class _Profile:
    name: str
    sharded: bool
    target_chunk_bytes: int
    target_shard_bytes: int | None
    compressor: CodecDescriptor

    def codecs(self, dtype: numpy.dtype) -> CodecStack:
        # Phase 1: never emit a ShardingCodec; the io layer can promote
        # to sharding via the ``sharded`` flag once that path lands.
        return CodecStack(
            array_to_array=(),
            array_to_bytes=_bytes_codec(),
            bytes_to_bytes=(self.compressor,),
            sharded=self.sharded,
        )


PROFILES: dict[str, _Profile] = {
    "local-fast": _Profile(
        name="local-fast",
        sharded=False,
        target_chunk_bytes=1 << 20,
        target_shard_bytes=None,
        compressor=_zstd(3),
    ),
    "cloud-balanced": _Profile(
        name="cloud-balanced",
        sharded=False,  # phase 1: sharding wired in phase 3
        target_chunk_bytes=2 << 20,
        target_shard_bytes=128 << 20,
        compressor=_zstd(3),
    ),
    "cloud-cold": _Profile(
        name="cloud-cold",
        sharded=False,
        target_chunk_bytes=2 << 20,
        target_shard_bytes=512 << 20,
        compressor=_zstd(9),
    ),
}

DEFAULT_PROFILE: str = "cloud-balanced"


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)


def profile(
    name: str | None = None,
    *,
    filters: Iterable[CodecDescriptor] | None = None,
    compressor: CodecDescriptor | None = None,
) -> CodecStack:
    """Build a :class:`CodecStack` from a named profile, with overrides."""
    if name is None:
        name = DEFAULT_PROFILE
    if name not in PROFILES:
        raise KeyError(
            f"unknown codec profile {name!r}; "
            f"available: {profile_names()!r}"
        )
    spec = PROFILES[name]
    base = spec.codecs(numpy.dtype("float32"))
    return CodecStack(
        array_to_array=tuple(filters) if filters else base.array_to_array,
        array_to_bytes=base.array_to_bytes,
        bytes_to_bytes=(
            (compressor,) if compressor is not None else base.bytes_to_bytes
        ),
        sharded=base.sharded,
    )


def auto_codecs(
    dtype: numpy.dtype,
    profile_name: str | None = None,
) -> CodecStack:
    """Pick a CodecStack for a dtype using the named (or default) profile."""
    name = profile_name or DEFAULT_PROFILE
    if name not in PROFILES:
        raise KeyError(f"unknown codec profile {name!r}")
    return PROFILES[name].codecs(numpy.dtype(dtype))


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

    Used by the io layer at write time.
    """
    name = descriptor["name"]
    cfg = descriptor.get("configuration", {}) or {}
    builder = _CODEC_BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"unsupported codec descriptor: {descriptor!r}")
    return builder(cfg)
