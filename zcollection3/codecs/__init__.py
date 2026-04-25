"""Codec stacks and default profiles for zcollection v3.

A :class:`CodecStack` is the user-facing description of how a variable's
chunks travel from numpy memory to bytes on disk. It maps onto Zarr v3's
three-stage codec pipeline (array-array, array-bytes, bytes-bytes) and is
serialised as part of each variable's schema.
"""
from __future__ import annotations

from .defaults import (
    DEFAULT_PROFILE,
    PROFILES,
    CodecStack,
    auto_codecs,
    profile,
    profile_names,
    resolve_codec,
)

__all__ = (
    "DEFAULT_PROFILE",
    "PROFILES",
    "CodecStack",
    "auto_codecs",
    "profile",
    "profile_names",
    "resolve_codec",
)
