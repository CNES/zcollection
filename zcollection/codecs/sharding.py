# Copyright (c) 2022-2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Auto-shard policy for Zarr v3 sharded arrays.

A *shard* is the unit Zarr writes to the store; it contains one or more
*inner chunks*, which are the units decoded into memory. Sharding cuts
the PUT count on object stores by approximately the shard-to-chunk
ratio (per dimension, the chosen multiplier; combined, the product of
those multipliers) while preserving per-chunk compression and
random-read locality.

Policy: pick a shard shape that is, along every dimension, the
inner-chunk extent multiplied by a **power of two**, and whose total
raw byte size is close to, but no larger than, ``target_shard_bytes``.
The multiplier is grown by repeated doubling: at each step the
dimension with the largest current shard extent is the one that gets
doubled. This keeps growth tracking the array's existing aspect ratio
— well-suited to time-series-dominated layouts where the leading axis
is the natural growth direction.
"""

from typing import TYPE_CHECKING
import math


if TYPE_CHECKING:
    import numpy


def compute_shard_shape(
    *,
    inner_chunks: tuple[int, ...],
    shape: tuple[int | None, ...],
    dtype: numpy.dtype,
    target_shard_bytes: int,
) -> tuple[int, ...]:
    """Return a shard shape (a power-of-two multiple of ``inner_chunks`` per dim).

    The doubling loop stops as soon as the *next* doubling would
    exceed ``target_shard_bytes`` — the returned shard's raw byte
    footprint is therefore always ``<= target_shard_bytes``. When the
    inner chunk already meets or exceeds the target, the chunk shape
    (clipped to the array shape) is returned unchanged.

    Args:
        inner_chunks: The inner chunk shape of the array.
        shape: The array shape; may carry ``None`` for unlimited dimensions.
        dtype: The array dtype (used only for ``itemsize``).
        target_shard_bytes: Target shard byte budget. The result's raw
            byte size is the largest power-of-two-multiple shape that
            does not exceed this value.

    Returns:
        A shard shape whose per-dim extent is ``2**k * inner_chunks[i]``
        for some non-negative integer ``k`` (after clipping to ``shape``).

    Raises:
        ValueError: If ``inner_chunks`` and ``shape`` have different
            ranks.

    """
    if len(inner_chunks) != len(shape):
        raise ValueError(
            f"inner_chunks {inner_chunks!r} and shape {shape!r} disagree on rank"
        )
    if not inner_chunks:
        return ()

    itemsize = max(int(dtype.itemsize), 1)
    inner_bytes = itemsize * math.prod(max(c, 1) for c in inner_chunks)
    if inner_bytes >= target_shard_bytes:
        # Inner chunk is already at/over the target; one chunk per shard.
        return tuple(
            _clip(c, s) for c, s in zip(inner_chunks, shape, strict=True)
        )

    # Per-dim cap: how many inner chunks fit before hitting the array's size.
    caps = tuple(
        _max_multiplier(c, s) for c, s in zip(inner_chunks, shape, strict=True)
    )

    multipliers = [1] * len(inner_chunks)

    def shard_bytes() -> int:
        return inner_bytes * math.prod(multipliers)

    # Greedy doubling: at each step pick the dim with the largest current shard
    # extent that still has slack, and double its multiplier.
    while shard_bytes() * 2 <= target_shard_bytes:
        # Candidates that can still grow.
        idx = -1
        best_extent = -1
        for i, m in enumerate(multipliers):
            if m * 2 > caps[i]:
                continue
            extent = inner_chunks[i] * m
            if extent > best_extent:
                best_extent = extent
                idx = i
        if idx < 0:
            break
        multipliers[idx] *= 2

    return tuple(
        _clip(inner_chunks[i] * multipliers[i], shape[i])
        for i in range(len(inner_chunks))
    )


def shard_decision(
    *,
    inner_chunks: tuple[int, ...],
    shape: tuple[int | None, ...],
    dtype: numpy.dtype,
    target_shard_bytes: int | None,
) -> tuple[int, ...] | None:
    """Return a shard shape, or ``None`` if sharding shouldn't apply.

    See :func:`compute_shard_shape` for argument semantics; this wrapper
    additionally short-circuits to ``None`` when sharding doesn't pay
    off. The two independent paths to ``None`` are:

    - ``target_shard_bytes is None`` — sharding disabled in the profile.
    - The geometry can't grow past the inner chunk shape: either the
      inner chunk already meets/exceeds ``target_shard_bytes`` (a single
      chunk fills a shard), or the array is small enough that every
      per-dim multiplier stays at 1.

    In both of the geometry cases the proposal equals the (clipped)
    inner-chunk shape, so wrapping it in a :class:`ShardingCodec` would
    add overhead with no PUT-count benefit.

    Args:
        inner_chunks: See :func:`compute_shard_shape`.
        shape: See :func:`compute_shard_shape`.
        dtype: See :func:`compute_shard_shape`.
        target_shard_bytes: Target shard byte budget, or ``None`` to
            disable sharding outright.

    Returns:
        A shard shape, or ``None`` when sharding should be skipped.

    """
    if target_shard_bytes is None:
        return None

    proposal = compute_shard_shape(
        inner_chunks=inner_chunks,
        shape=shape,
        dtype=dtype,
        target_shard_bytes=target_shard_bytes,
    )
    if proposal == tuple(
        _clip(c, s) for c, s in zip(inner_chunks, shape, strict=True)
    ):
        return None
    return proposal


def _max_multiplier(chunk: int, dim_size: int | None) -> int:
    """Maximum number of inner chunks that fit along ``dim_size``.

    Returned as an integer cap on the chunk multiplier; the doubling
    loop in :func:`compute_shard_shape` is what restricts the chosen
    multiplier to a power of two within this cap.
    """
    if dim_size is None:
        # Unlimited dimension — cap arbitrarily large; the doubling loop
        # is bounded by the byte target.
        return 1 << 30
    if chunk <= 0 or dim_size <= 0:
        return 1
    return max(1, dim_size // chunk)


def _clip(extent: int, dim_size: int | None) -> int:
    """Clip the shard extent to ``dim_size`` (when known), with a floor of 1.

    The minimum returned value is always ``1`` so callers never end up
    with a zero-extent dimension on degenerate inputs.
    """
    if dim_size is None:
        return max(extent, 1)
    return max(1, min(extent, dim_size))


__all__ = (
    "compute_shard_shape",
    "shard_decision",
)
