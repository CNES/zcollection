"""Auto-shard policy for Zarr v3 sharded arrays.

A *shard* is the unit Zarr writes to the store; it contains one or more
*inner chunks*, which is the unit decoded into memory. Sharding cuts the
PUT count on object stores by ~``shard / chunk`` while keeping per-chunk
compression and random-read locality.

Policy: pick a shard shape that is an integer multiple of the inner-chunk
shape along every dimension, and whose total raw byte size is close to,
but no larger than, ``target_shard_bytes``. We grow shard size dimension
by dimension (largest first) so wide arrays don't end up disproportionately
tall.
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
    """Return a shard shape (a multiple of ``inner_chunks`` per dim).

    Args:
        inner_chunks: The inner chunk shape of the array.
        shape: The array shape; may carry ``None`` for unlimited dimensions.
        dtype: The array dtype.
        target_shard_bytes: The sharding threshold in raw bytes.

    Returns:
        A shard shape, an integer multiple of ``inner_chunks`` along every
        dimension, whose total raw byte size is close to, but no larger than,
        ``target_shard_bytes``.

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

    Args:
        inner_chunks: The inner chunk shape of the array.
        shape: The array shape; may carry ``None`` for unlimited dimensions.
        dtype: The array dtype.
        target_shard_bytes: The sharding threshold in raw bytes, or ``None`` to disable sharding.

    Returns:
        A shard shape, or ``None`` to disable sharding.

    Note:
        This function returns ``None`` when:
        - ``target_shard_bytes`` is None (sharding disabled in the profile);
        - the array fits inside one inner chunk along every dim;
        - the computed shard equals the inner chunks (sharding adds no value).

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
    """Highest power-of-two multiplier of ``chunk`` that fits in ``dim_size``."""
    if dim_size is None:
        # Unlimited dimension — cap arbitrarily large; the doubling loop
        # is bounded by the byte target.
        return 1 << 30
    if chunk <= 0 or dim_size <= 0:
        return 1
    return max(1, dim_size // chunk)


def _clip(extent: int, dim_size: int | None) -> int:
    """Clip the shard extent to the dimension size, if known."""
    if dim_size is None:
        return max(extent, 1)
    return max(1, min(extent, dim_size))


__all__ = (
    "compute_shard_shape",
    "shard_decision",
)
