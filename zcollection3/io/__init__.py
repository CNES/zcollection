"""I/O primitives over Zarr v3 sync API.

Phase 1 uses Zarr v3's sync surface. Phase 2 swaps the implementation to
:mod:`zarr.api.asynchronous` while keeping these function signatures.
"""
from __future__ import annotations

from .async_partition import (
    open_partition_dataset_async,
    partition_exists_async,
    write_partition_dataset_async,
)
from .partition import (
    open_partition_dataset,
    partition_exists,
    write_partition_dataset,
)
from .root import read_root_config, write_root_config

__all__ = (
    "open_partition_dataset",
    "open_partition_dataset_async",
    "partition_exists",
    "partition_exists_async",
    "read_root_config",
    "write_partition_dataset",
    "write_partition_dataset_async",
    "write_root_config",
)
