# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Merging datasets within a partition
===================================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
import hashlib
import shutil

import fsspec
import fsspec.implementations.local

from zcollection import fs_utils

from .. import dataset, storage, store, sync
from .time_series import merge_time_series

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ('MergeCallable', 'merge_time_series', 'perform')


#: pylint: disable=too-few-public-methods,duplicate-code
class MergeCallable(Protocol):
    """Protocol for functions that merge partition datasets.

    A merge function receives the existing and inserted datasets, the axis
    used for merging and the partitioned dimension. It returns the merged
    dataset.
    """

    def __call__(
        self,
        existing_ds: dataset.Dataset,
        inserted_ds: dataset.Dataset,
        axis: str,
        partitioning_dim: str,
        **kwargs,
    ) -> dataset.Dataset:
        """Call the merge function.

        Args:
            existing_ds: The existing dataset.
            inserted_ds: The inserted dataset.
            axis: The axis to merge on.
            partitioning_dim: The partitioning dimension.
            **kwargs: Additional keyword arguments.

        Returns:
            The merged dataset.
        """

        # Ellipsis is necessary to make the function signature match the
        # protocol.
        ...  # pragma: no cover

    #: pylint: enable=too-few-public-methods,duplicate-code


def _rename(
    fs: fsspec.AbstractFileSystem,
    source: str,
    dest: str,
) -> None:
    """Move a directory to a new path, replacing the destination.

    On a local file system, the directory is moved directly. On other file
    systems, its files are copied before the source is removed.

    Args:
        fs: The file system.
        source: The source directory.
        dest: The destination directory.
    """
    if isinstance(fs, fsspec.implementations.local.LocalFileSystem):
        # shutil moves the directory in place.
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(source, dest)
        return

    fs.rm(dest, recursive=True)

    # A recursive expansion yields the prefixes of a store as well as its
    # keys, and copying a prefix raises. find() yields the keys alone.
    keys: list[str] = fs.find(source)
    fs.copy(keys, [f'{dest}{key[len(source):]}' for key in keys])
    fs.rm(source, recursive=True)


def _extract_root_dirname(dirname: str, sep: str) -> str:
    """Return the root directory from a partition path."""
    parts = filter(lambda x: '=' not in x, dirname.split(sep))
    return sep.join(parts)


def _update_fs(
    dirname: str,
    zds: dataset.Dataset,
    fs: fsspec.AbstractFileSystem,
    *,
    distributed: bool = True,
    synchronizer: sync.Sync | None = None,
) -> None:
    """Replace a partition dataset with new data.

    The new dataset is written to a temporary directory at the collection
    root, then moved to the partition path.

    Args:
        dirname: The name of the partition.
        zds: The dataset to update.
        fs: The file system storing the partition.
        distributed: Whether to use Dask. Defaults to ``True``.
        synchronizer: The instance handling access to critical resources.
    """
    # Create a temporary directory named after the partition path's hash
    # to store the new data.
    temp: str = fs_utils.join_path(
        _extract_root_dirname(dirname, fs.sep),
        hashlib.sha256(dirname.encode()).hexdigest())
    if fs.exists(temp):
        fs.rm(temp, recursive=True)

    # Initialize the temporary Zarr group.
    store.create_group(fs, temp)

    # Write the new dataset.
    try:
        # The synchronization is done by the caller.
        storage.write_zarr_group(zds,
                                 temp,
                                 fs,
                                 synchronizer or sync.NoSync(),
                                 distributed=distributed)
    except Exception:
        # Every scheduled task is finished when the exception arrives, so
        # the temporary directory is free.
        fs.rm(temp, recursive=True)
        raise

    # Replace the partition with the temporary directory.
    _rename(fs, temp, dirname)


def perform(
    ds_inserted: dataset.Dataset,
    dirname: str,
    axis: str,
    fs: fsspec.AbstractFileSystem,
    partitioning_dim: str,
    *,
    delayed: bool | None = None,
    distributed: bool = True,
    merge_callable: MergeCallable | None,
    synchronizer: sync.Sync | None = None,
    merge_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Merge an inserted dataset with an existing partition, or replace it.

    Args:
        ds_inserted: The dataset to merge.
        dirname: The name of the partition.
        axis: The axis to merge on.
        fs: The file system storing the partition.
        partitioning_dim: The partitioning dimension.
        delayed: Whether the merge callable receives the stored partition
            and the inserted one as Dask arrays. Defaults to ``None``, which
            gives it Dask arrays when the inserted dataset holds them.
        distributed: Whether to use Dask. Defaults to ``True``.
        merge_callable: The merge callable. If None, the inserted dataset
            overwrites the existing dataset stored in the partition.
            Defaults to None.
        synchronizer: The instance handling access to critical resources.
            Defaults to None.
        merge_kwargs: Keyword arguments passed to the merge callable.
    """
    if merge_callable is None:
        zds = ds_inserted
    else:
        if delayed is None:
            delayed = ds_inserted.delayed
        if not delayed:
            # Load the inserted data into memory: a merge cannot combine
            # a NumPy dataset with a Dask one.
            ds_inserted = ds_inserted.compute()
        ds = storage.open_zarr_group(dirname, fs, delayed=delayed)
        # The dataset read from storage lacks insertion properties. A merge
        # function may discard these properties, depending on its result.
        ds.copy_properties(ds=ds_inserted)
        zds = merge_callable(ds, ds_inserted, axis, partitioning_dim,
                             **(merge_kwargs or {}))
    _update_fs(dirname,
               zds,
               fs,
               distributed=distributed,
               synchronizer=synchronizer)
