# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Handle merging of datasets of a partition.
==========================================
"""
from __future__ import annotations

from typing import Protocol
import hashlib
import shutil

import fsspec
import fsspec.implementations.local
import zarr.storage

from zcollection import fs_utils

from .. import dataset, storage, sync
from .time_series import merge_time_series

__all__ = ('MergeCallable', 'perform', 'merge_time_series')


#: pylint: disable=too-few-public-methods,duplicate-code
class MergeCallable(Protocol):
    """Protocol to merge datasets stored in a partition.

    A merge callable is a function that accepts an existing dataset
    present in a partition, a new dataset to merge, the partitioning
    dimension and the axis to merge on. It returns the merged dataset.
    """

    def __call__(
        self,
        existing_ds: dataset.Dataset,
        inserted_ds: dataset.Dataset,
        axis: str,
        partitioning_dim: str,
        **kwargs,
    ) -> dataset.Dataset:  # pylint: disable=duplicate-code
        """Call the partition function.

        Args:
            existing_ds: The existing dataset.
            inserted_ds: The inserted dataset.
            axis: The axis to merge on.
            partitioning_dim: The partitioning dimension.
            **kwargs: Additional keyword arguments.

        Returns:
            The merged dataset.
        """
        # pylint: disable=unnecessary-ellipsis
        # Ellipsis is necessary to make the function signature match the
        # protocol.
        ...  # pragma: no cover
        # pylint: enable=unnecessary-ellipsis

    #: pylint: enable=too-few-public-methods,duplicate-code


def _rename(
    fs: fsspec.AbstractFileSystem,
    source: str,
    dest: str,
) -> None:
    """Rename a directory on a file system.

    Args:
        fs: The file system.
        source: The source directory.
        dest: The destination directory.
    """
    if isinstance(fs, fsspec.implementations.local.LocalFileSystem):
        # fspec implementation of the local file system, copy the source
        # directory to the destination directory and remove the source
        # directory. This is not efficient. So we use the shutil
        # implementation to rename the directory.
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(source, dest)
        return

    fs.rm(dest, recursive=True)
    fs.mv(source, dest, recursive=True)


def _extract_root_dirname(dirname: str, sep: str) -> str:
    """Extracts the root directory name from a partition name."""
    parts = filter(lambda x: '=' not in x, dirname.split(sep))
    return sep.join(parts)


def _update_fs(
    dirname: str,
    zds: dataset.Dataset,
    fs: fsspec.AbstractFileSystem,
    *,
    synchronizer: sync.Sync | None = None,
) -> None:
    """Updates a dataset stored in a partition.

    Args:
        dirname: The name of the partition.
        zds: The dataset to update.
        fs: The file system that the partition is stored on.
        synchronizer: The instance handling access to critical resources.
    """
    # Building a temporary directory to store the new data. The name of the
    # temporary directory is the hash of the partition name.
    temp: str = fs_utils.join_path(
        _extract_root_dirname(dirname, fs.sep),
        hashlib.sha256(dirname.encode()).hexdigest())
    if fs.exists(temp):
        fs.rm(temp, recursive=True)

    # Initializing Zarr group
    zarr.storage.init_group(store=fs.get_mapper(temp))

    # Writing new data.
    try:
        # The synchronization is done by the caller.
        storage.write_zarr_group(zds, temp, fs, synchronizer or sync.NoSync())
    except Exception:
        # The "write_zarr_group" method throws the exception if all scheduled
        # tasks are finished. So here we can delete the temporary directory.
        fs.rm(temp, recursive=True)
        raise

    # Rename the existing entry on the file system
    _rename(fs, temp, dirname)


def perform(
    ds_inserted: dataset.Dataset,
    dirname: str,
    axis: str,
    fs: fsspec.AbstractFileSystem,
    partitioning_dim: str,
    *,
    delayed: bool = True,
    merge_callable: MergeCallable | None,
    synchronizer: sync.Sync | None = None,
    **kwargs,
) -> None:
    """Merges a new dataset with an existing partition.

    Args:
        ds_inserted: The dataset to merge.
        dirname: The name of the partition.
        axis: The axis to merge on.
        fs: The file system on which the partition is stored.
        partitioning_dim: The partitioning dimension.
        delayed: If True, the existing dataset is loaded lazily. Defaults to
            True.
        merge_callable: The merge callable. If None, the inserted dataset
            overwrites the existing dataset stored in the partition.
            Defaults to None.
        synchronizer: The instance handling access to critical resources.
            Defaults to None.
        **kwargs: Additional keyword arguments are passed through to the merge
            callable.
    """
    if merge_callable is None:
        zds = ds_inserted
    else:
        ds = storage.open_zarr_group(dirname, fs, delayed=delayed)
        # Read dataset does not contain insertion properties.
        # This properties might be loss in the merge_callable depending on which
        # dataset is used.
        ds.copy_properties(ds=ds_inserted)
        zds = merge_callable(ds, ds_inserted, axis, partitioning_dim, **kwargs)
    _update_fs(dirname, zds, fs, synchronizer=synchronizer)
