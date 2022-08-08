# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Handle merging of datasets of a partition.
==========================================
"""
from typing import Optional, Protocol
import random

import fsspec
import zarr.storage

from .. import dataset, storage, sync
from .time_series import merge_time_series

__all__ = ['MergeCallable', 'perform', 'merge_time_series']

#: Character set used to create a temporary directory.
CHARACTERS = 'abcdefghijklmnopqrstuvwxyz0123456789_'


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
    ) -> dataset.Dataset:  # pylint: disable=duplicate-code
        """Call the partition function.

        Args:
            existing_ds: The existing dataset.
            inserted_ds: The inserted dataset.
            axis: The axis to merge on.
            partitioning_dim: The partitioning dimension.

        Returns:
            The merged dataset.
        """
        # pylint: disable=unnecessary-ellipsis
        # Ellipsis is necessary to make the function signature match the
        # protocol.
        ...  # pragma: no cover
        # pylint: enable=unnecessary-ellipsis

    #: pylint: enable=too-few-public-methods,duplicate-code


def _update_fs(
    dirname: str,
    ds: dataset.Dataset,
    fs: fsspec.AbstractFileSystem,
    synchronizer: Optional[sync.Sync] = None,
) -> None:
    """Updates a dataset stored in a partition.

    Args:
        dirname: The name of the partition.
        ds: The dataset to update.
        fs: The file system that the partition is stored on.
        synchronizer: The instance handling access to critical resources.
    """
    # Name of the temporary directory.
    temp = dirname + '.' + ''.join(
        random.choice(CHARACTERS) for _ in range(10))

    # Initializing Zarr group
    zarr.storage.init_group(store=fs.get_mapper(temp))

    # Writing new data.
    try:
        # The synchronization is done by the caller.
        storage.write_zarr_group(ds, temp, fs, synchronizer or sync.NoSync())
    except Exception:
        # The "write_zarr_group" method throws the exception if all scheduled
        # tasks are finished. So here we can delete the temporary directory.
        fs.rm(temp, recursive=True)
        raise

    # Rename the existing entry on the file system
    fs.rename(temp, dirname, recursive=True)


def perform(
    ds_inserted: dataset.Dataset,
    dirname: str,
    axis: str,
    fs: fsspec.AbstractFileSystem,
    partitioning_dim: str,
    merge_callable: Optional[MergeCallable],
    synchronizer: Optional[sync.Sync] = None,
) -> None:
    """Performs the merge between a new dataset and an existing partition.

    Args:
        ds_inserted: The dataset to merge.
        dirname: The name of the partition.
        axis: The axis to merge on.
        fs: The file system on which the partition is stored on.
        partitioning_dim: The partitioning dimension.
        merge_callable: The merge callable. If None, the inserted dataset
            overwrites the existing dataset stored in the partition.
        synchronizer: The instance handling access to critical resources.
    """
    ds = merge_callable(
        storage.open_zarr_group(dirname, fs), ds_inserted, axis,
        partitioning_dim) if merge_callable is not None else ds_inserted
    _update_fs(dirname, ds, fs, synchronizer)
