# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from typing import Optional, Union

import xarray

from .. import collection, dataset, partitioning, utilities


def create_collection(
    axis: str,
    ds: Union[xarray.Dataset, dataset.Dataset],
    partition_handler: partitioning.Partitioning,
    partition_base_dir: str,
    **kwargs,
) -> collection.Collection:
    """Create a collection.

    Args:
        axis: The axis to use for the collection.
        ds: The dataset to use.
        partition_handler: The partitioning handler to use.
        partition_base_dir: The base directory to use for the partitions.
        **kwargs: Additional parameters are passed through to the constructor
            of the class :py:class:`Collection`.

    Example:
        >>> import xarray as xr
        >>> import zcollection
        >>> data = xr.Dataset({
        ...     "a": xr.DataArray([1, 2, 3]),
        ...     "b": xr.DataArray([4, 5, 6])
        ... })
        >>> collection = zcollection.create_collection(
        ...     "a", data,
        ...     zcollection.partitioning.Sequence(("a", )),
        ...     "/tmp/my_collection")

    Returns:
        The collection.

    Raises:
        ValueError: If the base directory already exists.
    """
    filesystem = utilities.get_fs(kwargs.pop('filesystem', None))
    if filesystem.exists(partition_base_dir):
        raise ValueError(
            f'The directory {partition_base_dir!r} already exists.')
    if isinstance(ds, xarray.Dataset):
        ds = dataset.Dataset.from_xarray(ds)
    return collection.Collection(axis,
                                 ds.metadata(),
                                 partition_handler,
                                 partition_base_dir,
                                 mode='w',
                                 filesystem=filesystem,
                                 **kwargs)


# pylint: disable=redefined-builtin
def open_collection(path: str,
                    *,
                    mode: Optional[str] = None,
                    **kwargs) -> collection.Collection:
    """Open a collection.

    Args:
        path: The path to the collection.
        mode: The mode to open the collection.
        **kwargs: Additional parameters are passed through the method
            :py:meth:`zcollection.collection.Collection.from_config`.
    Returns:
        The collection.

    Example:
        >>> import zcollection
        >>> collection = zcollection.open_collection(
        ...     "/tmp/mycollection", mode="r")
    """
    return collection.Collection.from_config(path, mode=mode, **kwargs)
    # pylint: enable=redefined-builtin
