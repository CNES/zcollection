# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Convenience functions
=====================
"""
from typing import Optional, Union

import fsspec
import xarray

from . import collection, dataset, partitioning, sync, utilities, view


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
    filesystem = utilities.get_fs(kwargs.pop("filesystem", None))
    if filesystem.exists(partition_base_dir):
        raise ValueError(
            f"The directory {partition_base_dir!r} already exists.")
    if isinstance(ds, xarray.Dataset):
        ds = dataset.Dataset.from_xarray(ds)
    return collection.Collection(axis,
                                 ds.metadata(),
                                 partition_handler,
                                 partition_base_dir,
                                 mode="w",
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


def create_view(
    path: str,
    view_ref: view.ViewReference,
    *,
    filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
    synchronizer: Optional[sync.Sync] = None,
) -> view.View:
    """Create a new view.

    Args:
        path: View storage directory.
        view_ref: Access properties for the reference view.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.

    Example:
        >>> view_ref = ViewReference(
        ...     partition_base_dir="/data/mycollection")
        >>> view = create_view("/home/user/myview", view_ref)

    Returns:
        The created view.

    Raises:
        ValueError: If the path already exists.
    """
    filesystem = utilities.get_fs(filesystem)
    if filesystem.exists(path):
        raise ValueError(f"path {path!r} already exists.")
    return view.View(path,
                     view_ref,
                     ds=None,
                     filesystem=filesystem,
                     synchronizer=synchronizer)


def open_view(
    path: str,
    *,
    filesystem: Optional[Union[fsspec.AbstractFileSystem, str]] = None,
    synchronizer: Optional[sync.Sync] = None,
) -> view.View:
    """Open an existing view.

    Args:
        path: View storage directory.
        filesystem: The file system used to access the view.
        synchronizer: The synchronizer used to synchronize the view.

    Returns:
        The opened view.

    Example:
        >>> view = open_view("/home/user/myview")
    """
    return view.View.from_config(path,
                                 filesystem=filesystem,
                                 synchronizer=synchronizer)
