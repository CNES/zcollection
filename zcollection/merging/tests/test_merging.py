# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test merging.
=============
"""
import numpy
import zarr

# pylint: enable=unused-import
from .. import _update_fs, merge_time_series, perform
from ... import sync
from ...tests import data
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ...tests.fs import local_fs


class MyError(RuntimeError):
    """Custom error."""
    ...


class ThrowError(sync.Sync):
    """Throw an error when merging."""

    def __enter__(self) -> bool:
        raise MyError('This is an error')

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...


def test_update_fs(
        dask_client,  # pylint: disable=redefined-outer-name
        local_fs,  # pylint: disable=redefined-outer-name
):
    """Test the _update_fs function."""
    generator = data.create_test_dataset()
    ds = next(generator)

    partition_folder = local_fs.root.joinpath('partition_folder')

    zattrs = str(partition_folder.joinpath('.zattrs'))
    future = dask_client.submit(_update_fs, str(partition_folder),
                                dask_client.scatter(ds), local_fs.fs)
    dask_client.gather(future)
    assert local_fs.exists(zattrs)

    local_fs.fs.rm(str(partition_folder), recursive=True)
    assert not local_fs.exists(zattrs)
    seen_exception = False
    try:
        future = dask_client.submit(_update_fs, str(partition_folder),
                                    dask_client.scatter(ds), local_fs.fs,
                                    ThrowError())
        dask_client.gather(future)
    except MyError:
        seen_exception = True
    assert seen_exception
    assert not local_fs.exists(zattrs)


def test_perform(
        local_fs,  # pylint: disable=redefined-outer-name
        dask_client,  # pylint: disable=redefined-outer-name
):
    """Test the perform function."""
    generator = data.create_test_dataset()
    ds = next(generator)

    path = str(local_fs.root.joinpath('folder'))

    future = dask_client.submit(_update_fs, path, dask_client.scatter(ds),
                                local_fs.fs)
    dask_client.gather(future)

    future = dask_client.submit(perform, dask_client.scatter(ds), path, 'time',
                                local_fs.fs, 'time', merge_time_series)
    dask_client.gather(future)

    zgroup = zarr.open_consolidated(local_fs.get_mapper(path))
    assert numpy.all(zgroup['time'][...] == ds['time'].values)
    assert numpy.all(zgroup['var1'][...] == ds['var1'].values)
