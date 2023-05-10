# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test merging.
=============
"""
import numpy
import pytest
import zarr

from .. import _update_fs, merge_time_series, perform
from ... import sync
from ...tests import data
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ...tests.fixture import dask_arrays, numpy_arrays
from ...tests.fs import local_fs

# pylint: enable=unused-import


class MyError(RuntimeError):
    """Custom error."""


class ThrowError(sync.Sync):
    """Throw an error when merging."""

    def __enter__(self) -> bool:
        raise MyError('This is an error')

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    def is_locked(self) -> bool:
        return False


def test_update_fs(
        dask_client,  # pylint: disable=redefined-outer-name
        local_fs,  # pylint: disable=redefined-outer-name
) -> None:
    """Test the _update_fs function."""
    generator = data.create_test_dataset(delayed=False)
    zds = next(generator)

    partition_folder = local_fs.root.joinpath('partition_folder')

    zattrs = str(partition_folder.joinpath('.zattrs'))
    future = dask_client.submit(_update_fs, str(partition_folder),
                                dask_client.scatter(zds), local_fs.fs)
    dask_client.gather(future)
    assert local_fs.exists(zattrs)

    local_fs.fs.rm(str(partition_folder), recursive=True)
    assert not local_fs.exists(zattrs)
    seen_exception = False
    try:
        future = dask_client.submit(_update_fs,
                                    str(partition_folder),
                                    dask_client.scatter(zds),
                                    local_fs.fs,
                                    synchronizer=ThrowError())
        dask_client.gather(future)
    except MyError:
        seen_exception = True
    assert seen_exception
    assert not local_fs.exists(zattrs)


@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_perform(
    dask_client,  # pylint: disable=redefined-outer-name
    local_fs,  # pylint: disable=redefined-outer-name
    arrays_type,
    request,
) -> None:
    """Test the perform function."""
    delayed = request.getfixturevalue(arrays_type)
    generator = data.create_test_dataset(delayed=delayed)
    zds = next(generator)

    path = str(local_fs.root.joinpath('folder'))

    future = dask_client.submit(_update_fs, path, dask_client.scatter(zds),
                                local_fs.fs)
    dask_client.gather(future)

    future = dask_client.submit(perform,
                                dask_client.scatter(zds),
                                path,
                                'time',
                                local_fs.fs,
                                'time',
                                delayed=delayed,
                                merge_callable=merge_time_series)
    dask_client.gather(future)

    zgroup = zarr.open_consolidated(local_fs.get_mapper(path))
    assert numpy.all(zgroup['time'][...] == zds['time'].values)
    assert numpy.all(zgroup['var1'][...] == zds['var1'].values)
