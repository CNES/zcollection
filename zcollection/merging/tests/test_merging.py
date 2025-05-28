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
from ...tests.cluster import dask_client, dask_cluster  # noqa: F401
from ...tests.fixture import dask_arrays, numpy_arrays  # noqa: F401
from ...tests.fs import local_fs  # noqa: F401


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
        dask_client,  # noqa: F811
        local_fs,  # noqa: F811
) -> None:
    """Test the _update_fs function."""
    generator = data.create_test_dataset(delayed=False)
    zds = next(generator)
    zds_sc = dask_client.scatter(zds)

    partition_folder = local_fs.root.joinpath('variable=1')

    zattrs = str(partition_folder.joinpath('.zattrs'))
    future = dask_client.submit(_update_fs, str(partition_folder), zds_sc,
                                local_fs.fs)
    dask_client.gather(future)
    assert local_fs.exists(zattrs)

    local_fs.fs.rm(str(partition_folder), recursive=True)
    assert not local_fs.exists(zattrs)

    with pytest.raises(MyError):
        dask_client.gather(
            dask_client.submit(_update_fs,
                               str(partition_folder),
                               zds_sc,
                               local_fs.fs,
                               synchronizer=ThrowError()))

    assert not local_fs.exists(zattrs)


@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_perform(
    dask_client,  # noqa: F811
    local_fs,  # noqa: F811
    arrays_type,
    request,
) -> None:
    """Test the perform function."""
    delayed = request.getfixturevalue(arrays_type)
    generator = data.create_test_dataset(delayed=delayed)
    zds = next(generator)

    path = str(local_fs.root.joinpath('variable=1'))
    zds_sc = dask_client.scatter(zds)

    future = dask_client.submit(_update_fs, path, zds_sc, local_fs.fs)
    dask_client.gather(future)

    future = dask_client.submit(perform,
                                zds_sc,
                                path,
                                axis='time',
                                fs=local_fs.fs,
                                partitioning_dim='num_lines',
                                delayed=delayed,
                                merge_callable=merge_time_series)
    dask_client.gather(future)

    zgroup = zarr.open_consolidated(local_fs.get_mapper(path))
    assert numpy.all(zgroup['time'][...] == zds['time'].values)
    assert numpy.all(zgroup['var1'][...] == zds['var1'].values)
