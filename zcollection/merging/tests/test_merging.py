# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test merging.
=============
"""
import fsspec
import numpy
import zarr

from ... import sync
from ...tests import data
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_configurable, dask_threaded
from ...tests.fs import local_fs
# pylint: enable=unused-import
from .. import _update_fs, merge_time_series, perform


class MyError(RuntimeError):
    """Custom error."""
    ...


class ThrowError(sync.Sync):
    """Throw an error when merging."""

    def __enter__(self) -> bool:
        raise MyError("This is an error")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...


def test_update_fs(local_fs, dask_threaded):
    """Test the _update_fs function."""
    generator = data.create_test_dataset()
    ds = next(generator)

    zattrs = str(local_fs.root.joinpath(".zattrs"))
    root = str(local_fs.root)
    future = dask_threaded.submit(_update_fs, root, dask_threaded.scatter(ds),
                                  local_fs.fs)
    dask_threaded.gather(future)
    assert local_fs.exists(zattrs)

    local_fs.fs.rm(root, recursive=True)
    assert not local_fs.exists(zattrs)
    seen_exception = False
    try:
        future = dask_threaded.submit(_update_fs, root,
                                      dask_threaded.scatter(ds), local_fs.fs,
                                      ThrowError())
        dask_threaded.gather(future)
    except MyError:
        seen_exception = True
    assert seen_exception
    assert not local_fs.exists(zattrs)


def test_perform(dask_threaded):
    """Test the perform function."""
    generator = data.create_test_dataset()
    ds = next(generator)

    fs = fsspec.filesystem("memory")
    path = fs.sep.join(("", "folder"))

    future = dask_threaded.submit(_update_fs, path, dask_threaded.scatter(ds),
                                  fs)
    dask_threaded.gather(future)

    future = dask_threaded.submit(perform, dask_threaded.scatter(ds), path,
                                  "time", fs, "time", merge_time_series)
    dask_threaded.gather(future)

    zgroup = zarr.open_consolidated(fs.get_mapper(path))
    assert numpy.all(zgroup["time"][...] == ds["time"].values)
    assert numpy.all(zgroup["var1"][...] == ds["var1"].values)
