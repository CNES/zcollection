# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the storage module.
===========================
"""
import platform
import time

import dask.array
import dask.distributed
import numpy
import zarr

from .. import dataset, storage, sync
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster
from .fs import local_fs

# pylint: enable=unused-import


def test_execute_transaction(
        dask_client,  # pylint: disable=redefined-outer-name
):
    """Test the execute_transaction function."""
    # First case: no transaction to execute
    assert storage.execute_transaction(dask_client, sync.NoSync(), []) is None

    # General case: execute a transaction without error
    def func():
        return 1

    assert sum(
        storage.execute_transaction(
            dask_client, sync.NoSync(),
            [dask_client.submit(func) for i in range(10)])) == 10

    # Degraded case: execute a transaction with error
    def fail(data):
        time.sleep((10 - data) / 100)
        if data % 2 == 0:
            raise ValueError('Odd')
        return data

    exception_seen = False
    futures = []
    try:
        futures += storage.execute_transaction(
            dask_client, sync.NoSync(),
            [dask_client.submit(fail, i) for i in range(10)])
    except ValueError:
        exception_seen = True
    assert exception_seen
    for ix, item in enumerate(futures):
        if ix % 2 == 0:
            assert item.exception() is not None
        else:
            assert item.done()


def create_variable(shape, fill_value=None):
    """Create a variable."""
    data = numpy.ones(shape, dtype='uint8')
    if fill_value is not None:
        data[:, 0] = fill_value
    return dataset.Variable(name='var',
                            data=data,
                            dimensions=('x', 'y'),
                            fill_value=fill_value,
                            attrs=[
                                dataset.Attribute('a', 1),
                                dataset.Attribute('b', 2),
                                dataset.Attribute('long_name', 'long name')
                            ])


def create_dataset(shape):
    """Create a dataset."""
    return dataset.Dataset([create_variable(shape)],
                           attrs=[
                               dataset.Attribute('a', 1),
                               dataset.Attribute('b', 2),
                               dataset.Attribute('long_name', 'long name')
                           ])


def test_write_attrs(
        local_fs,  # pylint: disable=redefined-outer-name
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the write_attrs function."""
    var = create_variable((10, 2))
    path = local_fs.root.joinpath('var')
    local_fs.mkdir(str(path))
    storage.write_zattrs(str(local_fs.root), var, local_fs.fs)
    path = str(path.joinpath(storage.ZATTRS))
    expected = [
        b'{\n',
        b'  "a": 1,\n',
        b'  "b": 2,\n',
        b'  "long_name": "long name",\n',
        b'  "_ARRAY_DIMENSIONS": [\n',
        b'    "x",\n',
        b'    "y"\n',
        b'  ]\n',
        b'}',
    ]
    if platform.platform().startswith('Windows'):
        expected = [item.replace(b'\n', b'\r\n') for item in expected]
    assert local_fs.exists(path)
    with local_fs.open(path) as stream:
        lines = stream.readlines()
        assert lines == expected
    assert local_fs.exists(str(local_fs.root.joinpath('var', storage.ZATTRS)))


def test_write_variable(
        local_fs,  # pylint: disable=redefined-outer-name
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the write_variable function."""
    var = create_variable((1024, 1024))
    storage.write_zarr_variable(('var', var), str(local_fs.root), local_fs.fs)
    path = str(local_fs.root.joinpath('var'))
    assert local_fs.exists(path)
    mapper = local_fs.get_mapper(path)
    zarray = zarr.open(mapper)
    assert zarray.shape == (1024, 1024)
    assert numpy.all(zarray[...] == 1)

    other = storage.open_zarr_array(zarray, 'var')  # type:ignore
    assert other.metadata() == var.metadata()
    assert numpy.all(other.values == var.values)


def test_write_zarr_group(
        local_fs,  # pylint: disable=redefined-outer-name
        dask_client,  # pylint: disable=redefined-outer-name
):
    """Test the write_zarr_group function."""
    ds = create_dataset((1024, 1024))
    # memory fs does not support multi-processes
    future = dask_client.submit(storage.write_zarr_group,
                                dask_client.scatter(ds), str(local_fs.root),
                                local_fs.fs, sync.NoSync())
    future.result()
    mapper = local_fs.get_mapper(str(local_fs.root))
    zarray = zarr.open_group(mapper)
    assert numpy.all(zarray['var'][...] == 1)
    assert zarray.attrs['a'] == 1
    assert zarray.attrs['b'] == 2
    assert zarray.attrs['long_name'] == 'long name'
    assert zarray['var'].attrs['_ARRAY_DIMENSIONS'] == ['x', 'y']

    other = storage.open_zarr_group(str(local_fs.root), local_fs.fs)
    assert other.metadata() == ds.metadata()


def test_update_zarr_array(
        local_fs,  # pylint: disable=redefined-outer-name
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the update_zarr_array function."""
    var = create_variable((1024, 1024), fill_value=10)
    storage.write_zarr_variable(('var', var), str(local_fs.root), local_fs.fs)
    path = str(local_fs.root.joinpath('var'))
    storage.update_zarr_array(path, dask.array.full((1024, 1024), 2),
                              local_fs.fs)
    mapper = local_fs.get_mapper(path)
    zarray = zarr.open(mapper)
    assert numpy.all(zarray[...] == 2)
    data = numpy.full((1024, 1024), 2)
    data[:, 0] = 5
    data = numpy.ma.masked_equal(data, 5)
    assert numpy.all(data[:, 0].mask)
    storage.update_zarr_array(path, data, local_fs.fs)
    zarray = zarr.open(mapper)
    assert numpy.all(zarray[:, 0] == 10)


def test_del_zarr_array(
        local_fs,  # pylint: disable=redefined-outer-name
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the del_zarr_array function."""
    var = create_variable((1024, 1024))
    root = str(local_fs.root)
    storage.write_zarr_variable(('var', var), root, local_fs.fs)
    storage.del_zarr_array(root, 'var', local_fs.fs)
    assert not local_fs.exists(str(local_fs.root.joinpath('var')))


def test_add_zarr_array(
        local_fs,  # pylint: disable=redefined-outer-name
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the add_zarr_array function."""
    var = create_variable((1024, 1024), fill_value=10)
    root = str(local_fs.root)
    var.name = 'var1'
    storage.write_zarr_variable(('var1', var), root, local_fs.fs)
    var.name = 'var2'
    storage.add_zarr_array(root, var.metadata(), 'var1', local_fs.fs)
    mapper = local_fs.get_mapper(root)
    zarray = zarr.open(mapper)
    assert numpy.all(zarray['var2'][...] == 10)
