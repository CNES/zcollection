# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the storage module.
===========================
"""
from typing import Any, cast
import math
import time

import dask.array
import numpy
import pytest
import zarr

from .. import dataset, storage, store, sync
from ..fs_utils import join_path
from ..variable.abc import not_equal
from .cluster import dask_client, dask_cluster  # noqa: F401
from .fs import local_fs  # noqa: F401


def test_execute_transaction(
        dask_client,  # noqa: F811
) -> None:
    """Test the execute_transaction function."""
    # First case: no transaction to execute
    assert storage.execute_transaction(dask_client, []) is None

    # General case: execute a transaction without error
    def func():
        return 1

    assert sum(
        storage.execute_transaction(
            dask_client, [dask_client.submit(func) for i in range(10)])) == 10

    # All submitted tasks finish before a transaction raises.
    def fail(data):
        time.sleep((10 - data) / 100)
        if data % 2 == 0:
            raise ValueError('injected failure')
        return data

    futures = [dask_client.submit(fail, i) for i in range(10)]
    with pytest.raises(ValueError, match='injected failure'):
        storage.execute_transaction(dask_client, futures)
    assert all(future.done() for future in futures)


def create_variable(shape, fill_value=None) -> dataset.DelayedArray:
    """Create a variable."""
    data = numpy.ones(shape, dtype='uint8')
    if fill_value is not None:
        data[:, 0] = fill_value
    return dataset.DelayedArray(name='var',
                                data=data,
                                dimensions=('x', 'y'),
                                fill_value=fill_value,
                                attrs=[
                                    dataset.Attribute('a', 1),
                                    dataset.Attribute('b', 2),
                                    dataset.Attribute('long_name', 'long name')
                                ])


def create_dataset(shape) -> dataset.Dataset:
    """Create a dataset."""
    return dataset.Dataset([create_variable(shape)],
                           attrs=[
                               dataset.Attribute('a', 1),
                               dataset.Attribute('b', 2),
                               dataset.Attribute('long_name', 'long name')
                           ])


def test_variable_attributes() -> None:
    """A variable has the attributes it declares.

    Dimension names are stored in a separate array metadata field, not
    among the variable attributes.
    """
    var = create_variable((10, 2))

    assert storage.variable_attributes(var) == {
        'a': 1,
        'b': 2,
        'long_name': 'long name',
    }


@pytest.mark.parametrize(('chunks', 'chunks_expected'), [
    (None, (1024, 1024)),
    ({
        'x': -1,
        'y': -1
    }, (1024, 1024)),
    ({
        'x': -1,
        'y': 1024
    }, (1024, 1024)),
    ({
        'x': 1024,
        'y': -1
    }, (1024, 1024)),
    ({
        'x': 'auto',
        'y': 'auto'
    }, (1024, 1024)),
    ({
        'x': 512,
        'y': 'auto'
    }, (512, 1024)),
    ({
        'x': 256,
        'y': 512
    }, (256, 512)),
    ({
        'x': 53,
        'y': 826
    }, (53, 826)),
])
def test_write_variable(
    local_fs,  # noqa: F811
    chunks,
    chunks_expected,
) -> None:
    """Test the write_variable function."""
    dim_size = 1024
    var = create_variable((dim_size, dim_size))
    storage.write_zarr_variable(('var', var),
                                dirname=str(local_fs.root),
                                fs=local_fs.fs,
                                chunks=chunks)
    path = str(local_fs.root.joinpath('var'))
    assert local_fs.exists(path)
    mapper = local_fs.get_mapper(path)
    zarray = zarr.open_array(mapper, mode='a')
    assert zarray.shape == (dim_size, dim_size)
    assert numpy.all(zarray[...] == 1)

    chunks_number = math.ceil(dim_size / chunks_expected[0]) * math.ceil(
        dim_size / chunks_expected[1])
    assert zarray.chunks == chunks_expected
    assert zarray.nchunks == chunks_number

    other = storage.open_zarr_array(
        zarray,
        'var',
    )
    assert other.metadata() == var.metadata()
    assert numpy.all(other.values == var.values)


@pytest.mark.parametrize(
    ('block_size', 'chunks', 'blocks_number'),
    [
        # Everything fit in one block
        (1048756, None, 1),
        # Keep dimensions ``x`` and ``y`` together in one block.
        (1048756 // 20, {
            'x': -1,
            'y': -1
        }, 1),
        # Halve the block size and split into two blocks along ``x``.
        (1048756 // 2, {
            'x': 512,
            'y': 'auto'
        }, 2),
        # Halve the block size and split into two blocks along ``y``.
        (1048756 // 2, {
            'x': 1024,
            'y': 'auto'
        }, 2),
    ])
def test_write_variable_block_size_limit(
    local_fs,  # noqa: F811
    block_size,
    chunks,
    blocks_number,
) -> None:
    """Test the write_variable function with different block size limits."""
    dim_size = 1024
    var = create_variable((dim_size, dim_size))
    storage.write_zarr_variable(('var', var),
                                dirname=str(local_fs.root),
                                fs=local_fs.fs,
                                chunks=chunks,
                                block_size_limit=block_size)
    path = str(local_fs.root.joinpath('var'))
    mapper = local_fs.get_mapper(path)
    zarray = zarr.open_array(mapper, mode='a')

    assert zarray.nchunks == blocks_number


def test_write_zarr_group(
        local_fs,  # noqa: F811
        dask_client,  # noqa: F811
) -> None:
    """Test the write_zarr_group function."""
    zds = create_dataset((1024, 1024))
    # The memory file system does not support multiprocessing.
    future = dask_client.submit(storage.write_zarr_group,
                                dask_client.scatter(zds), str(local_fs.root),
                                local_fs.fs, sync.NoSync())
    future.result()
    mapper = local_fs.get_mapper(str(local_fs.root))
    zarray = zarr.open_group(mapper)
    assert numpy.all(cast('zarr.Array', zarray['var'])[...] == 1)
    assert zarray.attrs['a'] == 1
    assert zarray.attrs['b'] == 2
    assert zarray.attrs['long_name'] == 'long name'
    metadata = cast('zarr.core.metadata.ArrayV3Metadata',
                    cast('zarr.Array', zarray['var']).metadata)
    assert metadata.dimension_names == ('x', 'y')

    other = storage.open_zarr_group(str(local_fs.root), local_fs.fs)
    assert other.metadata() == zds.metadata()


def test_write_meta_checks_the_group_holds_the_variables(
        local_fs,  # noqa: F811
) -> None:
    """A group that lists none of the variables written to it is an error."""
    zds = create_dataset((16, 16))
    path = str(local_fs.root)
    store.create_group(local_fs.fs, path)

    #: pylint: disable=protected-access
    with pytest.raises(OSError, match='does not hold the variables'):
        storage._write_meta(zds, path, local_fs.fs)
    #: pylint: enable=protected-access


@pytest.mark.parametrize(('chunks', 'chunks_expected'), [
    (None, (1024, 1024)),
    ({
        'x': -1,
        'y': 1024
    }, (1024, 1024)),
    ({
        'x': 'auto',
        'y': 'auto'
    }, (1024, 1024)),
    ({
        'x': 512,
        'y': 'auto'
    }, (512, 1024)),
])
def test_update_zarr_array(
    local_fs,  # noqa: F811
    chunks,
    chunks_expected,
) -> None:
    """Test the update_zarr_array function."""
    dim_size = 1024

    var = create_variable((dim_size, dim_size), fill_value=10)
    storage.write_zarr_variable(('var', var),
                                str(local_fs.root),
                                local_fs.fs,
                                chunks=chunks)
    path = str(local_fs.root.joinpath('var'))
    storage.update_zarr_array(path, dask.array.full((dim_size, dim_size), 2),
                              local_fs.fs)
    mapper = local_fs.get_mapper(path)
    zarray = zarr.open_array(mapper, mode='a')
    assert numpy.all(zarray[...] == 2)
    data: Any = numpy.full((dim_size, dim_size), 2)
    data[:, 0] = 5
    data = numpy.ma.masked_equal(data, 5)
    assert numpy.all(data[:, 0].mask)
    storage.update_zarr_array(path, data, local_fs.fs)
    zarray = zarr.open_array(mapper, mode='a')
    assert numpy.all(zarray[:, 0] == 10)

    chunks_number = math.ceil(dim_size / chunks_expected[0]) * math.ceil(
        dim_size / chunks_expected[1])
    assert zarray.chunks == chunks_expected
    assert zarray.nchunks == chunks_number


def test_del_zarr_array(
        local_fs,  # noqa: F811
) -> None:
    """Test the del_zarr_array function."""
    var = create_variable((1024, 1024))
    root = str(local_fs.root)
    storage.write_zarr_variable(('var', var), root, local_fs.fs)
    storage.del_zarr_array(root, 'var', local_fs.fs)
    assert not local_fs.exists(str(local_fs.root.joinpath('var')))


@pytest.mark.parametrize(('chunks', 'chunks_expected'), [
    (None, (1024, 1024)),
    ({
        'x': 1024,
        'y': 1024
    }, (1024, 1024)),
    ({
        'x': -1,
        'y': -1
    }, (1024, 1024)),
    ({
        'x': 512,
        'y': 256
    }, (512, 256)),
    ({
        'x': -1,
        'y': 128
    }, (1024, 128)),
])
def test_add_zarr_array(
    local_fs,  # noqa: F811
    chunks,
    chunks_expected,
) -> None:
    """Test the add_zarr_array function."""
    dim_size = 1024
    var = create_variable((dim_size, dim_size), fill_value=10)
    root = str(local_fs.root)
    var.name = 'var1'
    storage.write_zarr_variable(('var1', var),
                                root,
                                local_fs.fs,
                                chunks=chunks)
    var.name = 'var2'
    storage.add_zarr_array(dirname=root,
                           variable=var.metadata(),
                           dimensions={
                               'x': dim_size,
                               'y': dim_size
                           },
                           fs=local_fs.fs,
                           chunks=chunks)
    mapper = local_fs.get_mapper(root)
    zarray = zarr.open_group(mapper, mode='a')
    assert numpy.all(cast('zarr.Array', zarray['var2'])[...] == 10)

    chunks_number = math.ceil(dim_size / chunks_expected[0]) * math.ceil(
        dim_size / chunks_expected[1])

    for varname in ['var1', 'var2']:
        var_array = cast('zarr.Array', zarray[varname])
        assert var_array.chunks == chunks_expected
        assert var_array.nchunks == chunks_number

    var.name = 'var3'
    var.fill_value = 30
    with pytest.raises(ValueError, match='unknown dimensions'):
        storage.add_zarr_array(dirname=root,
                               variable=var.metadata(),
                               dimensions={
                                   'x': dim_size,
                               },
                               fs=local_fs.fs,
                               chunks=chunks)

    # We'll try to read 'y' from existing data and generate a zarr
    # exception
    with pytest.raises(Exception, match=r'.*'):
        storage.add_zarr_array(dirname=root,
                               variable=var.metadata(),
                               dimensions={
                                   'x': dim_size,
                               },
                               fs=local_fs.fs,
                               axis='y',
                               chunks=chunks)


@pytest.mark.parametrize(('dtype', 'fill_value', 'expected'), [
    ('uint8', None, None),
    ('uint8', 255, 255),
    ('float64', -999.0, -999.0),
    ('float64', numpy.nan, 'NaN'),
])
def test_variable_attributes_fill_value(
    dtype,
    fill_value,
    expected,
) -> None:
    """The declared fill value is stored in encoded form."""
    var = dataset.DelayedArray(name='var',
                               data=numpy.ones((4, ), dtype=dtype),
                               dimensions=('x', ),
                               fill_value=fill_value)

    attrs = storage.variable_attributes(var)

    if expected is None:
        assert storage.FILL_VALUE not in attrs
    else:
        assert attrs[storage.FILL_VALUE] == expected


@pytest.mark.parametrize(('dtype', 'fill_value'), [
    ('float64', numpy.nan),
    ('float64', numpy.inf),
    ('float64', -999.0),
    ('datetime64[s]', numpy.datetime64('1970-01-01', 's')),
])
def test_a_written_variable_reads_back_its_fill_value(
    local_fs,  # noqa: F811
    dtype,
    fill_value,
) -> None:
    """A written variable retains its declared fill value.

    The fill-value attribute stores an encoded representation. For NaN,
    infinity and dates, that representation differs from the value itself.
    """
    data = numpy.zeros((4, ), dtype=dtype)
    data[0] = fill_value
    var = dataset.DelayedArray(name='var',
                               data=data,
                               dimensions=('x', ),
                               fill_value=fill_value)
    storage.write_zarr_variable(('var', var), str(local_fs.root), local_fs.fs)

    group = zarr.open_group(local_fs.get_mapper(str(local_fs.root)))
    read = dataset.DelayedArray.from_zarr(cast('zarr.Array', group['var']),
                                          'var')

    assert not_equal(read.fill_value, fill_value) is False


def test_a_group_reads_its_variables_in_order(
        local_fs,  # noqa: F811
        dask_client,  # noqa: F811
) -> None:
    """Group variables are read in name order.

    Their names are selected from a set, which may iterate in different
    orders in different processes.
    """
    zds = dataset.Dataset([
        dataset.DelayedArray(name=name,
                             data=numpy.ones((4, ), dtype='uint8'),
                             dimensions=('x', ))
        for name in ('zeta', 'alpha', 'mu')
    ])
    future = dask_client.submit(storage.write_zarr_group,
                                dask_client.scatter(zds), str(local_fs.root),
                                local_fs.fs, sync.NoSync())
    future.result()

    zds = storage.open_zarr_group(str(local_fs.root), local_fs.fs)

    assert tuple(zds.variables) == ('alpha', 'mu', 'zeta')


def test_a_failed_write_names_the_variable(
        local_fs,  # noqa: F811
) -> None:
    """A failed write reports the name of the variable.

    The write fails when its fill value cannot be encoded for its data type.
    """
    var = dataset.DelayedArray(name='ssh',
                               data=numpy.zeros((4, ), dtype='float32'),
                               dimensions=('x', ),
                               fill_value='not a number')

    with pytest.raises(Exception) as caught:  # noqa: PT011
        storage.write_zarr_variable(('ssh', var), str(local_fs.root),
                                    local_fs.fs)

    assert "while writing the variable 'ssh'" in caught.value.__notes__


@pytest.mark.parametrize('remove_partition', [False, True])
def test_read_partition_refreshes_cached_listings(s3_fs,
                                                  remove_partition) -> None:
    """Missing metadata is propagated; an absent partition is classified."""
    fs = s3_fs.fs
    dirname = join_path(str(s3_fs.collection), 'key=1')
    store.create_group(fs, dirname)
    store.create_array(fs,
                       dirname,
                       name='key',
                       shape=(1, ),
                       dtype='int64',
                       dimensions=('row', ))
    store.consolidate(fs, dirname)
    fs.ls(str(s3_fs.collection))

    external = type(fs)(**fs.storage_options, skip_instance_cache=True)
    external.rm(
        dirname if remove_partition else join_path(dirname, 'zarr.json'),
        recursive=remove_partition)
    assert fs.exists(dirname)

    with pytest.raises(FileNotFoundError) as caught:
        storage.read_partition(dirname, fs, delayed=False)

    assert isinstance(caught.value,
                      storage.AbsentPartition) == remove_partition


def test_an_absent_group_is_not_valid(
        local_fs,  # noqa: F811
) -> None:
    """An absent directory is not a valid Zarr group."""
    assert not storage.check_zarr_group(
        join_path(str(local_fs.root), 'absent'), local_fs.fs)


def test_an_absent_group_takes_no_variable(
        local_fs,  # noqa: F811
) -> None:
    """A variable cannot be added to a group that does not exist."""
    dirname = join_path(str(local_fs.root), 'absent')
    var = create_variable((4, 4), fill_value=10)
    var.name = 'var1'

    storage.add_zarr_array(dirname=dirname,
                           variable=var.metadata(),
                           dimensions={
                               'x': 4,
                               'y': 4
                           },
                           fs=local_fs.fs)

    assert not local_fs.fs.exists(dirname)
