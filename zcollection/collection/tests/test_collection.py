# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of the collections
=======================
"""
from __future__ import annotations

import concurrent.futures
import datetime
import io

import dask.array.core
import dask.distributed
import fsspec
import numpy
import pytest
import zarr

from ... import (
    collection,
    convenience,
    dataset,
    merging,
    meta,
    partitioning,
    storage,
    sync,
    variable,
)
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ...tests.data import (
    DELTA,
    END_DATE,
    FILE_SYSTEM_DATASET,
    START_DATE,
    create_test_collection,
    create_test_dataset,
    create_test_dataset_with_fillvalue,
)
from ...tests.fixture import dask_arrays, numpy_arrays
from ...tests.fs import local_fs, s3, s3_base, s3_fs

# pylint: disable=unused-import


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_collection_creation(
    fs,
    request,
) -> None:
    """Test the creation of a collection."""
    tested_fs = request.getfixturevalue(fs)
    zds = next(create_test_dataset(False))
    zcollection = collection.Collection(
        axis='time',
        ds=zds.metadata(),
        partition_handler=partitioning.Date(('time', ), 'D'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs)
    assert isinstance(str(zcollection), str)
    assert zcollection.immutable is False

    serialized = collection.Collection.from_config(str(tested_fs.collection),
                                                   filesystem=tested_fs.fs)
    assert serialized.axis == zcollection.axis
    assert serialized.metadata == zcollection.metadata
    assert serialized.partition_properties == zcollection.partition_properties
    assert serialized.partitioning.get_config(
    ) == zcollection.partitioning.get_config()

    with pytest.raises(ValueError):
        collection.Collection.from_config(str(tested_fs.collection.parent))

    with pytest.raises(ValueError):
        collection.Collection('time_tai', zds.metadata(),
                              partitioning.Date(('time', ), 'D'),
                              str(tested_fs.collection))

    with pytest.raises(ValueError):
        collection.Collection('time', zds.metadata(),
                              partitioning.Date(('time_tai', ), 'D'),
                              str(tested_fs.collection))

    with pytest.raises(ValueError):
        collection.Collection(
            axis='time',
            ds=zds.metadata(),
            mode='X',  # type: ignore[arg-type]
            partition_handler=partitioning.Date(('time', ), 'D'),
            partition_base_dir=str(tested_fs.collection),
            filesystem=tested_fs.fs)


# pylint: disable=too-many-statements
@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_insert(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arrays_type,
    fs,
    request,
    tmpdir,
) -> None:
    """Test the insertion of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    datasets = list(create_test_dataset(delayed=False))
    zcollection = collection.Collection('time',
                                        datasets[0].metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs,
                                        synchronizer=sync.ProcessSync(
                                            str(tmpdir / 'lock.lck')))

    indices = numpy.arange(0, len(datasets))
    numpy.random.shuffle(indices)
    for idx in indices:
        zcollection.insert(datasets[idx],
                           merge_callable=merging.merge_time_series)

    data = zcollection.load(delayed=delayed)
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    # Adding same datasets once more (should not change anything)
    for idx in indices[:5]:
        zcollection.insert(datasets[idx])

    assert list(zcollection.partitions()) == sorted(
        list(zcollection.partitions()))

    data = zcollection.load(delayed=delayed)
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    values = data.variables['var1'].values
    numpy.all(values == numpy.vstack((numpy.arange(values.shape[0]), ) *
                                     values.shape[1]).T)

    values = data.variables['var2'].values
    numpy.all(values == numpy.vstack((numpy.arange(values.shape[0]), ) *
                                     values.shape[1]).T)

    data = zcollection.load(delayed=delayed, filters='year == 2020')
    assert data is None

    data = zcollection.load(delayed=delayed, filters='year == 2000')
    assert data is not None
    assert data.variables['time'].shape[0] == 61

    data = zcollection.load(delayed=delayed,
                            filters='year == 2000 and month == 4')
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[M]') == numpy.datetime64('2000-04-01'))

    data = zcollection.load(
        delayed=delayed, filters='year == 2000 and month == 4 and day == 15')
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[D]') == numpy.datetime64('2000-04-15'))

    data = zcollection.load(
        delayed=delayed,
        filters='year == 2000 and month == 4 and day in range(5, 25)')
    assert data is not None
    data = zcollection.load(delayed=delayed,
                            filters=lambda keys: datetime.date(2000, 4, 5) <=
                            datetime.date(keys['year'], keys['month'], keys[
                                'day']) <= datetime.date(2000, 4, 24))
    assert data is not None
    dates = data.variables['time'].values.astype('datetime64[D]')
    assert dates.min() == numpy.datetime64('2000-04-06')
    assert dates.max() == numpy.datetime64('2000-04-24')

    for path, item in zcollection.iterate_on_records(relative=True):
        assert isinstance(path, str)
        assert isinstance(item, zarr.Group)

    zcollection = convenience.open_collection(str(tested_fs.collection),
                                              mode='r',
                                              filesystem=tested_fs.fs)
    zds = zcollection.load(delayed=delayed, selected_variables=['var1'])
    assert zds is not None
    assert 'var1' in zds.variables
    assert 'var2' not in zds.variables

    zds = zcollection.load(delayed=delayed, selected_variables=[])
    assert zds is not None
    assert len(zds.variables) == 0

    zds = zcollection.load(delayed=delayed, selected_variables=['varX'])
    assert zds is not None
    assert len(zds.variables) == 0

    # pylint: enable=too-many-statements


@pytest.mark.parametrize('fs,create_test_data', FILE_SYSTEM_DATASET)
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_update(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    create_test_data,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = next(create_test_data(delayed=delayed))
    zcollection = collection.Collection('time',
                                        zds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    zcollection.insert(zds)

    def update(zds: dataset.Dataset, shift: int = 3):
        """Update function used for this test."""
        return {'var2': zds.variables['var1'].values * -1 + shift}

    zcollection.update(update, delayed=delayed)  # type: ignore

    data = zcollection.load()
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 3,
                          rtol=0)

    zcollection.update(
        update,  # type: ignore
        delayed=delayed,
        depth=1,
        shift=5)

    data = zcollection.load(delayed=delayed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 5,
                          rtol=0)

    # Test case if the selected variables does not contains the variable
    # to update.
    zcollection.update(
        update,  # type: ignore
        delayed=delayed,
        selected_variables=['var1'],
        depth=1,
        shift=5)

    data = zcollection.load(delayed=delayed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 5,
                          rtol=0)

    def update_with_info(zds: dataset.Dataset, partition_info=None, shift=3):
        """Update function used for this test."""
        assert partition_info is not None
        assert isinstance(partition_info, tuple)
        assert len(partition_info) == 2
        assert isinstance(partition_info[0], str)
        assert isinstance(partition_info[1], slice)
        assert partition_info[0] == 'num_lines'
        return {'var2': zds.variables['var1'].values * -1 + shift}

    zcollection.update(
        update_with_info,  # type: ignore
        delayed=delayed,
        depth=1,
        shift=10)

    data = zcollection.load(delayed=delayed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 10,
                          rtol=0)

    def update_and_trim(zds: dataset.Dataset, partition_info=None):
        """Update function used for this test."""
        assert partition_info is not None
        assert partition_info[0] == 'num_lines'
        zds = zds.isel(dict((partition_info, )))
        return {'var2': zds.variables['var1'].values * -1}

    zcollection.update(
        update_and_trim,  # type: ignore
        delayed=delayed,
        trim=False,
        depth=1)

    data = zcollection.load(delayed=delayed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1,
                          rtol=0)

    def invalid_var_name(zds: dataset.Dataset):
        """Update function used to test if the user wants to update a non-
        existent variable name."""
        return {'var99': zds.variables['var1'].values * -1 + 3}

    with pytest.raises(ValueError):
        zcollection.update(invalid_var_name)  # type: ignore


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_drop_partitions(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs, delayed=False)

    all_partitions = list(zcollection.partitions())
    assert 'month=01' in [
        item.split(zcollection.fs.sep)[-2] for item in all_partitions
    ]

    zcollection.drop_partitions(filters='year == 2000 and month==1')
    partitions = list(zcollection.partitions())
    assert 'month=01' not in [
        item.split(zcollection.fs.sep)[-2] for item in partitions
    ]

    npartitions = len(partitions)
    zcollection.drop_partitions(timedelta=datetime.timedelta(days=1))
    partitions = list(zcollection.partitions())
    assert len(partitions) == npartitions

    zcollection.drop_partitions(timedelta=datetime.timedelta(0))
    partitions = list(zcollection.partitions())
    assert len(partitions) == 0

    zcollection = convenience.open_collection(str(tested_fs.collection),
                                              mode='r',
                                              filesystem=tested_fs.fs)
    with pytest.raises(io.UnsupportedOperation):
        zcollection.drop_partitions()


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_drop_variable(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
) -> None:
    """Test the dropping of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs, delayed=False)

    with pytest.raises(ValueError):
        zcollection.drop_variable('time')
    zcollection.drop_variable('var1')

    with pytest.raises(ValueError):
        zcollection.drop_variable('var1')

    zds = zcollection.load(delayed=False)
    assert zds is not None
    assert 'var1' not in zds.variables

    zcollection = convenience.open_collection(str(tested_fs.collection),
                                              mode='r',
                                              filesystem=tested_fs.fs)
    with pytest.raises(io.UnsupportedOperation):
        zcollection.drop_partitions()


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_add_variable(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
) -> None:
    """Test the adding of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs, delayed=False)

    # Variable already exists
    new = meta.Variable(name='time',
                        dtype=numpy.dtype('float64'),
                        dimensions=('time', ))
    with pytest.raises(ValueError):
        zcollection.add_variable(new)

    # Variable doesn't use the partitioning dimension.
    new = meta.Variable(name='x',
                        dtype=numpy.dtype('float64'),
                        dimensions=('x', ))
    with pytest.raises(ValueError):
        zcollection.add_variable(new)

    # Variable doesn't use the dataset dimension.
    new = meta.Variable(name='x',
                        dtype=numpy.dtype('float64'),
                        dimensions=('time', 'x'))
    with pytest.raises(ValueError):
        zcollection.add_variable(new)

    new = meta.Variable(
        name='var3',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    zcollection.add_variable(new)

    assert new.name in zcollection.metadata.variables

    # Testing the configuration update by reopening the collection
    zcollection = collection.Collection.from_config(path=str(
        tested_fs.collection),
                                                    filesystem=tested_fs.fs)

    assert new.name in zcollection.metadata.variables

    zds = zcollection.load(delayed=False)
    assert zds is not None
    values = zds.variables['var3'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.all(values.mask)  # type: ignore


@pytest.mark.parametrize('fs,create_test_data', FILE_SYSTEM_DATASET)
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_add_update(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    create_test_data,
    request,
) -> None:
    """Test the adding and updating of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = next(create_test_data(delayed=delayed))
    zcollection = collection.Collection('time',
                                        zds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    zcollection.insert(zds)

    new1 = meta.Variable(name='var3',
                         dtype=numpy.dtype('float64'),
                         dimensions=('num_lines', 'num_pixels'),
                         attrs=(dataset.Attribute(name='attr', value=1), ),
                         fill_value=1000000.5)

    new2 = meta.Variable(
        name='var4',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    zcollection.add_variable(new1)
    zcollection.add_variable(new2)

    data = zcollection.load(delayed=delayed)
    assert data is not None

    def update_1(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 201.5}

    def update_2(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values // 5}

    zcollection.update(update_1, new1.name, delayed=delayed)  # type: ignore
    zcollection.update(update_2, new2.name, delayed=delayed)  # type: ignore

    if delayed is False:
        # If the dataset is not delayed, we need to reload it.
        data = zcollection.load(delayed=False)
        assert data is not None

    assert numpy.allclose(data.variables[new1.name].values,
                          data.variables['var1'].values * 201.5,
                          rtol=0)
    assert numpy.allclose(data.variables[new2.name].values,
                          data.variables['var1'].values // 5,
                          rtol=0)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_fillvalue(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    request,
) -> None:
    """Test the management of masked values."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcollection = create_test_collection(tested_fs,
                                         delayed=delayed,
                                         with_fillvalue=True)

    # Load the dataset written with masked values in the collection and
    # compare it to the original dataset.
    data = zcollection.load(delayed=delayed)
    assert data is not None

    zds = next(create_test_dataset_with_fillvalue(delayed=delayed))

    values = data.variables['var1'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.ma.allclose(zds.variables['var1'].values, values)

    values = data.variables['var2'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.ma.allclose(zds.variables['var2'].values, values)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_degraded_tests(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
) -> None:
    """Test the degraded functionality."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

    fake_ds = next(create_test_dataset())
    fake_ds.variables['var3'] = fake_ds.variables['var1']
    fake_ds.variables['var3'].name = 'var3'

    with pytest.raises(ValueError):
        zcollection.insert(fake_ds)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_insert_with_missing_variable(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
) -> None:
    """Test of the insertion of a dataset in which a variable is missing.

    This happens, for example, when a variable is not acquired, but
    created by an algorithm.
    """
    tested_fs = request.getfixturevalue(arg)
    zds = next(create_test_dataset_with_fillvalue()).to_xarray()
    zcollection = convenience.create_collection(
        axis='time',
        ds=zds,
        partition_handler=partitioning.Date(('time', ), 'M'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs)
    zcollection.insert(zds, merge_callable=merging.merge_time_series)

    zds = next(create_test_dataset_with_fillvalue())
    zds.drops_vars('var1')
    zcollection.insert(zds)

    data = zcollection.load()
    assert data is not None
    assert numpy.ma.allequal(
        data.variables['var1'].values,
        numpy.ma.masked_equal(
            numpy.full(zds.variables['var1'].shape,
                       zds.variables['var1'].fill_value,
                       zds.variables['var1'].dtype),
            zds.variables['var1'].fill_value))


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_insert_failed(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    request,
) -> None:
    """Test the insertion of a dataset in which the insertion failed."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = next(create_test_dataset(delayed=delayed))
    zcollection = collection.Collection('time',
                                        zds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    partitions = list(zcollection.partitioning.split_dataset(zds, 'time'))

    # Create a file in the directory where the dataset should be written. This
    # should cause the insertion to fail.
    sep = zcollection.fs.sep
    one_directory = sep.join((zcollection.partition_properties.dir, ) +
                             partitions[0][0])
    zcollection.fs.makedirs(sep.join(one_directory.split(sep)[:-1]))
    zcollection.fs.touch(one_directory)

    with pytest.raises((OSError, ValueError)):
        zcollection.insert(zds)

    # Because the insert failed, the partition that was supposed to be created
    # was deleted.
    assert not zcollection.fs.exists(one_directory)
    zcollection.insert(zds)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_insert_validation(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
) -> None:
    """Test the insertion of a dataset with metadata validation."""
    tested_fs = request.getfixturevalue(arg)
    zds = next(create_test_dataset_with_fillvalue())

    zcollection = convenience.create_collection(
        axis='time',
        ds=zds,
        partition_handler=partitioning.Date(('time', ), 'M'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs)
    zcollection.insert(zds, merge_callable=merging.merge_time_series)

    zds = next(create_test_dataset_with_fillvalue())

    # Inserting a dataset containing valid attributes
    zcollection.insert(zds)

    # Inserting a dataset containing an invalid attributes
    zds = next(create_test_dataset_with_fillvalue())
    zds.attrs = (meta.Attribute('invalid', 1), )

    with pytest.raises(ValueError):
        zcollection.insert(zds, validate=True)

    # Inserting a dataset containing variables with invalid attributes
    zds = next(create_test_dataset_with_fillvalue())

    for var in zds.variables.values():
        var.attrs = (meta.Attribute('invalid', 1), )

    with pytest.raises(ValueError):
        zcollection.insert(zds, validate=True)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_map_partition(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcollection = create_test_collection(tested_fs, delayed=delayed)

    result = zcollection.map(
        lambda x: x.variables['var1'].values * 2,  # type: ignore
        delayed=delayed)
    for item in result.compute():
        folder = zcollection.fs.sep.join(
            (zcollection.partition_properties.dir,
             zcollection.partitioning.join(item[0], zcollection.fs.sep)))
        zds = storage.open_zarr_group(folder, zcollection.fs, delayed=False)
        assert numpy.allclose(item[1], zds.variables['var1'].values * 2)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_indexer(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcollection = create_test_collection(tested_fs, delayed=delayed)

    indexers = zcollection.map(
        lambda x: slice(0, x.dimensions['num_lines'])  # type: ignore
    ).compute()
    zds1 = zcollection.load(indexer=indexers, delayed=delayed)
    assert zds1 is not None
    zds2 = zcollection.load(delayed=delayed)
    assert zds2 is not None

    assert numpy.allclose(zds1.variables['var1'].values,
                          zds2.variables['var1'].values)


def test_variables() -> None:
    """Test the listing of the variables in a collection."""
    fs = fsspec.filesystem('memory')
    zds = next(create_test_dataset(delayed=False))
    zcollection = collection.Collection(axis='time',
                                        ds=zds.metadata(),
                                        partition_handler=partitioning.Date(
                                            ('time', ), 'D'),
                                        partition_base_dir='/',
                                        filesystem=fs)
    variables = zcollection.variables()
    assert isinstance(variables, tuple)
    assert len(variables) == 3
    assert variables[0].name == 'time'
    assert variables[1].name == 'var1'
    assert variables[2].name == 'var2'

    variables = zcollection.variables(('time', ))
    assert len(variables) == 1
    assert variables[0].name == 'time'


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_map_overlap(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    request,
) -> None:
    """Test the map overlap method."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcollection = create_test_collection(tested_fs, delayed=delayed)

    def func(zds: dataset.Dataset, partition_info: tuple[str, slice]):
        assert partition_info[0] == 'num_lines'
        var = zds.variables['var1']
        slices = [slice(None)] * len(var.dimensions)
        slices[var.dimensions.index(partition_info[0])] = partition_info[1]
        return var.values[tuple(slices)] * 2

    result = zcollection.map_overlap(
        func,  # type: ignore
        delayed=delayed,
        depth=1)

    for partition, data in result.compute():
        folder = zcollection.fs.sep.join(
            (zcollection.partition_properties.dir,
             zcollection.partitioning.join(partition, zcollection.fs.sep)))
        zds = storage.open_zarr_group(folder, zcollection.fs, delayed=False)
        assert numpy.allclose(data, zds.variables['var1'].values * 2)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_insert_immutable(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
) -> None:
    """Test the insertion of a dataset with variables that are immutable
    relative to the partitioning."""
    tested_fs = request.getfixturevalue(arg)

    zds_reference = dataset.Dataset(
        [
            dataset.Array(
                'time',
                numpy.arange(numpy.datetime64('2000-01-01'),
                             numpy.datetime64('2000-01-30'),
                             numpy.timedelta64(1, 'D')),
                ('time', ),
            ),
            dataset.Array(
                'lon',
                numpy.arange(0, 360, 1),
                ('lon', ),
            ),
            dataset.Array(
                'lat',
                numpy.arange(-90, 90, 1),
                ('lat', ),
            ),
            dataset.Array(
                'grid',
                numpy.random.rand(
                    29,
                    360,
                    180,
                ),
                ('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcollection = collection.Collection('time',
                                        zds_reference.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    assert zcollection.immutable
    assert not tested_fs.fs.exists(zcollection._immutable)
    zcollection.insert(zds_reference)
    assert tested_fs.fs.exists(zcollection._immutable)

    zds = zcollection.load(delayed=False)
    assert zds is not None

    assert numpy.all(
        zds.variables['grid'].values == zds_reference.variables['grid'].values)
    assert numpy.all(
        zds.variables['time'].values == zds_reference.variables['time'].values)
    assert numpy.all(
        zds.variables['lon'].values == zds_reference.variables['lon'].values)
    assert numpy.all(
        zds.variables['lat'].values == zds_reference.variables['lat'].values)

    def update(zds: dataset.Dataset, varname: str) -> dict[str, numpy.ndarray]:
        """Update function used for this test."""
        return {varname: zds.variables['grid'].values * -1}

    zcollection.update(update, delayed=False, varname='grid')  # type: ignore
    zds = zcollection.load(delayed=False)
    assert zds is not None

    assert numpy.all(zds.variables['grid'].values ==
                     zds_reference.variables['grid'].values * -1)
    assert numpy.all(
        zds.variables['time'].values == zds.variables['time'].values)
    assert numpy.all(
        zds.variables['lon'].values == zds.variables['lon'].values)
    assert numpy.all(
        zds.variables['lat'].values == zds.variables['lat'].values)

    new_variable = meta.Variable(
        'new_var',
        numpy.float64,
        dimensions=('time', 'lon', 'lat'),
        attrs=(meta.Attribute('units', 'm'), ),
    )
    zcollection.add_variable(new_variable)
    zcollection.update(update, varname='new_var')  # type: ignore
    zds = zcollection.load()
    assert zds is not None
    assert numpy.all(zds.variables['new_var'].values ==
                     zds_reference.variables['grid'].values)

    new_variable = meta.Variable(
        'new_var2',
        numpy.float64,
        dimensions=('another_dim', ),
        attrs=(meta.Attribute('units', 'm'), ),
    )
    with pytest.raises(ValueError):
        zcollection.add_variable(new_variable)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_copy_collection(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
        arg,
        request,
        tmpdir) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

    target = str(tmpdir / 'copy')
    zcopy = zcollection.copy(target, filesystem=fsspec.filesystem('file'))

    ds_before_copy = zcollection.load()
    ds_after_copy = zcopy.load()

    assert ds_before_copy is not None
    assert ds_after_copy is not None

    assert numpy.all(ds_before_copy.variables['var1'].values ==
                     ds_after_copy.variables['var1'].values)
    assert numpy.all(ds_before_copy.variables['var2'].values ==
                     ds_after_copy.variables['var2'].values)


def _insert(zds: dataset.Dataset, base_dir: str, lock_file: str,
            scheduler_file: str) -> None:
    client = dask.distributed.Client(scheduler_file=scheduler_file)
    zcollection = collection.Collection.from_config(
        base_dir, mode='w', synchronizer=sync.ProcessSync(lock_file))
    zcollection.insert(zds, merge_callable=merging.merge_time_series)
    client.close()


# pylint: disable=too-many-statements
def test_concurrent_insert(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    tmpdir,
) -> None:
    """Test the insertion of a dataset."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset(delayed=False))
    zds = datasets[0]
    lock_file = str(tmpdir / 'lock.lck')
    synchronizer = sync.ProcessSync(lock_file)
    base_dir = str(tmpdir / 'test')
    zcollection = collection.Collection('time',
                                        zds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        base_dir,
                                        filesystem=fs,
                                        synchronizer=synchronizer)

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=32)
    futures = []

    assert zcollection.is_locked() is False

    indices = list(numpy.arange(0, len(datasets)))
    numpy.random.shuffle(indices)
    futures = [
        pool.submit(_insert, datasets[ix], base_dir, lock_file,
                    dask_client.scheduler_file) for ix in indices
    ]

    def update(zds: dataset.Dataset, shift: int = 3):
        """Update function used for this test."""
        return {'var2': zds.variables['var1'].values * -1 + shift}

    launch_update = True
    for item in concurrent.futures.as_completed(futures):
        assert item.exception() is None
        if launch_update:
            zcollection.update(update)  # type: ignore
            launch_update = False

    data = zcollection.load()
    assert data is not None
    values = data.variables['time'].values

    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))


def test_partition_modified(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
        tmpdir):
    """Test the loading of a variable that has been modified since its
    creation."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset())
    zds = datasets[0]
    base_dir = str(tmpdir / 'test')
    zcollection = collection.Collection('time',
                                        zds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        base_dir,
                                        filesystem=fs)
    zcollection.insert(zds)

    last_month = zcollection.load(filters=lambda keys: keys['year'] == 2000 and
                                  keys['month'] == 1 and keys['day'] == 16)
    assert last_month is not None

    zds = zcollection.load()
    assert zds is not None

    def new_shape(
        var: variable.Variable,
        selected_dim: str,
        new_size: int,
    ) -> tuple[int, ...]:
        """Compute the new shape of a variable."""
        return tuple(new_size if dim == selected_dim else size
                     for dim, size in zip(var.dimensions, var.shape))

    dim, size = 'num_lines', last_month.dimensions['num_lines'] * 25
    last_month = dataset.Dataset(
        [
            variable.DelayedArray(
                name,
                numpy.resize(var.array.compute(), new_shape(var, dim, size)),
                var.dimensions,
                attrs=var.attrs,
                compressor=var.compressor,
                fill_value=var.fill_value,
                filters=var.filters,
            ) for name, var in last_month.variables.items()
        ],
        attrs=zds.attrs,
    )
    zcollection.insert(last_month)

    with pytest.raises(RuntimeError, match='Try to re-load'):
        _ = zds['var2'].values

    with pytest.raises(RuntimeError, match='Try to re-load'):
        _ = zds['time'].values
