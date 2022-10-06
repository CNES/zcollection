# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of the collections
=======================
"""
import datetime
import io

import fsspec
import numpy
import pytest
import zarr

from .. import (
    collection,
    convenience,
    dataset,
    merging,
    meta,
    partitioning,
    storage,
)
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster
from .data import (
    DELTA,
    END_DATE,
    FILE_SYSTEM_DATASET,
    START_DATE,
    create_test_collection,
    create_test_dataset,
    create_test_dataset_with_fillvalue,
)
from .fs import local_fs, s3, s3_base, s3_fs

# pylint: disable=unused-import


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_collection_creation(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the creation of a collection."""
    tested_fs = request.getfixturevalue(arg)
    ds = next(create_test_dataset())
    zcollection = collection.Collection(
        axis='time',
        ds=ds.metadata(),
        partition_handler=partitioning.Date(('time', ), 'D'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs)
    assert isinstance(str(zcollection), str)
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
        collection.Collection('time_tai', ds.metadata(),
                              partitioning.Date(('time', ), 'D'),
                              str(tested_fs.collection))

    with pytest.raises(ValueError):
        collection.Collection('time', ds.metadata(),
                              partitioning.Date(('time_tai', ), 'D'),
                              str(tested_fs.collection))

    with pytest.raises(ValueError):
        collection.Collection(axis='time',
                              ds=ds.metadata(),
                              mode='X',
                              partition_handler=partitioning.Date(('time', ),
                                                                  'D'),
                              partition_base_dir=str(tested_fs.collection),
                              filesystem=tested_fs.fs)


# pylint: disable=too-many-statements
@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_insert(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the insertion of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    datasets = list(create_test_dataset())
    ds = datasets[0]
    zcollection = collection.Collection('time',
                                        ds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)

    indices = numpy.arange(0, len(datasets))
    numpy.random.shuffle(indices)
    for ix in indices:
        zcollection.insert(datasets[ix],
                           merge_callable=merging.merge_time_series)

    data = zcollection.load()
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    # Adding same datasets once more (should not change anything)
    for ix in indices[:5]:
        zcollection.insert(datasets[ix])

    assert list(zcollection.partitions()) == sorted(
        list(zcollection.partitions()))

    data = zcollection.load()
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    values = data.variables['var1'].values
    numpy.all(values == numpy.vstack((numpy.arange(values.shape[0]), ) *
                                     values.shape[1]).T)

    values = data.variables['var2'].values
    numpy.all(values == numpy.vstack((numpy.arange(values.shape[0]), ) *
                                     values.shape[1]).T)

    data = zcollection.load(filters='year == 2020')
    assert data is None

    data = zcollection.load(filters='year == 2000')
    assert data is not None
    assert data.variables['time'].shape[0] == 61

    data = zcollection.load(filters='year == 2000 and month == 4')
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[M]') == numpy.datetime64('2000-04-01'))

    data = zcollection.load(
        filters='year == 2000 and month == 4 and day == 15')
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[D]') == numpy.datetime64('2000-04-15'))

    data = zcollection.load(
        filters='year == 2000 and month == 4 and day in range(5, 25)')
    assert data is not None
    data = zcollection.load(filters=lambda keys: datetime.date(
        2000, 4, 5) <= datetime.date(keys['year'], keys['month'], keys['day'])
                            <= datetime.date(2000, 4, 24))
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
    ds = zcollection.load(selected_variables=['var1'])
    assert ds is not None
    assert 'var1' in ds.variables
    assert 'var2' not in ds.variables

    ds = zcollection.load(selected_variables=[])
    assert ds is not None
    assert len(ds.variables) == 0

    ds = zcollection.load(selected_variables=['varX'])
    assert ds is not None
    assert len(ds.variables) == 0

    # pylint: enable=too-many-statements


@pytest.mark.parametrize('arg,create_test_data', FILE_SYSTEM_DATASET)
def test_update(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    create_test_data,
    request,
):
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    ds = next(create_test_data())
    zcollection = collection.Collection('time',
                                        ds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    zcollection.insert(ds)

    data = zcollection.load()
    assert data is not None

    def update(ds: dataset.Dataset):
        """Update function used for this test."""
        return dict(var2=ds.variables['var1'].values * -1 + 3)

    zcollection.update(update)  # type: ignore

    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 3,
                          rtol=0)

    def invalid_var_name(ds: dataset.Dataset):
        """Update function used to test if the user wants to update a non-
        existent variable name."""
        return dict(var99=ds.variables['var1'].values * -1 + 3)

    with pytest.raises(ValueError):
        zcollection.update(invalid_var_name)  # type: ignore


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_drop_partitions(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

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
):
    """Test the dropping of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

    with pytest.raises(ValueError):
        zcollection.drop_variable('time')
    zcollection.drop_variable('var1')

    with pytest.raises(ValueError):
        zcollection.drop_variable('var1')

    ds = zcollection.load()
    assert ds is not None
    assert 'var1' not in ds.variables

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
):
    """Test the adding of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

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

    ds = zcollection.load()
    assert ds is not None
    values = ds.variables['var3'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.all(values.mask)  # type: ignore


@pytest.mark.parametrize('arg,create_test_data', FILE_SYSTEM_DATASET)
def test_add_update(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    create_test_data,
    request,
):
    """Test the adding and updating of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    ds = next(create_test_data())
    zcollection = collection.Collection('time',
                                        ds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    zcollection.insert(ds)

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

    data = zcollection.load()
    assert data is not None

    def update_1(ds, varname):
        """Update function used for this test."""
        return {varname: ds.variables['var1'].data * 201.5}

    def update_2(ds, varname):
        """Update function used for this test."""
        return {varname: ds.variables['var1'].data // 5}

    zcollection.update(update_1, new1.name)  # type: ignore
    zcollection.update(update_2, new2.name)  # type: ignore

    assert numpy.allclose(data.variables[new1.name].values,
                          data.variables['var1'].values * 201.5,
                          rtol=0)
    assert numpy.allclose(data.variables[new2.name].values,
                          data.variables['var1'].values // 5,
                          rtol=0)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_fillvalue(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the management of masked values."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs, with_fillvalue=True)

    # Load the dataset written with masked values in the collection and
    # compare it to the original dataset.
    data = zcollection.load()
    assert data is not None

    ds = next(create_test_dataset_with_fillvalue())

    values = data.variables['var1'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.ma.allclose(ds.variables['var1'].values, values)

    values = data.variables['var2'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.ma.allclose(ds.variables['var2'].values, values)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_degraded_tests(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
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
):
    """Test of the insertion of a dataset in which a variable is missing.

    This happens, for example, when a variable is not acquired, but
    created by an algorithm.
    """
    tested_fs = request.getfixturevalue(arg)
    ds = next(create_test_dataset_with_fillvalue()).to_xarray()
    zcollection = convenience.create_collection(
        axis='time',
        ds=ds,
        partition_handler=partitioning.Date(('time', ), 'M'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs)
    zcollection.insert(ds, merge_callable=merging.merge_time_series)

    ds = next(create_test_dataset_with_fillvalue())
    ds.drops_vars('var1')
    zcollection.insert(ds)

    data = zcollection.load()
    assert data is not None
    assert numpy.ma.allequal(
        data.variables['var1'].values,
        numpy.ma.masked_equal(
            numpy.full(ds.variables['var1'].shape,
                       ds.variables['var1'].fill_value,
                       ds.variables['var1'].dtype),
            ds.variables['var1'].fill_value))


# For the moment, this test does not work with S3: minio creates a
# directory for the file "time"; therefore zarr cannot detect an invalid
# array.
@pytest.mark.parametrize('arg', ['local_fs'])  # , "s3_fs"])
def test_insert_failed(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the insertion of a dataset in which the insertion failed."""
    tested_fs = request.getfixturevalue(arg)
    ds = next(create_test_dataset())
    zcollection = collection.Collection('time',
                                        ds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    partitions = list(zcollection.partitioning.split_dataset(ds, 'time'))
    one_directory = zcollection.fs.sep.join(
        (zcollection.partition_properties.dir, ) + partitions[0][0])
    zcollection.fs.makedirs(one_directory, exist_ok=False)
    zcollection.fs.touch(zcollection.fs.sep.join((one_directory, 'time')))

    with pytest.raises(OSError):
        zcollection.insert(ds)

    zcollection.insert(ds)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_map_partition(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

    result = zcollection.map(
        lambda x: x.variables['var1'].values * 2)  # type: ignore
    for item in result.compute():
        folder = zcollection.fs.sep.join(
            (zcollection.partition_properties.dir,
             zcollection.partitioning.join(item[0], zcollection.fs.sep)))
        ds = storage.open_zarr_group(folder, zcollection.fs)
        assert numpy.allclose(item[1], ds.variables['var1'].values * 2)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_indexer(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

    indexers = zcollection.map(
        lambda x: slice(0, x.dimensions['num_lines'])  # type: ignore
    ).compute()
    ds1 = zcollection.load(indexer=indexers)
    assert ds1 is not None
    ds2 = zcollection.load()
    assert ds2 is not None

    assert numpy.allclose(ds1.variables['var1'].values,
                          ds2.variables['var1'].values)


def test_variables():
    """Test the listing of the variables in a collection."""
    fs = fsspec.filesystem('memory')
    ds = next(create_test_dataset())
    zcollection = collection.Collection(axis='time',
                                        ds=ds.metadata(),
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


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_map_overlap(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the map overlap method."""
    tested_fs = request.getfixturevalue(arg)
    zcollection = create_test_collection(tested_fs)

    result = zcollection.map_overlap(
        lambda x: x.variables['var1'].values * 2,  # type: ignore
        depth=3)  # type: ignore

    for partition, indices, data in result.compute():
        folder = zcollection.fs.sep.join(
            (zcollection.partition_properties.dir,
             zcollection.partitioning.join(partition, zcollection.fs.sep)))
        ds = storage.open_zarr_group(folder, zcollection.fs)
        assert numpy.allclose(data[indices, :] * 2,
                              ds.variables['var1'].values * 2)
