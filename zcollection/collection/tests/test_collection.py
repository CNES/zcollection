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
import copy
import datetime
import io
import logging
import multiprocessing

import dask.array.core
import dask.distributed
import fsspec
import numpy
import pytest

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
from ...tests.cluster import dask_client, dask_cluster  # noqa: F401
from ...tests.data import (
    DELTA,
    END_DATE,
    FILE_SYSTEM_DATASET,
    START_DATE,
    create_test_collection,
    create_test_dataset,
    create_test_dataset_with_fillvalue,
)
from ...tests.fixture import dask_arrays, numpy_arrays  # noqa: F401
from ...tests.fs import local_fs, s3, s3_base, s3_fs  # noqa: F401


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_collection_creation(
    fs,
    request,
) -> None:
    """Test the creation of a collection."""
    tested_fs = request.getfixturevalue(fs)
    zds = next(create_test_dataset(False))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    assert isinstance(str(zcol), str)
    assert zcol.have_immutable is False
    assert zcol.load(distributed=False) is None

    serialized = collection.Collection.from_config(
        path=str(tested_fs.collection),
        filesystem=tested_fs.fs,
    )
    assert serialized.axis == zcol.axis
    assert serialized.metadata == zcol.metadata
    assert serialized.partition_properties == zcol.partition_properties
    assert serialized.partitioning.get_config(
    ) == zcol.partitioning.get_config()

    with pytest.raises(ValueError,
                       match='zarr collection not found at path .*'):
        collection.Collection.from_config(str(tested_fs.collection.parent))

    with pytest.raises(ValueError,
                       match='The variable .* is not defined in the dataset'):
        collection.Collection('time_tai', zds.metadata(),
                              partitioning.Date(('time', ), 'D'),
                              str(tested_fs.collection))

    with pytest.raises(
            ValueError,
            match='The partitioning key .* is not defined in the dataset.'):
        collection.Collection('time', zds.metadata(),
                              partitioning.Date(('time_tai', ), 'D'),
                              str(tested_fs.collection))

    with pytest.raises(ValueError, match='he mode .* is not supported'):
        collection.Collection(
            axis='time',
            ds=zds.metadata(),
            mode='X',  # type: ignore[arg-type]
            partition_handler=partitioning.Date(('time', ), 'D'),
            partition_base_dir=str(tested_fs.collection),
            filesystem=tested_fs.fs)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert(  # noqa: PLR0915
    dask_client,  # noqa: F811
    arrays_type,
    distributed,
    fs,
    request,
    tmpdir,
) -> None:
    """Test the insertion of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    datasets = list(create_test_dataset(delayed=False))
    zcol = collection.Collection('time',
                                 datasets[0].metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs,
                                 synchronizer=sync.ProcessSync(
                                     str(tmpdir / 'lock.lck')))

    indices = numpy.arange(0, len(datasets))
    rng = numpy.random.default_rng(42)
    rng.shuffle(indices)
    for idx in indices:
        zcol.insert(datasets[idx],
                    merge_callable=merging.merge_time_series,
                    distributed=distributed)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    # Adding same datasets once more (should not change anything)
    for idx in indices[:5]:
        zcol.insert(datasets[idx], distributed=distributed)

    assert list(zcol.partitions()) == sorted(zcol.partitions())

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    values = data.variables['var1'].values
    numpy.all(values == numpy.vstack((numpy.arange(values.shape[0]), ) *
                                     values.shape[1]).T)

    values = data.variables['var2'].values
    numpy.all(values == numpy.vstack((numpy.arange(values.shape[0]), ) *
                                     values.shape[1]).T)

    data = zcol.load(delayed=delayed,
                     filters='year == 2020',
                     distributed=distributed)
    assert data is None

    data = zcol.load(delayed=delayed,
                     filters='year == 2000',
                     distributed=distributed)
    assert data is not None
    assert data.variables['time'].shape[0] == 61

    data = zcol.load(delayed=delayed,
                     filters='year == 2000 and month == 4',
                     distributed=distributed)
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[M]') == numpy.datetime64('2000-04-01'))

    data = zcol.load(delayed=delayed,
                     filters='year == 2000 and month == 4 and day == 15',
                     distributed=distributed)
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[D]') == numpy.datetime64('2000-04-15'))

    data = zcol.load(
        delayed=delayed,
        filters='year == 2000 and month == 4 and day in range(5, 25)',
        distributed=distributed)
    assert data is not None
    data = zcol.load(delayed=delayed,
                     filters=lambda keys: datetime.date(2000, 4, 5) <= datetime
                     .date(keys['year'], keys['month'], keys['day']
                           ) <= datetime.date(2000, 4, 24),
                     distributed=distributed)
    assert data is not None
    dates = data.variables['time'].values.astype('datetime64[D]')
    assert dates.min() == numpy.datetime64('2000-04-06')
    assert dates.max() == numpy.datetime64('2000-04-24')

    for path_relative, path_absolute in zcol.iterate_on_records():
        assert isinstance(path_relative, str)
        assert isinstance(path_absolute, str)
        assert path_relative in path_absolute

    zcol = convenience.open_collection(str(tested_fs.collection),
                                       mode='r',
                                       filesystem=tested_fs.fs)
    zds = zcol.load(delayed=delayed,
                    selected_variables=['var1'],
                    distributed=distributed)
    assert zds is not None
    assert 'var1' in zds.variables
    assert 'var2' not in zds.variables

    zds = zcol.load(delayed=delayed,
                    selected_variables=[],
                    distributed=distributed)
    assert zds is not None
    assert len(zds.variables) == 0

    zds = zcol.load(delayed=delayed,
                    selected_variables=['varX'],
                    distributed=distributed)
    assert zds is not None
    assert len(zds.variables) == 0


@pytest.mark.parametrize(('fs', 'create_test_data'), FILE_SYSTEM_DATASET)
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_update(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    create_test_data,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = next(create_test_data(delayed=delayed))
    zcol = collection.Collection('time',
                                 zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds, distributed=distributed)

    def update(zds: dataset.Dataset, shift: int = 3):
        """Update function used for this test."""
        return {'var2': zds.variables['var1'].values * -1 + shift}

    zcol.update(
        update,
        delayed=delayed,  # type: ignore[arg-type]
        distributed=distributed)

    data = zcol.load(distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 3,
                          rtol=0)

    zcol.update(
        update,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        variables=('var2', ),
        depth=1,
        shift=5)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 5,
                          rtol=0)

    # Test case if the selected variables do not contain the variable
    # to update.
    zcol.update(
        update,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        selected_variables=['var1'],
        depth=1,
        shift=5)

    data = zcol.load(delayed=delayed, distributed=distributed)
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

    zcol.update(
        update_with_info,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        depth=1,
        shift=10)

    data = zcol.load(delayed=delayed, distributed=distributed)
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

    zcol.update(
        update_and_trim,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        trim=False,
        depth=1)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1,
                          rtol=0)

    def invalid_var_name(zds: dataset.Dataset):
        """Update function used to test if the user wants to update a non-
        existent variable name."""
        return {'var99': zds.variables['var1'].values * -1 + 3}

    with pytest.raises(ValueError, match='Unknown variables'):
        zcol.update(
            invalid_var_name,  # type: ignore[arg-type]
            distributed=distributed)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_list_partitions(
    dask_client,  # noqa: F811
    arg,
    request,
) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, delayed=False)

    all_partitions = list(zcol.partitions())
    assert len(all_partitions) == 6

    def full_path(partition):
        return zcol.fs.sep.join((zcol.partition_properties.dir, partition))

    selected_partitions = ['year=2000/month=01/day=01']
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert partitions == list(map(full_path, selected_partitions))

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=01'
    ]
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert partitions == list(map(full_path, selected_partitions[:1]))

    selected_partitions = ['year=2000/month=01/day=02']
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert not partitions

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=02',
        'year=2000/month=01/day=07'
    ]
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert partitions == list(
        map(full_path, [selected_partitions[0], selected_partitions[2]]))

    indexer = zcol.map(
        lambda x: slice(0, x.dimensions['num_lines'])  # type: ignore[arg-type]
    ).compute()[3:]

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=02',
        'year=2000/month=01/day=13'
    ]
    partitions = list(
        zcol.partitions(indexer=indexer,
                        selected_partitions=selected_partitions))
    assert partitions == list(map(full_path, selected_partitions[-1:]))


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_drop_partitions(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs,
                                  delayed=False,
                                  distributed=distributed)

    all_partitions = list(zcol.partitions())
    assert 'month=01' in [
        item.split(zcol.fs.sep)[-2] for item in all_partitions
    ]

    zcol.drop_partitions(filters='year == 2000 and month==1 and day<13',
                         distributed=distributed)
    partitions = list(zcol.partitions())
    part_ge_13 = [
        int(item.split(zcol.fs.sep)[-1][-2:]) >= 13 for item in partitions
    ]
    assert all(part_ge_13)

    npartitions = len(partitions)
    zcol.drop_partitions(timedelta=datetime.timedelta(days=1),
                         distributed=distributed)
    partitions = list(zcol.partitions())
    assert len(partitions) == npartitions

    zcol.drop_partitions(timedelta=datetime.timedelta(0),
                         distributed=distributed)
    partitions = list(zcol.partitions())
    assert len(partitions) == 0

    zcol = convenience.open_collection(str(tested_fs.collection),
                                       mode='r',
                                       filesystem=tested_fs.fs)
    assert zcol.is_readonly()
    with pytest.raises(io.UnsupportedOperation):
        zcol.drop_partitions(distributed=distributed)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_drop_variable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the dropping of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs,
                                  delayed=False,
                                  distributed=distributed)

    with pytest.raises(ValueError,
                       match='The variable .* is part of the partitioning'):
        zcol.drop_variable('time', distributed=distributed)
    zcol.drop_variable('var1', distributed=distributed)

    with pytest.raises(ValueError, match='does not exist in the collection'):
        zcol.drop_variable('var1', distributed=distributed)

    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None
    assert 'var1' not in zds.variables

    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='r',
                                       filesystem=tested_fs.fs)

    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None
    assert 'var1' not in zds.variables


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_add_variable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test adding a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs,
                                  delayed=False,
                                  distributed=distributed)

    # Variable already exists
    new = meta.Variable(name='time',
                        dtype=numpy.dtype('float64'),
                        dimensions=('num_lines', ))
    with pytest.raises(ValueError, match='already exists'):
        zcol.add_variable(new, distributed=distributed)

    # Variable have unknown dimension.
    new = meta.Variable(name='x',
                        dtype=numpy.dtype('float64'),
                        dimensions=('num_lines', 'x'))
    with pytest.raises(ValueError, match='must use the dataset dimensions'):
        zcol.add_variable(new, distributed=distributed)

    new = meta.Variable(
        name='var3',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    zcol.add_variable(new, distributed=distributed)

    assert new.name in zcol.metadata.variables

    # Testing the configuration update by reopening the collection
    zcol = collection.Collection.from_config(path=str(tested_fs.collection),
                                             filesystem=tested_fs.fs)

    assert new.name in zcol.metadata.variables

    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None
    values = zds.variables['var3'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.all(values.mask)  # type: ignore[arg-type]


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_variable_immutable(  # noqa: PLR0915
        dask_client,  # noqa: F811
        arg,
        request) -> None:
    """Test the adding of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, delayed=False)

    # Unknown dimension
    known_dimensions, _ = zcol.dimensions_properties()
    dim = 'num_pixels'

    new_arr = meta.Variable(
        name='var_immutable',
        dtype=numpy.dtype('int16'),
        dimensions=(dim, 'x'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )

    with pytest.raises(ValueError, match='must use the dataset dimensions'):
        zcol.add_variable(variable=new_arr, distributed=False)

    assert new_arr.name not in zcol.metadata.variables

    new = meta.Variable(
        name='var_immutable',
        dtype=numpy.dtype('int16'),
        dimensions=(dim, ),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )

    zcol.add_variable(variable=new, distributed=False)

    assert new.name in zcol.metadata.variables

    # Invalid dimension size
    with pytest.raises(ValueError, match='with an invalid size'):
        zcol.update_immutable(
            name=new.name,
            data=numpy.arange(known_dimensions[dim] * 2, dtype='int16'),
        )

    # Invalid name
    with pytest.raises(ValueError, match='does not exist'):
        zcol.update_immutable(
            name='invalid name',
            data=numpy.arange(known_dimensions[dim] * 2, dtype='int16'),
        )

    data = zcol.load(selected_variables=[new_arr.name],
                     delayed=False,
                     distributed=False)

    assert data is not None
    assert numpy.array_equal(
        data[new_arr.name].values,
        numpy.full(shape=known_dimensions[dim], fill_value=new.fill_value))

    arr_data = numpy.arange(known_dimensions[dim], dtype='int16')
    zcol.update_immutable(name=new.name, data=arr_data)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)
    assert new_arr.name in zcol.metadata.variables

    data = zcol.load(selected_variables=[new_arr.name],
                     delayed=False,
                     distributed=False)

    assert data is not None
    assert numpy.array_equal(data[new_arr.name].values, arr_data)
    assert data[new_arr.name].attrs == new_arr.attrs
    assert data[new_arr.name].fill_value == new_arr.fill_value

    # Adding missing dimension
    new_dim = meta.Dimension(name='x', value=2, chunks=1)
    zcol.add_dimension(dimension=new_dim)

    new_data = numpy.arange(known_dimensions[dim] * new_dim.value,
                            dtype='int16').reshape(known_dimensions[dim],
                                                   new_dim.value)

    new_x = variable.Array(name='var_immutable_x',
                           data=new_data,
                           dimensions=(dim, new_dim.name))

    zcol.add_variable(variable=new_x, distributed=False)
    zcol.update_immutable(name=new_x.name, data=new_x.values)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)
    assert new_x.name in zcol.metadata.variables

    data = zcol.load(selected_variables=[new_x.name],
                     delayed=True,
                     distributed=True)

    assert data is not None
    new_data_r = data[new_x.name]

    assert numpy.array_equal(new_data_r.values, new_x.data)
    assert new_data_r.attrs == new_x.attrs
    assert new_data_r.fill_value == new_x.fill_value
    assert new_data_r.data.chunksize[1] == new_dim.chunks

    # Updating immutable with the update function
    with pytest.raises(ValueError, match='Immutable variables'):
        zcol.update(
            lambda zds: {new_x.name: 5},  # type: ignore[arg-type]
            distributed=False,
        )

    zcol.drop_variable(variable=new_x.name)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)
    assert new_arr.name in zcol.metadata.variables
    assert new_x.name not in zcol.metadata.variables

    zcol.drop_variable(variable=new_arr.name)

    assert new_arr.name not in zcol.metadata.variables

    data = zcol.load(selected_variables=[new_x.name],
                     delayed=True,
                     distributed=True)

    assert data is not None

    assert new_x.name not in data.variables
    assert new_arr.name not in data.variables


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_dimension_add_drop(arg, request) -> None:
    """Test the adding of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, delayed=False)

    dim_dum = meta.Dimension(name='dummy', value=20, chunks=2)
    zcol.add_dimension(dimension=dim_dum)

    with pytest.raises(ValueError, match='already exists in the'):
        zcol.add_dimension(dimension=dim_dum)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)

    assert dim_dum.name in zcol.metadata.dimensions
    assert dim_dum.name in zcol.metadata.dim_size
    assert dim_dum.name in zcol.metadata.dim_chunks

    zcol.drop_dimension(dimension=dim_dum.name)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)

    assert dim_dum.name not in zcol.metadata.dimensions
    assert dim_dum.name not in zcol.metadata.dim_size
    assert dim_dum.name not in zcol.metadata.dim_chunks


@pytest.mark.parametrize(('fs', 'create_test_data'), FILE_SYSTEM_DATASET)
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_add_update(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    create_test_data,
    request,
) -> None:
    """Test the adding and updating of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = next(create_test_data(delayed=delayed))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds, distributed=distributed)

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
    zcol.add_variable(new1, distributed=distributed)
    zcol.add_variable(new2, distributed=distributed)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None

    # Invalid update function
    with pytest.raises(TypeError, match='must be a callable'):
        zcol.update(1, new1.name, delayed=delayed)  # type: ignore[arg-type]

    def update_1(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 201.5}

    def update_2(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values // 5}

    zcol.update(update_1, new1.name, delayed=delayed)  # type: ignore[arg-type]
    zcol.update(update_2, new2.name, delayed=delayed)  # type: ignore[arg-type]

    if not (delayed and distributed):
        # If the dataset is not delayed, we need to reload it.
        data = zcol.load(delayed=False, distributed=distributed)
        assert data is not None

    assert numpy.allclose(data.variables[new1.name].values,
                          data.variables['var1'].values * 201.5,
                          rtol=0)
    assert numpy.allclose(data.variables[new2.name].values,
                          data.variables['var1'].values // 5,
                          rtol=0)


@pytest.mark.parametrize(('fs', 'create_test_data'), FILE_SYSTEM_DATASET)
def test_update_with_immutable(
    fs,
    create_test_data,
    request,
) -> None:
    """Test the inclusion of immutable variables in the dataset provided to the
    update callback function."""
    tested_fs = request.getfixturevalue(fs)
    zds = next(create_test_data(delayed=False))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds, distributed=False)

    new_var = meta.Variable(
        name='var4',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    new_im = meta.Variable(
        name='var_immutable',
        dtype=numpy.dtype('int16'),
        dimensions=('num_pixels', ),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )

    zcol.add_variable(new_var, distributed=False)
    zcol.add_variable(variable=new_im, distributed=False)

    # Immutable variables are included
    def update_1(_zds):
        """Update function used for this test."""
        assert new_im.name in _zds.variables

        return {new_var.name: _zds.variables['var1'].values * 201.5}

    zcol.update(
        update_1,  # type: ignore[arg-type]
        delayed=False,
        distributed=False,
    )

    # Immutable variables are not included if excluded
    def update_2(_zds):
        """Update function used for this test."""
        assert new_im.name not in _zds.variables

        return {new_var.name: _zds.variables['var1'].values * 201.5}

    zcol.update(
        update_2,  # type: ignore[arg-type]
        selected_variables=['var1'],
        delayed=False,
        distributed=False,
    )


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_fillvalue(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    request,
) -> None:
    """Test the management of masked values."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcol = create_test_collection(tested_fs,
                                  delayed=delayed,
                                  with_fillvalue=True)

    # Load the dataset written with masked values in the collection and
    # compare it to the original dataset.
    data = zcol.load(delayed=delayed)
    assert data is not None

    zds = next(create_test_dataset_with_fillvalue(delayed=delayed))

    values = data.variables['var1'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.ma.allclose(zds.variables['var1'].values, values)

    values = data.variables['var2'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.ma.allclose(zds.variables['var2'].values, values)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_degraded_tests(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the degraded functionality."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, distributed=distributed)

    fake_ds = next(create_test_dataset())
    fake_ds.variables['var3'] = fake_ds.variables['var1']
    fake_ds.variables['var3'].name = 'var3'

    with pytest.raises(ValueError, match='is unknown'):
        zcol.insert(fake_ds, distributed=distributed)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_with_missing_variable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test of the insertion of a dataset in which a variable is missing.

    This happens, for example, when a variable is not acquired, but
    created by an algorithm.
    """
    tested_fs = request.getfixturevalue(arg)
    zds = next(create_test_dataset_with_fillvalue())
    zcol = convenience.create_collection(
        axis='time',
        ds=zds,
        partition_handler=partitioning.Date(('time', ), 'M'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs)
    zcol.insert(
        ds=zds,
        merge_callable=merging.merge_time_series,  # type: ignore[arg-type]
        distributed=distributed)

    zds = next(create_test_dataset_with_fillvalue())

    zds_v1 = zds.variables['var1']
    zds.drops_vars('var1')
    zcol.insert(zds, distributed=distributed)

    data = zcol.load()
    assert data is not None
    assert numpy.ma.allequal(
        data.variables['var1'].values,
        numpy.ma.masked_equal(
            x=numpy.full(shape=zds_v1.shape,
                         fill_value=zds_v1.fill_value,
                         dtype=zds_v1.dtype),
            value=zds_v1.fill_value,
        ))


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_insert_missing_dimensions(fs, request) -> None:
    """Test the insertion of data with missing dimensions and variables."""
    tested_fs = request.getfixturevalue(fs)
    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='grid',
                data=numpy.arange(4 * 360 * 180).reshape(4, 360, 180),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection(axis='time',
                                 ds=zds_reference.metadata(),
                                 partition_handler=partitioning.Date(
                                     variables=('time', ), resolution='D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    # Adding a new dimension and a variable with this dimension so we
    # can test the insertion of a dataset not containing all dimensions
    new_dim = meta.Dimension(name='x', value=2, chunks=1)
    new_var = meta.Variable(name='new_var',
                            dtype=numpy.dtype('int8'),
                            dimensions=(zcol.dimension, new_dim.name),
                            fill_value=0)

    zcol.add_dimension(dimension=new_dim)
    zcol.add_variable(variable=new_var, distributed=False)

    zcol.insert(ds=zds_reference, distributed=False)

    data = zcol.load(distributed=False)

    assert data is not None
    assert set(data.variables) == {'time', 'grid', new_var.name}
    assert set(data.dimensions) == {'lat', 'lon', 'time', new_dim.name}

    #: pylint: disable=protected-access
    data_ins = zcol._set_ds_for_insertion(ds=data)
    assert set(data_ins.variables) == {'time', 'grid', new_var.name}
    assert set(data_ins.dimensions) == {'lat', 'lon', 'time', new_dim.name}


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_failed(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    request,
) -> None:
    """Test the insertion of a dataset in which the insertion failed."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = next(create_test_dataset(delayed=delayed))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    partitions = list(zcol.partitioning.split_dataset(zds, 'time'))

    # Create a file in the directory where the dataset should be written. This
    # should cause the insertion to fail.
    sep = zcol.fs.sep
    one_directory = sep.join((zcol.partition_properties.dir, ) +
                             partitions[0][0])
    zcol.fs.makedirs(sep.join(one_directory.split(sep)[:-1]))
    zcol.fs.touch(one_directory)

    with pytest.raises((OSError, ValueError)):
        zcol.insert(zds, distributed=distributed)

    # Because the insert failed, the partition that was supposed to be created
    # was deleted.
    assert not zcol.fs.exists(one_directory)
    zcol.insert(zds, distributed=distributed)


def test_insert_invalid_dimensions(
    dask_client,  # noqa: F811
    request,
) -> None:
    """Test the insertion of a dataset with invalid dimensions properties."""
    tested_fs = request.getfixturevalue('local_fs')
    delayed = False
    zds = next(create_test_dataset(delayed=delayed))
    zcol = collection.Collection('time',
                                 zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    # Swapping dimensions names to create invalid ones
    ds = zds.to_xarray().rename({
        'num_lines': 'num_pixels',
        'num_pixels': 'num_lines'
    })

    with pytest.raises(ValueError, match='Inserted dimension '):
        zcol.insert(ds)


def test_insert_invalid_variable(
    dask_client,  # noqa: F811
    request,
) -> None:
    """Test the insertion of a dataset containing a variable with invalid
    properties."""
    tested_fs = request.getfixturevalue('local_fs')
    delayed = False
    zds = next(create_test_dataset(delayed=delayed))
    zds_x = copy.deepcopy(zds)

    # Changing variable dtype and fill value
    zds_x['var1'].fill_value = 10
    zds_x['var1'].array = zds_x['var1'].array.astype('int8')

    zcol = collection.Collection(axis='time',
                                 ds=zds_x.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    with pytest.raises(ValueError, match='has invalid dtype'):
        zcol.insert(zds)

    # Fixing dtype
    zds['var1'].array = zds['var1'].array.astype('int8')

    with pytest.raises(ValueError, match='has invalid fill_value'):
        zcol.insert(zds)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_map_partition(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcol = create_test_collection(tested_fs, delayed=delayed)

    result = zcol.map(
        lambda x: x.variables['var1'].values * 2,  # type: ignore[arg-type]
        delayed=delayed)
    for item in result.compute():
        folder = zcol.fs.sep.join(
            (zcol.partition_properties.dir,
             zcol.partitioning.join(item[0], zcol.fs.sep)))
        zds = storage.open_zarr_group(folder, zcol.fs, delayed=False)
        assert numpy.allclose(item[1], zds.variables['var1'].values * 2)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_indexer(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcol = create_test_collection(tested_fs,
                                  delayed=delayed,
                                  distributed=distributed)

    indexers = zcol.map(lambda x: slice(  # type: ignore[arg-type]
        0, x.dimensions['num_lines'])).compute()
    zds1 = zcol.load(indexer=indexers,
                     delayed=delayed,
                     distributed=distributed)
    assert zds1 is not None
    zds2 = zcol.load(delayed=delayed, distributed=distributed)
    assert zds2 is not None

    assert numpy.allclose(zds1.variables['var1'].values,
                          zds2.variables['var1'].values)


def test_variables() -> None:
    """Test the listing of the variables in a collection."""
    fs = fsspec.filesystem('memory')
    zds = next(create_test_dataset(delayed=False))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir='/',
                                 filesystem=fs)
    variables = zcol.variables()
    assert isinstance(variables, tuple)
    assert len(variables) == 3
    assert variables[0].name == 'time'
    assert variables[1].name == 'var1'
    assert variables[2].name == 'var2'

    variables = zcol.variables(('time', ))
    assert len(variables) == 1
    assert variables[0].name == 'time'


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_map_overlap(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    request,
) -> None:
    """Test the map overlap method."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcol = create_test_collection(tested_fs, delayed=delayed)

    def func(zds: dataset.Dataset, partition_info: tuple[str, slice]):
        assert partition_info[0] == 'num_lines'
        var = zds.variables['var1']
        slices = [slice(None)] * len(var.dimensions)
        slices[var.dimensions.index(partition_info[0])] = partition_info[1]
        return var.values[tuple(slices)] * 2

    result = zcol.map_overlap(
        func,  # type: ignore[arg-type]
        delayed=delayed,
        depth=1)

    for partition, data in result.compute():
        folder = zcol.fs.sep.join(
            (zcol.partition_properties.dir,
             zcol.partitioning.join(partition, zcol.fs.sep)))
        zds = storage.open_zarr_group(folder, zcol.fs, delayed=False)
        assert numpy.allclose(data, zds.variables['var1'].values * 2)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_immutable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the insertion of a dataset with variables that are immutable
    relative to the partitioning."""
    tested_fs = request.getfixturevalue(arg)
    rng = numpy.random.default_rng(42)

    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='lon',
                data=numpy.arange(0, 360, 1),
                dimensions=('lon', ),
            ),
            dataset.Array(
                name='lat',
                data=numpy.arange(-90, 90, 1),
                dimensions=('lat', ),
            ),
            dataset.Array(
                name='grid',
                data=rng.random((4, 360, 180)),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection('time',
                                 zds_reference.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    assert zcol.have_immutable
    assert tested_fs.fs.exists(zcol.immutable_path)

    zcol.insert(zds_reference, distributed=distributed)

    zds = zcol.load(delayed=False, distributed=distributed)
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

    zcol.update(
        update,  # type: ignore[arg-type]
        delayed=False,
        distributed=distributed,
        varname='grid')
    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None

    assert numpy.all(zds.variables['grid'].values ==
                     zds_reference.variables['grid'].values * -1)
    assert numpy.all(
        zds.variables['time'].values == zds_reference.variables['time'].values)
    assert numpy.all(
        zds.variables['lon'].values == zds_reference.variables['lon'].values)
    assert numpy.all(
        zds.variables['lat'].values == zds_reference.variables['lat'].values)

    new_variable = meta.Variable(
        'new_var',
        numpy.float64,
        dimensions=('time', 'lon', 'lat'),
        attrs=(meta.Attribute('units', 'm'), ),
    )
    zcol.add_variable(new_variable, distributed=distributed)
    zcol.update(
        update,  # type: ignore[arg-type]
        distributed=distributed,
        varname='new_var')
    zds = zcol.load(distributed=distributed)
    assert zds is not None
    assert numpy.all(zds.variables['new_var'].values ==
                     zds_reference.variables['grid'].values)

    new_variable = meta.Variable(
        'new_var2',
        numpy.float64,
        dimensions=('another_dim', ),
        attrs=(meta.Attribute('units', 'm'), ),
    )
    with pytest.raises(
            ValueError,
            match='The new variable must use the dataset dimensions.'):
        zcol.add_variable(new_variable, distributed=distributed)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_insert_immutable_only(fs, request, caplog) -> None:
    """Test insertion of data only containing immutable variables."""
    tested_fs = request.getfixturevalue(fs)
    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='lat',
                data=numpy.arange(-90, 90, 1),
                dimensions=('lat', ),
            ),
            dataset.Array(
                name='grid',
                data=numpy.arange(4 * 360 * 180).reshape(4, 360, 180),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection(axis='time',
                                 ds=zds_reference.metadata(),
                                 partition_handler=partitioning.Date(
                                     variables=('time', ), resolution='D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    assert zcol.have_immutable
    assert zcol.load(distributed=False) is None

    caplog.set_level(logging.DEBUG)
    partitions = zcol.insert(ds=zds_reference.select_vars(names='lat'),
                             distributed=False)

    assert "Writing Zarr variable 'lat'" in caplog.text
    assert partitions == ()

    data = zcol.load(distributed=False)

    assert data is not None
    assert zcol.dimension not in data.variables
    assert numpy.array_equal(data['lat'].values, zds_reference['lat'].values)

    caplog.clear()
    zcol.insert(ds=zds_reference, distributed=False)

    assert "Writing Zarr variable 'lat'" not in caplog.text

    data = zcol.load(distributed=False)

    assert data is not None
    assert numpy.array_equal(data['time'].values, zds_reference['time'].values)
    assert numpy.array_equal(data['grid'].values, zds_reference['grid'].values)
    assert numpy.array_equal(data['lat'].values, zds_reference['lat'].values)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_copy_collection(
        dask_client,  # noqa: F811
        arg,
        distributed,
        request,
        tmpdir) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, distributed=distributed)

    target = str(tmpdir / 'copy')
    zcopy = zcol.copy(target,
                      filesystem=fsspec.filesystem('file'),
                      distributed=distributed)

    ds_before_copy = zcol.load(distributed=distributed)
    ds_after_copy = zcopy.load(distributed=distributed)

    assert ds_before_copy is not None
    assert ds_after_copy is not None

    assert numpy.all(ds_before_copy.variables['var1'].values ==
                     ds_after_copy.variables['var1'].values)
    assert numpy.all(ds_before_copy.variables['var2'].values ==
                     ds_after_copy.variables['var2'].values)


def _insert(zds: dataset.Dataset, base_dir: str, lock_file: str,
            scheduler_file: str) -> None:
    client = dask.distributed.Client(scheduler_file=scheduler_file)
    zcol = collection.Collection.from_config(
        base_dir, mode='w', synchronizer=sync.ProcessSync(lock_file))
    zcol.insert(zds, merge_callable=merging.merge_time_series)
    client.close()


def test_concurrent_insert(
    dask_client,  # noqa: F811
    tmpdir,
) -> None:
    """Test the insertion of a dataset."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset(delayed=False))
    zds = datasets[0]
    lock_file = str(tmpdir / 'lock.lck')
    synchronizer = sync.ProcessSync(lock_file)
    base_dir = str(tmpdir / 'test')
    zcol = collection.Collection(
        axis='time',
        ds=zds.metadata(),
        partition_handler=partitioning.Date(('time', ), 'D'),
        partition_base_dir=base_dir,
        filesystem=fs,
        synchronizer=synchronizer,
    )

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, mp_context=multiprocessing.get_context('spawn'))
    futures = []

    assert zcol.is_locked() is False

    indices = list(numpy.arange(0, len(datasets)))
    rng = numpy.random.default_rng()
    rng.shuffle(indices)
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
            zcol.update(update)  # type: ignore[arg-type]
            launch_update = False

    data = zcol.load()
    assert data is not None
    values = data.variables['time'].values

    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))


def test_partition_modified(
        dask_client,  # noqa: F811
        tmpdir):
    """Test the loading of a variable that has been modified since its
    creation."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset())
    zds = datasets[0]
    base_dir = str(tmpdir / 'test')
    zcol = collection.Collection('time',
                                 zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 base_dir,
                                 filesystem=fs)
    zcol.insert(zds)

    last_month = zcol.load(filters=lambda keys: keys['year'] == 2000 and keys[
        'month'] == 1 and keys['day'] == 16)
    assert last_month is not None

    zds = zcol.load()
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
    zcol.insert(last_month)

    with pytest.raises(RuntimeError, match='Try to re-load'):
        _ = zds['var2'].values

    with pytest.raises(RuntimeError, match='Try to re-load'):
        _ = zds['time'].values


@pytest.mark.parametrize('distributed', [False, True])
def test_invalid_partitions(
        dask_client,  # noqa: F811
        distributed,
        tmpdir) -> None:
    """Test the validate_partitions function."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset())
    zds = datasets.pop(0)
    zds.concat(datasets, 'num_lines')
    base_dir = str(tmpdir / 'test')
    zcol = collection.Collection(
        axis='time',
        ds=zds.metadata(),
        partition_handler=partitioning.Date(variables=('time', ),
                                            resolution='D'),
        partition_base_dir=base_dir,
        filesystem=fs,
    )
    zcol.insert(zds)
    partitions = tuple(zcol.partitions())
    rng = numpy.random.default_rng()
    choices = rng.choice(len(partitions), size=2, replace=False)
    for idx in choices:
        var2 = fs.sep.join((partitions[idx], 'var2', '0.0'))
        with fs.open(var2, 'wb') as file:
            file.write(b'invalid')

    with pytest.raises(ValueError, match='When changing to a larger dtype'):
        _ = zcol.load(delayed=False, distributed=distributed)

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        invalid_partitions = zcol.validate_partitions(distributed=distributed)

    assert len(invalid_partitions) == 2
    assert sorted(invalid_partitions) == sorted(partitions[ix]
                                                for ix in choices)
    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        zcol.validate_partitions(fix=True, distributed=distributed)

    assert zcol.load() is not None

    # Filters excludes all partitions
    assert not zcol.validate_partitions(filters='day==36',
                                        distributed=distributed)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_with_chunks(
    dask_client,  # noqa: F811
    fs,
    distributed,
    request,
    tmpdir,
) -> None:
    """Test the insertion of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    datasets = list(create_test_dataset(delayed=False))[:2]

    ds_meta = datasets[0].metadata()
    chunk_size = 5
    ds_meta.dimensions['num_pixels'].chunks = chunk_size

    zcol = collection.Collection('time',
                                 ds_meta,
                                 partitioning.Date(('time', ), 'M'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs,
                                 synchronizer=sync.ProcessSync(
                                     str(tmpdir / 'lock.lck')))

    # First insertion
    zcol.insert(datasets[0],
                merge_callable=merging.merge_time_series,
                distributed=distributed)

    # Not setting distributed to False when loading otherwise we won't have any chunk
    data = zcol.load()

    assert data is not None
    assert data.variables['var1'].data.chunksize[1] == chunk_size
    assert data.variables['var2'].data.chunksize[1] == chunk_size

    # Insertion with merge
    zcol.insert(datasets[1],
                merge_callable=merging.merge_time_series,
                distributed=distributed)
    data = zcol.load()

    assert data is not None
    # Insertion properties are kept
    assert data.variables['var1'].data.chunksize[1] == chunk_size
    assert data.variables['var2'].data.chunksize[1] == chunk_size


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_with_empty_collection(fs, request):
    """Test the behavior when working with an empty collection."""
    tested_fs = request.getfixturevalue(fs)
    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='grid',
                data=numpy.arange(4 * 360 * 180).reshape(4, 360, 180),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection(axis='time',
                                 ds=zds_reference.metadata(),
                                 partition_handler=partitioning.Date(
                                     variables=('time', ), resolution='D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs,
                                 mode='w')

    var = meta.Variable(name='var2',
                        dtype=numpy.float64,
                        dimensions=('time', 'lat'))

    zcol.add_variable(variable=var, distributed=False)

    assert var.name in zcol.metadata.variables

    def update(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 0 + 5}

    with pytest.warns(Warning, match='update an empty collection'):
        zcol.update(update, varname=var.name, distributed=False)
