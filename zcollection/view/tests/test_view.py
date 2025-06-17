# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of views
=============
"""
from __future__ import annotations

import logging
import pathlib
import shutil

import dask.distributed
import numpy
import pytest

from ... import (
    collection,
    convenience,
    dataset,
    meta,
    partitioning,
    variable,
    view,
)
from ...tests.cluster import dask_client, dask_cluster  # noqa: F401
from ...tests.data import (
    create_test_collection,
    create_test_dataset,
    make_dataset,
)
from ...tests.fixture import dask_arrays, numpy_arrays  # noqa: F401
from ...tests.fs import local_fs, s3, s3_base, s3_fs  # noqa: F401
from ...view.detail import _calculate_axis_reference


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_view(  # noqa: PLR0915
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    request,
):
    """Test the creation of a view."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)

    create_test_collection(tested_fs, delayed=False, distributed=distributed)
    zview = convenience.create_view(str(tested_fs.view),
                                    view.ViewReference(
                                        str(tested_fs.collection),
                                        tested_fs.fs),
                                    filesystem=tested_fs.fs,
                                    distributed=distributed)
    assert isinstance(zview, view.View)
    assert isinstance(str(zview), str)

    # No variable recorded
    assert not zview.variables()

    # Only reading reference variables
    zds = zview.load(delayed=delayed, distributed=distributed)
    assert set(zds.variables) == {v.name for v in zview.view_ref.variables()}

    var = meta.Variable(
        name='var2',
        dtype=numpy.float64,
        dimensions=('num_lines', 'num_pixels'),
        attrs=(meta.Attribute(name='attr', value=1), ),
    )

    with pytest.raises(ValueError, match='Variable var2 already exists'):
        zview.add_variable(var, distributed=distributed)

    assert not zview.variables()

    var.name = 'var3'
    zview.add_variable(var, distributed=distributed)

    assert {v.name for v in zview.variables()} == {var.name}

    with pytest.raises(
            ValueError,
            match='The variable .* already exists in the collection.'):
        zview.add_variable(var, distributed=distributed)

    zview = convenience.open_view(str(tested_fs.view), filesystem=tested_fs.fs)
    zds = zview.load(delayed=delayed, distributed=distributed)
    assert zds is not None
    assert set(zds['time'].values.astype('datetime64[D]')) == {
        numpy.datetime64('2000-01-01'),
        numpy.datetime64('2000-01-04'),
        numpy.datetime64('2000-01-07'),
        numpy.datetime64('2000-01-10'),
        numpy.datetime64('2000-01-13'),
        numpy.datetime64('2000-01-16'),
    }

    # Loading a variable existing only in the view.
    zds = zview.load(delayed=delayed,
                     selected_variables=('var3', ),
                     distributed=distributed)
    assert zds is not None
    assert tuple(zds.variables) == ('var3', )
    assert 'var3' in zds.metadata().variables.keys()

    # The metadata of the reference collection is not modified.
    assert 'var3' not in zview.view_ref.metadata.variables.keys()

    # Loading a set of variables one from the view and another from
    # the reference having different dimensions.
    zds = zview.load(delayed=delayed,
                     selected_variables=('time', 'var3'),
                     distributed=distributed)
    assert zds is not None
    assert tuple(zds.variables) == ('time', 'var3')
    assert 'var3' in zds.metadata().variables.keys()

    # Loading a non-existing variable.
    zds = zview.load(delayed=delayed,
                     selected_variables=('var55', ),
                     distributed=distributed)
    assert zds is not None
    assert len(zds.variables) == 0

    # Test view loading that is no longer synchronized with the reference
    # collection.
    tested_fs.fs.rm(str(
        tested_fs.view.joinpath('year=2000', 'month=01', 'day=13')),
                    recursive=True)

    assert len(tuple(zview.partitions())) == 5
    assert len(tuple(zview.view_ref.partitions())) == 6

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=07',
        'year=2000/month=01/day=13'
    ]
    assert len(tuple(
        zview.partitions(selected_partitions=selected_partitions))) == 2
    assert len(
        tuple(
            zview.view_ref.partitions(
                selected_partitions=selected_partitions))) == 3

    zds = zview.load(delayed=delayed, distributed=distributed)
    assert zds is not None
    assert set(zds['time'].values.astype('datetime64[D]')) == {
        numpy.datetime64('2000-01-01'),
        numpy.datetime64('2000-01-04'),
        numpy.datetime64('2000-01-07'),
        numpy.datetime64('2000-01-10'),
        numpy.datetime64('2000-01-16'),
    }

    # Create a variable with the unsynchronized view
    var.name = 'var4'
    zview.add_variable(var, distributed=distributed)

    # Testing variables method
    assert {v.name for v in zview.variables()} == {'var3', 'var4'}
    assert {v.name
            for v in zview.variables(selected_variables=[var.name])
            } == {var.name}

    zds = zview.load(delayed=delayed, distributed=distributed)
    assert zds is not None

    def update(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 0 + 5}

    zview.update(update, 'var3', delayed=delayed, distributed=distributed)

    with pytest.raises(ValueError, match='Variable varX does not exist'):
        zview.update(update, 'varX', distributed=distributed)

    with pytest.raises(ValueError, match='Variable var2 is read-only'):
        zview.update(update, 'var2', distributed=distributed)

    zds = zview.load(delayed=delayed, distributed=distributed)
    assert zds is not None
    assert numpy.all(zds.variables['var3'].values == 5)

    indexers = zview.map(lambda x: slice(  # type: ignore[arg-type]
        0, x.dimensions['num_lines'])).compute()
    ds1 = zview.load(delayed=delayed,
                     indexer=indexers,
                     distributed=distributed)
    assert ds1 is not None

    ds2 = zview.load(delayed=delayed, distributed=distributed)
    assert ds2 is not None

    assert numpy.allclose(ds1.variables['var1'].values,
                          ds2.variables['var1'].values)

    # Filters will eliminate all partition allowing to test a
    # branch handling this case
    ds1 = zview.load(delayed=delayed,
                     indexer=indexers,
                     filters='day > 60',
                     distributed=distributed)

    assert ds1 is None
    zview.drop_variable('var3', distributed=distributed)

    assert tuple(
        str(pathlib.Path(item))
        for item in zview.partitions(filters=zview.sync())) == (str(
            tested_fs.view.joinpath('year=2000', 'month=01', 'day=13')), )

    with pytest.raises(ValueError, match='zarr view not found at path .*'):
        convenience.open_view(str(tested_fs.collection),
                              filesystem=tested_fs.fs)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_add_variable_immutable(fs, request):
    """Test the addition of an immutable variable."""
    tested_fs = request.getfixturevalue(fs)

    create_test_collection(tested_fs, delayed=False, distributed=False)
    zview = convenience.create_view(
        path=str(tested_fs.view),
        view_ref=view.ViewReference(str(tested_fs.collection), tested_fs.fs),
        filesystem=tested_fs.fs,
        distributed=False,
    )

    # Cannot add an immutable variable to a view
    known_dimensions, _ = zview.view_ref.dimensions_properties()
    dim = 'num_pixels'

    new = variable.Array(
        name='var_immutable',
        data=numpy.arange(known_dimensions[dim], dtype='int16'),
        dimensions=(dim, ),
    )

    with pytest.raises(ValueError,
                       match='Immutable variable cannot be added to views'):
        zview.add_variable(variable=new, distributed=False)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_read_immutable(fs, request):
    """Test the reading of an immutable variable."""
    tested_fs = request.getfixturevalue(fs)

    col_ref = create_test_collection(tested_fs,
                                     delayed=False,
                                     distributed=False)
    view_ref = view.ViewReference(str(tested_fs.collection), tested_fs.fs)

    # Cannot add an immutable variable to a view
    known_dimensions, _ = col_ref.dimensions_properties()
    dim = 'num_pixels'
    var_data = numpy.arange(known_dimensions[dim], dtype='int16')

    new = variable.Array(
        name='var_immutable',
        data=var_data,
        dimensions=(dim, ),
    )

    col_ref.add_variable(variable=new, distributed=False)
    col_ref.update_immutable(name=new.name, data=var_data)

    zview = convenience.create_view(
        path=str(tested_fs.view),
        view_ref=view_ref,
        filesystem=tested_fs.fs,
        distributed=False,
    )

    # Reading immutable variable from the view
    ds = zview.load(selected_variables=[new.name], distributed=False)

    assert numpy.array_equal(ds[new.name].values, var_data)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_read_immutable_only(fs, request):
    """Test the reading in a view based on a collection only containing
    immutable variables."""
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

    view_ref = view.ViewReference(str(tested_fs.collection), tested_fs.fs)
    zview = convenience.create_view(
        path=str(tested_fs.view),
        view_ref=view_ref,
        filesystem=tested_fs.fs,
        distributed=False,
    )
    assert zview.load(distributed=False) is None

    zcol.insert(ds=zds_reference.select_vars(names='lat'), distributed=False)

    # Reading immutable variable from the view
    data = zview.load(distributed=False)

    assert set(data.variables) == {'lat'}
    assert numpy.array_equal(data['lat'].values, zds_reference['lat'].values)

    zcol.insert(ds=zds_reference, distributed=False)

    assert not zview.is_synced(distributed=False)
    zview.sync(distributed=False)

    data = zview.load(distributed=False)

    assert set(data.variables) == {'lat', 'grid', 'time'}
    assert numpy.array_equal(data['lat'].values, zds_reference['lat'].values)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_update(
    dask_client,  # noqa: F811
    fs,
    request,
    tmp_path,
):
    """Test updating variable."""
    tested_fs = request.getfixturevalue(fs)

    create_test_collection(tested_fs, delayed=False)
    zview = convenience.create_view(path=str(tested_fs.view),
                                    view_ref=view.ViewReference(
                                        str(tested_fs.collection),
                                        tested_fs.fs),
                                    filesystem=tested_fs.fs)

    var = meta.Variable(name='var3',
                        dtype=numpy.float64,
                        dimensions=('num_lines', 'num_pixels'))

    zview.add_variable(var)

    def to_zero(zds: dataset.Dataset, varname):
        """Update function used to set a variable to 0."""
        return {varname: zds.variables['var1'].values * 0}

    zview.update(to_zero, var.name)  # type: ignore[arg-type]

    data = zview.load(delayed=False)
    assert numpy.all(data.variables[var.name].values == 0)

    test_dir: pathlib.Path = tmp_path / 'test'
    test_dir.mkdir()

    def plus_one_with_log(zds: dataset.Dataset, varname):
        """Update function increasing a variable by 1.

        This update function create a new file each time its called.
        """
        with dask.distributed.Lock('update'):
            i = 0
            f = test_dir / f'file_{i}'
            while f.exists():
                i += 1
                f = test_dir / f'file_{i}'

            pathlib.Path(f).touch()

        return {varname: zds.variables[var.name].values + 1}

    zview.update(plus_one_with_log, var.name)  # type: ignore[arg-type]

    # One log per partition + 1 log for the initial call
    assert len(tuple(test_dir.iterdir())) == len(list(zview.partitions())) + 1

    shutil.rmtree(test_dir)
    test_dir.mkdir()

    data = zview.load(delayed=False)
    assert numpy.all(data.variables[var.name].values == 1)

    zview.update(
        plus_one_with_log,  # type: ignore[arg-type]
        var.name,
        variables=[var.name])

    assert len(tuple(test_dir.iterdir())) == len(list(zview.partitions()))

    data = zview.load(delayed=False)
    assert numpy.all(data.variables[var.name].values == 2)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_update_with_immutable(fs, request) -> None:
    """Test the inclusion of immutable variables in the dataset provided to the
    update callback function."""
    tested_fs = request.getfixturevalue(fs)

    zcol = create_test_collection(tested_fs=tested_fs,
                                  delayed=False,
                                  distributed=False)
    view_ref = view.ViewReference(str(tested_fs.collection), tested_fs.fs)

    new_im = meta.Variable(
        name='var_immutable',
        dtype=numpy.dtype('int16'),
        dimensions=('num_pixels', ),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    zcol.add_variable(variable=new_im, distributed=False)

    zview = convenience.create_view(path=str(tested_fs.view),
                                    view_ref=view_ref,
                                    filesystem=tested_fs.fs,
                                    distributed=False)

    new_var = meta.Variable(
        name='var4',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    zview.add_variable(variable=new_var, distributed=False)

    # Immutable variables are included
    def update_1(_zds):
        """Update function used for this test."""
        assert new_im.name in _zds.variables

        return {new_var.name: _zds.variables['var1'].values * 201.5}

    zview.update(
        update_1,  # type: ignore[arg-type]
        delayed=False,
        distributed=False,
    )

    # Immutable variables are not included if excluded
    def update_2(_zds):
        """Update function used for this test."""
        assert new_im.name not in _zds.variables

        return {new_var.name: _zds.variables['var1'].values * 201.5}

    zview.update(
        update_2,  # type: ignore[arg-type]
        selected_variables=['var1'],
        delayed=False,
        distributed=False,
    )


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_view_overlap(
    dask_client,  # noqa: F811
    arg,
    request,
):
    """Test the map_overlap function."""
    tested_fs = request.getfixturevalue(arg)

    create_test_collection(tested_fs)
    zview = convenience.create_view(str(tested_fs.view),
                                    view.ViewReference(
                                        str(tested_fs.collection),
                                        tested_fs.fs),
                                    filesystem=tested_fs.fs)

    var = meta.Variable(
        name='var3',
        dtype=numpy.int8,
        dimensions=('num_lines', 'num_pixels'),
    )

    zview.add_variable(var)

    def update(zds, varname, partition_info: tuple[str, slice],
               trim_result: bool) -> dict[str, numpy.ndarray]:
        """Update function used for this test."""
        assert isinstance(partition_info, tuple)
        assert len(partition_info) == 2
        assert isinstance(partition_info[0], str)
        assert isinstance(partition_info[1], slice)
        assert partition_info[0] == 'num_lines'
        if trim_result:
            zds = zds.isel(dict((partition_info, )))
        return {varname: zds.variables['var1'].values * 1 + 5}

    zview.update(
        update,  # type: ignore[arg-type]
        'var3',
        depth=1,
        trim_result=False)
    zds = zview.load()
    assert zds is not None
    numpy.all(zds.variables['var3'].values == 5)

    zview.update(
        update,  # type: ignore[arg-type]
        'var3',
        depth=1,
        trim=False,
        trim_result=True)
    zds = zview.load()
    assert zds is not None
    numpy.all(zds.variables['var3'].values == 5)

    with pytest.raises(
            ValueError,
            match='If the depth is greater than 0, the selected variables '
            'must contain the variables updated by the function.'):
        zview.update(
            update,  #  type: ignore[arg-type]
            'var3',
            depth=1,
            trim_result=False,
            selected_variables=('var1', ))

    def map_func(_, partition_info: tuple[str, slice]):
        """Map function used for this test."""
        assert isinstance(partition_info, tuple)
        assert len(partition_info) == 2
        assert isinstance(partition_info[0], str)
        assert isinstance(partition_info[1], slice)
        assert partition_info[0] == 'num_lines'
        return partition_info

    with pytest.raises(ValueError, match='must be greater than or equal to 0'):
        zview.map_overlap(
            map_func,  # type: ignore[arg-type]
            depth=-1,
        )

    indexers = zview.map_overlap(
        map_func,  # type: ignore[arg-type]
        depth=1,
    ).compute()

    for _, data in indexers:
        assert isinstance(data, tuple)
        assert len(data) == 2
        assert isinstance(data[0], str)
        assert isinstance(data[1], slice)


def test_view_checksum(
    dask_client,  # noqa: F811
    tmp_path,
) -> None:
    """Test the checksum calculation."""
    zds = next(create_test_dataset())
    zcol = collection.Collection('time', zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tmp_path))

    zcol.insert(zds)
    partition = tmp_path / 'year=2000' / 'month=01' / 'day=01'
    axis_ref = _calculate_axis_reference(str(partition), zcol)
    assert isinstance(axis_ref.array, numpy.ndarray)
    assert isinstance(axis_ref.checksum, str)
    assert isinstance(axis_ref.dimension, str)
    assert len(axis_ref.checksum) == 64
    assert axis_ref.dimension == 'num_lines'
    assert axis_ref.checksum == ('1a1727b18e729c376442e76d806565b8'
                                 '359e3af9c93572af3e5fe8980ced6956')


@pytest.mark.filterwarnings('ignore:.*cannot be serialized.*')
@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_view_sync(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
):
    """Test the synchronization of a view."""
    tested_fs = request.getfixturevalue(arg)
    create_test_collection(tested_fs, distributed=distributed)
    zview = convenience.create_view(str(tested_fs.view),
                                    view.ViewReference(
                                        str(tested_fs.collection),
                                        tested_fs.fs),
                                    filesystem=tested_fs.fs,
                                    distributed=distributed)
    var = meta.Variable(name='var3',
                        dtype=numpy.float64,
                        dimensions=('num_lines', 'num_pixels'))
    zview.add_variable(var, distributed=distributed)
    del zview

    zcol = convenience.open_collection(str(tested_fs.collection),
                                       filesystem=tested_fs.fs,
                                       mode='w')
    zds = zcol.load(filters=lambda keys: keys['year'] == 2000 and keys['month']
                    == 1 and keys['day'] == 16)
    assert zds is not None
    dates = numpy.arange(numpy.datetime64('2000-01-16'),
                         numpy.datetime64('2000-01-16T23:59:59'),
                         numpy.timedelta64(1, 'h'))
    zds = make_dataset(
        dates.astype('M8[ns]'),
        numpy.ones((len(dates), zds.dimensions['num_pixels']),
                   dtype=numpy.int64))
    zcol.insert(zds)
    del zcol
    zview = convenience.open_view(str(tested_fs.view), filesystem=tested_fs.fs)
    assert zview is not None
    assert zview.is_synced(distributed=distributed) is False
    zview.sync(filters=lambda keys: True, distributed=distributed)
    zds = zview.load(distributed=distributed)
    assert zds is not None


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_with_empty_collection(fs, request, caplog):
    """Test the behavior when working with an empty collection."""
    caplog.set_level(logging.INFO)

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
    convenience.create_collection(axis='time',
                                  ds=zds_reference,
                                  partition_handler=partitioning.Date(
                                      variables=('time', ), resolution='D'),
                                  partition_base_dir=str(tested_fs.collection),
                                  filesystem=tested_fs.fs)

    view_ref = view.ViewReference(str(tested_fs.collection), tested_fs.fs)

    zview = convenience.create_view(
        path=str(tested_fs.view),
        view_ref=view_ref,
        filesystem=tested_fs.fs,
        distributed=False,
    )
    var = meta.Variable(name='var2',
                        dtype=numpy.float64,
                        dimensions=('time', 'lat'))

    caplog.clear()
    zview.add_variable(variable=var, distributed=False)

    assert 'skipping variable creation' in caplog.text

    assert var.name in zview.metadata.variables

    def update(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 0 + 5}

    with pytest.warns(Warning, match='function is not applied'):
        zview.update(update, varname=var.name, distributed=False)
