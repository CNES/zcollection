# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of views
=============
"""
from __future__ import annotations

import json
import logging
import pathlib
import shutil

import dask.distributed
import fsspec
import numpy
import pytest

from ... import (
    PartitionInfo,
    collection,
    convenience,
    dataset,
    meta,
    partitioning,
    store,
    variable,
    view,
)
from ...partitions import FILENAME
from ...tests.cluster import dask_client, dask_cluster  # noqa: F401
from ...tests.data import (
    create_test_collection,
    create_test_dataset,
    make_dataset,
)
from ...tests.fixture import dask_arrays, numpy_arrays  # noqa: F401
from ...tests.fs import local_fs  # noqa: F401
from ...view.detail import _calculate_axis_reference


@pytest.fixture(params=[
    pytest.param(('memory', False), id='memory-local'),
    pytest.param(('s3_fs', False), id='s3-local'),
    pytest.param(('s3_fs', True), id='s3-distributed'),
])
def mixed_filesystem_view(request, tmp_path):
    """Keep the reference on memory or S3 and the view on local disk."""
    fs_name, distributed = request.param
    if fs_name == 'memory':
        fs = fsspec.filesystem('memory')
        source = str(tmp_path / 'reference')
    else:
        tested_fs = request.getfixturevalue(fs_name)
        fs = tested_fs.fs
        source = str(tested_fs.collection)
    data = dataset.Dataset([
        dataset.Array('key', numpy.array([1, 1, 2, 2, 2, 3]), ('row', )),
        dataset.Array('value',
                      numpy.arange(12).reshape(6, 2), ('row', 'pixel')),
        dataset.Array('pixel', numpy.array([10, 20]), ('pixel', )),
    ],
                           delayed=False)
    reference = convenience.create_collection('key',
                                              data,
                                              partitioning.Sequence(('key', )),
                                              source,
                                              filesystem=fs)
    reference.insert(data.isel({'row': slice(0, 5)}), distributed=False)
    target = convenience.create_view(str(tmp_path / 'view'),
                                     view.ViewReference(source, fs),
                                     distributed=False)
    yield reference, target, data, distributed
    if fs_name == 'memory':
        fs.rm(source, recursive=True)


def test_view_add_and_sync_with_distinct_filesystems(
        mixed_filesystem_view,
        dask_client,  # noqa: F811
):
    """View arrays use reference partition sizes when added and synchronized."""
    reference, target, data, distributed = mixed_filesystem_view
    target.add_variable(meta.VariableSchema('derived',
                                            'int64',
                                            dimensions=('row', 'pixel'),
                                            fill_value=-1),
                        distributed=distributed)
    loaded = target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(loaded['derived'].values,
                                     numpy.full((5, 2), -1))

    reference.insert(data.isel({'row': slice(5, 6)}), distributed=False)
    target.sync(distributed=distributed)
    loaded = target.load(delayed=False, distributed=False)
    numpy.testing.assert_array_equal(loaded['derived'].values,
                                     numpy.full((6, 2), -1))
    numpy.testing.assert_array_equal(loaded['value'].values,
                                     data['value'].values)


def test_view_callbacks_with_distinct_filesystems(
        mixed_filesystem_view,
        dask_client,  # noqa: F811
):
    """Mappings and updates read immutables from the reference filesystem."""
    _, target, data, distributed = mixed_filesystem_view
    expected = data['value'].values[:5] + data['pixel'].values

    def mapped(zds, *, partition_info):
        return (zds['value'].values +
                zds['pixel'].values)[partition_info.indices]

    for method in ('map', 'map_overlap'):
        result = getattr(target, method)(
            mapped, selected_variables=('value', 'pixel'),
            delayed=False).compute(
                scheduler='distributed' if distributed else 'synchronous')
        numpy.testing.assert_array_equal(
            numpy.concatenate([values for _, values in result]), expected)

    target.add_variable(meta.VariableSchema('derived',
                                            'int64',
                                            dimensions=('row', 'pixel'),
                                            fill_value=-1),
                        distributed=False)

    def updated(zds):
        return {'derived': zds['value'].values + zds['pixel'].values}

    target.update(updated,
                  variables=('derived', ),
                  selected_variables=('value', 'pixel'),
                  depth=1,
                  delayed=False,
                  distributed=distributed)
    loaded = target.load(selected_variables=('derived', ),
                         delayed=False,
                         distributed=False)
    numpy.testing.assert_array_equal(loaded['derived'].values, expected)


@pytest.mark.parametrize('selection', [
    {
        'selected_partitions': []
    },
    {
        'filters': 'key > 99'
    },
    {
        'indexer': []
    },
],
                         ids=['partitions', 'filter', 'indexer'])
def test_view_empty_selection_matches_collection(tmp_path, selection):
    """An explicit empty selection returns None despite reference immutables."""
    data = dataset.Dataset([
        dataset.Array('key', numpy.array([1, 2]), ('row', )),
        dataset.Array('pixel', numpy.array([10, 20]), ('pixel', )),
    ],
                           delayed=False)
    reference = convenience.create_collection('key', data,
                                              partitioning.Sequence(('key', )),
                                              str(tmp_path / 'reference'))
    reference.insert(data, distributed=False)
    target = convenience.create_view(str(tmp_path / 'view'),
                                     view.ViewReference(
                                         str(tmp_path / 'reference')),
                                     distributed=False)

    assert reference.load(distributed=False, **selection) is None
    assert target.load(distributed=False, **selection) is None


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

    var = meta.VariableSchema(
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
            match=r'The variable .* already exists in the collection.'):
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

    # The index of the view still names the partition: nothing has opened
    # it, and a listing trusts the index.
    assert len(tuple(zview.partitions())) == 6
    assert len(zview.reindex()) == 5

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

    with pytest.raises(ValueError, match=r'zarr view not found at path .*'):
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

    var = meta.VariableSchema(name='var3',
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

    new_im = meta.VariableSchema(
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

    new_var = meta.VariableSchema(
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

    var = meta.VariableSchema(
        name='var3',
        dtype=numpy.int8,
        dimensions=('num_lines', 'num_pixels'),
    )

    zview.add_variable(var)

    def update(zds, varname, partition_info: PartitionInfo,
               trim_result: bool) -> dict[str, numpy.ndarray]:
        """Update function used for this test."""
        assert partition_info.dimension == 'num_lines'
        if trim_result:
            zds = zds.isel({partition_info.dimension: partition_info.indices})
        return {varname: zds.variables['var1'].values * 1 + 5}

    zview.update(
        update,  # type: ignore[arg-type]
        'var3',
        depth=1,
        trim_result=False)
    zds = zview.load()
    assert zds is not None
    numpy.testing.assert_array_equal(zds['var3'].values,
                                     zds['var1'].values + 5)

    zview.update(
        update,  # type: ignore[arg-type]
        'var3',
        depth=1,
        trim=False,
        trim_result=True)
    zds = zview.load()
    assert zds is not None
    numpy.testing.assert_array_equal(zds['var3'].values,
                                     zds['var1'].values + 5)

    zview.update(
        update,  # type: ignore[arg-type]
        'var3',
        depth=1,
        trim_result=False,
        selected_variables=('var1', ))
    zds = zview.load()
    assert zds is not None
    numpy.testing.assert_array_equal(zds['var3'].values,
                                     zds['var1'].values + 5)

    def map_func(_, partition_info: PartitionInfo):
        """Map function used for this test."""
        assert partition_info.dimension == 'num_lines'
        return partition_info

    indexers = zview.map_overlap(
        map_func,  # type: ignore[arg-type]
        depth=1,
    ).compute()

    assert len(indexers) == len(tuple(zview.partitions()))
    assert all(info.dimension == 'num_lines' for _, info in indexers)


def test_view_checksum(tmp_path) -> None:
    """Test the checksum calculation."""
    values = numpy.array(['2000-01-01'], dtype='datetime64[ns]')
    zds = dataset.Dataset([
        dataset.Array('time', values, ('num_lines', )),
    ],
                          delayed=False)
    zcol = collection.Collection('time', zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tmp_path))
    partition = str(tmp_path / 'year=2000' / 'month=01' / 'day=01')
    store.create_group(zcol.fs, partition)
    array = store.create_array(zcol.fs,
                               partition,
                               name='time',
                               shape=values.shape,
                               dtype=values.dtype,
                               dimensions=('num_lines', ))
    array[:] = values
    store.consolidate(zcol.fs, partition)

    axis_ref = _calculate_axis_reference(partition, zcol)
    assert isinstance(axis_ref.array, numpy.ndarray)
    numpy.testing.assert_array_equal(axis_ref.array, values)
    assert isinstance(axis_ref.checksum, str)
    assert isinstance(axis_ref.dimension, str)
    assert len(axis_ref.checksum) == 64
    assert axis_ref.dimension == 'num_lines'
    assert axis_ref.checksum == ('1a1727b18e729c376442e76d806565b8'
                                 '359e3af9c93572af3e5fe8980ced6956')


@pytest.mark.parametrize('distributed', [False, True])
def test_view_is_synced_with_filters(tmp_path, request, distributed):
    """Selected reference partitions must be present and synchronized."""
    if distributed:
        request.getfixturevalue('dask_client')
    data = dataset.Dataset([
        dataset.Array('key', numpy.array([1, 3, 2]), ('row', )),
    ],
                           delayed=False)
    reference = convenience.create_collection('key', data,
                                              partitioning.Sequence(('key', )),
                                              str(tmp_path / 'reference'))
    reference.insert(data.isel({'row': slice(0, 2)}), distributed=False)
    target = convenience.create_view(str(tmp_path / 'view'),
                                     view.ViewReference(
                                         str(tmp_path / 'reference')),
                                     filters='key < 3',
                                     distributed=False)

    assert not target.is_synced(distributed)
    assert target.is_synced(distributed, filters='key < 3')

    reference.insert(data.isel({'row': slice(2, 3)}), distributed=False)
    assert not target.is_synced(distributed, filters='key < 3')
    assert len(tuple(target.partitions())) == 1

    target.sync(filters='key == 2', distributed=distributed)
    assert target.is_synced(distributed, filters='key < 3')
    assert not target.is_synced(distributed)


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
    var = meta.VariableSchema(name='var3',
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

    zview = convenience.open_view(str(tested_fs.view), filesystem=tested_fs.fs)
    assert zview is not None
    assert zview.is_synced(distributed=distributed) is False
    zview.sync(filters='day >= 16', distributed=distributed)
    zds = zview.load(distributed=distributed)
    assert zds is not None
    assert numpy.array_equal(zds['time'].values, zcol.load()['time'].values)


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
    var = meta.VariableSchema(name='var2',
                              dtype=numpy.float64,
                              dimensions=('time', 'lat'))

    caplog.clear()
    zview.add_variable(variable=var, distributed=False)

    assert 'skipping variable creation' in caplog.text

    assert var.name in zview.metadata.variables
    reopened = convenience.open_view(str(tested_fs.view),
                                     filesystem=tested_fs.fs)
    assert var.name in reopened.metadata.variables

    def update(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 0 + 5}

    with pytest.warns(Warning, match='function is not applied'):
        zview.update(update, varname=var.name, distributed=False)


def _view_on_one_partition(tested_fs) -> view.View:
    """Create a view whose reference collection holds a single partition."""
    zds = next(create_test_dataset())
    zcol = collection.Collection('time',
                                 zds.metadata(),
                                 partitioning.Date(('time', ), 'M'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds, distributed=False)
    zview = convenience.create_view(str(tested_fs.view),
                                    view.ViewReference(
                                        str(tested_fs.collection),
                                        tested_fs.fs),
                                    filesystem=tested_fs.fs,
                                    distributed=False)
    zview.add_variable(meta.VariableSchema(name='var3',
                                           dtype=numpy.float64,
                                           dimensions=('num_lines',
                                                       'num_pixels')),
                       distributed=False)
    return zview


def test_map_overlap_on_a_single_partition(local_fs) -> None:  # noqa: F811
    """A partition with no neighbor is mapped over on its own."""
    zview = _view_on_one_partition(local_fs)
    assert len(tuple(zview.partitions())) == 1
    loaded = zview.load(delayed=False, distributed=False)
    assert loaded is not None

    sizes = zview.map_overlap(lambda zds: zds.dimensions['num_lines'],
                              depth=1).compute()

    assert [size for _, size in sizes] == [loaded.dimensions['num_lines']]


def test_update_on_a_single_partition(local_fs) -> None:  # noqa: F811
    """A partition with no neighbor is updated on its own."""
    zview = _view_on_one_partition(local_fs)

    zview.update(lambda zds: {'var3': zds.variables['var1'].values * 2},
                 depth=1,
                 distributed=False)

    loaded = zview.load(delayed=False, distributed=False)
    assert loaded is not None
    numpy.testing.assert_array_equal(loaded.variables['var3'].values,
                                     loaded.variables['var1'].values * 2)


def test_map_takes_the_scheduler_of_its_caller(
        local_fs,  # noqa: F811
        monkeypatch) -> None:
    """Mapping over a view asks for no Dask client."""

    def refuse() -> None:
        """Raise when a Dask client is requested."""
        raise AssertionError('a Dask client was requested')

    zview = _view_on_one_partition(local_fs)
    monkeypatch.setattr('zcollection.dask_utils.get_client', refuse)

    assert len(zview.map(lambda zds: 1).compute()) == 1
    assert len(zview.map_overlap(lambda zds: 1, depth=1).compute()) == 1


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_written_in_a_newer_layout(fs, request) -> None:
    """A view written in a layout this code does not know is refused."""
    tested_fs = request.getfixturevalue(fs)
    zds = next(create_test_dataset())
    collection.Collection('time',
                          zds.metadata(),
                          partitioning.Date(('time', ), 'D'),
                          str(tested_fs.collection),
                          filesystem=tested_fs.fs)
    convenience.create_view(path=str(tested_fs.view),
                            view_ref=view.ViewReference(
                                str(tested_fs.collection), tested_fs.fs),
                            filesystem=tested_fs.fs,
                            distributed=False)

    config = view.View._config(str(tested_fs.view))
    with tested_fs.fs.open(config) as stream:
        data = json.load(stream)
    data['version'] = meta.FORMAT_VERSION + 1
    with tested_fs.fs.open(config, mode='w') as stream:
        json.dump(data, stream)

    with pytest.raises(ValueError, match='written in version'):
        convenience.open_view(str(tested_fs.view), filesystem=tested_fs.fs)


def _view_owning_a_variable(
        tested_fs) -> tuple[collection.Collection, view.View]:
    """Create a view owning one variable, on a reference of six partitions.

    Args:
        tested_fs: The file system holding the collection and the view.

    Returns:
        The reference collection and the view.
    """
    zcol = create_test_collection(tested_fs, distributed=False)
    zview = convenience.create_view(str(tested_fs.view),
                                    view.ViewReference(
                                        str(tested_fs.collection),
                                        tested_fs.fs),
                                    filesystem=tested_fs.fs,
                                    distributed=False)
    zview.add_variable(meta.VariableSchema('var3',
                                           numpy.float64,
                                           dimensions=('num_lines',
                                                       'num_pixels')),
                       distributed=False)
    return zcol, zview


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_validate_partitions_reports_a_missing_variable(
        fs, request, distributed) -> None:
    """A partition short of a declared variable is written again."""
    tested_fs = request.getfixturevalue(fs)
    if distributed:
        request.getfixturevalue('dask_client')
    _, zview = _view_owning_a_variable(tested_fs)
    partitions = tuple(zview.partitions())
    zview.fs.rm(zview.fs.sep.join((partitions[1], 'var3')), recursive=True)
    zview.fs.invalidate_cache()

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        invalid = zview.validate_partitions(distributed=distributed)
    assert invalid == (zview._relative_path(partitions[1]), )

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        zview.validate_partitions(distributed=distributed, fix=True)

    assert zview.validate_partitions(distributed=distributed) == ()
    assert tuple(zview.partitions()) == partitions


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_validate_partitions_reports_a_missing_checksum(fs, request) -> None:
    """A partition short of its checksum array is written again."""
    tested_fs = request.getfixturevalue(fs)
    _, zview = _view_owning_a_variable(tested_fs)
    partitions = tuple(zview.partitions())
    zview.fs.rm(zview.fs.sep.join((partitions[0], '.checksum')),
                recursive=True)
    zview.fs.invalidate_cache()

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        invalid = zview.validate_partitions(distributed=False, fix=True)
    assert invalid == (zview._relative_path(partitions[0]), )
    assert zview.validate_partitions(distributed=False) == ()


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_validate_partitions_reports_an_orphan(fs, request) -> None:
    """A partition the reference no longer has is deleted and unindexed."""
    tested_fs = request.getfixturevalue(fs)
    zcol, zview = _view_owning_a_variable(tested_fs)
    partitions = tuple(zview.partitions())
    assert len(partitions) == 6
    zcol.drop_partitions(filters='day == 1', distributed=False)

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        invalid = zview.validate_partitions(distributed=False)
    assert invalid == (zview._relative_path(partitions[0]), )

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        zview.validate_partitions(distributed=False, fix=True)

    assert not zview.fs.exists(partitions[0])
    assert zview.validate_partitions(distributed=False) == ()
    assert len(zview._held_partitions()) == 5


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_validate_partitions_reports_an_unreadable_array(fs, request) -> None:
    """A variable whose chunk does not decode is written again."""
    tested_fs = request.getfixturevalue(fs)
    _, zview = _view_owning_a_variable(tested_fs)
    partitions = tuple(zview.partitions())
    var3 = zview.fs.sep.join((partitions[2], 'var3'))
    array = store.open_array(zview.fs, var3, mode='r+')
    array[...] = 1
    chunks = [
        item for item in zview.fs.find(var3) if not item.endswith('zarr.json')
    ]
    assert chunks
    with zview.fs.open(chunks[0], 'wb') as stream:
        stream.write(b'invalid')
    zview.fs.invalidate_cache()

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        invalid = zview.validate_partitions(distributed=False, fix=True)
    assert invalid == (zview._relative_path(partitions[2]), )
    assert zview.validate_partitions(distributed=False) == ()


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_validate_partitions_checks_the_selected_partitions(fs,
                                                            request) -> None:
    """A filter excluding the invalid partition leaves it unreported."""
    tested_fs = request.getfixturevalue(fs)
    _, zview = _view_owning_a_variable(tested_fs)
    partitions = tuple(zview.partitions())
    zview.fs.rm(zview.fs.sep.join((partitions[1], 'var3')), recursive=True)
    zview.fs.invalidate_cache()

    assert zview.validate_partitions(filters='day == 1',
                                     distributed=False) == ()
    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        assert zview.validate_partitions(
            filters='day == 4',
            distributed=False) == (zview._relative_path(partitions[1]), )


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_creating_a_view_writes_its_index(fs, request, caplog) -> None:
    """A view holds an index of its partitions as soon as it is created."""
    tested_fs = request.getfixturevalue(fs)
    create_test_collection(tested_fs, distributed=False)
    with caplog.at_level(logging.WARNING, logger='zcollection.partitions'):
        zview = convenience.create_view(str(tested_fs.view),
                                        view.ViewReference(
                                            str(tested_fs.collection),
                                            tested_fs.fs),
                                        filesystem=tested_fs.fs,
                                        distributed=False)
    assert not caplog.records
    assert zview.fs.exists(zview.fs.sep.join((zview.base_dir, FILENAME)))
