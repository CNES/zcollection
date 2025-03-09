# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of views
=============
"""
from __future__ import annotations

import pathlib

import distributed
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
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ...tests.data import (
    create_test_collection,
    create_test_dataset,
    make_dataset,
)
from ...tests.fixture import dask_arrays, numpy_arrays
from ...tests.fs import local_fs, s3, s3_base, s3_fs
from ...type_hints import ArrayLike
from ...view.detail import _calculate_axis_reference

# pylint: enable=unused-import


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_view(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    fs,
    arrays_type,
    distributed,
    request,
):
    """Test the creation of a view."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)

    create_test_collection(tested_fs, delayed=False)
    instance = convenience.create_view(str(tested_fs.view),
                                       view.ViewReference(
                                           str(tested_fs.collection),
                                           tested_fs.fs),
                                       filesystem=tested_fs.fs,
                                       distributed=distributed)
    assert isinstance(instance, view.View)
    assert isinstance(str(instance), str)

    # No variable recorded, so no data can be loaded
    with pytest.raises(ValueError):
        instance.load(delayed=delayed, distributed=distributed)

    var = meta.Variable(
        name='var2',
        dtype=numpy.float64,
        dimensions=('num_lines', 'num_pixels'),
        attrs=(meta.Attribute(name='attr', value=1), ),
    )

    with pytest.raises(ValueError):
        instance.add_variable(var, distributed=distributed)

    var.name = 'var3'
    instance.add_variable(var, distributed=distributed)

    with pytest.raises(ValueError):
        instance.add_variable(var, distributed=distributed)

    instance = convenience.open_view(str(tested_fs.view),
                                     filesystem=tested_fs.fs)
    zds = instance.load(delayed=delayed, distributed=distributed)
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
    zds = instance.load(delayed=delayed,
                        selected_variables=('var3', ),
                        distributed=distributed)
    assert zds is not None
    assert tuple(zds.variables) == ('var3', )
    assert 'var3' in zds.metadata().variables.keys()

    # The metadata of the reference collection is not modified.
    assert 'var3' not in instance.view_ref.metadata.variables.keys()

    # Loading a non-existing variable.
    zds = instance.load(delayed=delayed,
                        selected_variables=('var55', ),
                        distributed=distributed)
    assert zds is not None
    assert len(zds.variables) == 0

    # Test view loading that is no longer synchronized with the reference
    # collection.
    tested_fs.fs.rm(str(
        tested_fs.view.joinpath('year=2000', 'month=01', 'day=13')),
                    recursive=True)

    assert len(tuple(instance.partitions())) == 5
    assert len(tuple(instance.view_ref.partitions())) == 6

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=07',
        'year=2000/month=01/day=13'
    ]
    assert len(
        tuple(
            instance.partitions(selected_partitions=selected_partitions))) == 2
    assert len(
        tuple(
            instance.view_ref.partitions(
                selected_partitions=selected_partitions))) == 3

    zds = instance.load(delayed=delayed, distributed=distributed)
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
    instance.add_variable(var, distributed=distributed)

    zds = instance.load(delayed=delayed, distributed=distributed)
    assert zds is not None

    def update(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 0 + 5}

    instance.update(
        update,  # type: ignore
        'var3',
        delayed=delayed,
        distributed=distributed)

    with pytest.raises(ValueError):
        instance.update(
            update,  # type: ignore
            'varX',
            distributed=distributed)

    with pytest.raises(ValueError):
        instance.update(
            update,  # type: ignore
            'var2',
            distributed=distributed)

    zds = instance.load(delayed=delayed, distributed=distributed)
    assert zds is not None
    assert numpy.all(zds.variables['var3'].values == 5)

    indexers = instance.map(
        lambda x: slice(0, x.dimensions['num_lines'])  # type: ignore
    ).compute()
    ds1 = instance.load(delayed=delayed,
                        indexer=indexers,
                        distributed=distributed)
    assert ds1 is not None
    ds2 = instance.load(delayed=delayed, distributed=distributed)
    assert ds2 is not None

    assert numpy.allclose(ds1.variables['var1'].values,
                          ds2.variables['var1'].values)

    instance.drop_variable('var3', distributed=distributed)

    assert tuple(
        str(pathlib.Path(item))
        for item in instance.partitions(filters=instance.sync())) == (str(
            tested_fs.view.joinpath('year=2000', 'month=01', 'day=13')), )

    with pytest.raises(ValueError):
        convenience.open_view(str(tested_fs.collection),
                              filesystem=tested_fs.fs)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_add_variable_immutable(fs, request):
    """Test the creation of a view."""
    tested_fs = request.getfixturevalue(fs)

    create_test_collection(tested_fs, delayed=False)
    instance = convenience.create_view(
        path=str(tested_fs.view),
        view_ref=view.ViewReference(str(tested_fs.collection), tested_fs.fs),
        filesystem=tested_fs.fs,
        distributed=False,
    )

    # Cannot add an immutable variable to a view
    known_dimensions, _ = instance.view_ref.dimensions_properties()
    dim = 'num_pixels'

    new = variable.Array(
        name='var_immutable',
        data=numpy.arange(known_dimensions[dim], dtype='int16'),
        dimensions=(dim, ),
    )

    with pytest.raises(ValueError,
                       match='Immutable variable cannot be added to views'):
        instance.add_variable(variable=new, distributed=False)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_view_update(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
        fs,
        request,
        tmpdir):
    """Test the creation of a view."""
    tested_fs = request.getfixturevalue(fs)

    create_test_collection(tested_fs, delayed=False)
    instance = convenience.create_view(path=str(tested_fs.view),
                                       view_ref=view.ViewReference(
                                           str(tested_fs.collection),
                                           tested_fs.fs),
                                       filesystem=tested_fs.fs)

    var_name = 'var3'
    log_msg = 'Update called'

    var = meta.Variable(name=var_name,
                        dtype=numpy.float64,
                        dimensions=('num_lines', 'num_pixels'))

    instance.add_variable(var)

    def to_zero(zds: dataset.Dataset, varname):
        """Update function used to set a variable to 0."""
        return {varname: zds.variables['var1'].values * 0}

    instance.update(to_zero, var_name)  # type: ignore

    data = instance.load(delayed=False)
    assert numpy.all(data.variables[var_name].values == 0)

    test_dir = tmpdir / 'test'
    test_dir.mkdir()

    def plus_one_with_log(zds: dataset.Dataset, varname):
        """Update function increasing a variable by 1.

        This update function create a new file each time its called.
        """
        with distributed.Lock('update'):
            i = 0
            f = test_dir / f'file_{i}'
            while f.exists():
                i += 1
                f = test_dir / f'file_{i}'

            pathlib.Path(f).touch()

        return {varname: zds.variables[var_name].values + 1}

    instance.update(plus_one_with_log, var_name)  # type: ignore

    # One log per partition + 1 log for the initial call
    assert len(test_dir.listdir()) == len(list(instance.partitions())) + 1

    test_dir.remove()
    test_dir.mkdir()

    data = instance.load(delayed=False)
    assert numpy.all(data.variables[var_name].values == 1)

    instance.update(
        plus_one_with_log,  # type: ignore
        var_name,
        variables=[var_name])

    assert len(test_dir.listdir()) == len(list(instance.partitions()))

    data = instance.load(delayed=False)
    assert numpy.all(data.variables[var_name].values == 2)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_view_overlap(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    request,
):
    """Test the creation of a view."""
    tested_fs = request.getfixturevalue(arg)

    create_test_collection(tested_fs)
    instance = convenience.create_view(str(tested_fs.view),
                                       view.ViewReference(
                                           str(tested_fs.collection),
                                           tested_fs.fs),
                                       filesystem=tested_fs.fs)

    var = meta.Variable(
        name='var3',
        dtype=numpy.int8,
        dimensions=('num_lines', 'num_pixels'),
    )

    instance.add_variable(var)

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

    instance.update(
        update,  # type: ignore
        'var3',
        depth=1,
        trim_result=False)
    zds = instance.load()
    assert zds is not None
    numpy.all(zds.variables['var3'].values == 5)

    instance.update(
        update,  # type: ignore
        'var3',
        depth=1,
        trim=False,
        trim_result=True)
    zds = instance.load()
    assert zds is not None
    numpy.all(zds.variables['var3'].values == 5)

    with pytest.raises(ValueError):
        instance.update(
            update,  #  type: ignore
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

    indexers = instance.map_overlap(
        map_func,  # type: ignore
        depth=1,
    ).compute()

    for _, data in indexers:
        assert isinstance(data, tuple)
        assert len(data) == 2
        assert isinstance(data[0], str)
        assert isinstance(data[1], slice)


def test_view_checksum(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
        tmpdir) -> None:
    """Test the checksum calculation."""
    zds = next(create_test_dataset())
    zcollection = collection.Collection('time', zds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tmpdir))

    zcollection.insert(zds)
    partition = tmpdir / 'year=2000' / 'month=01' / 'day=01'
    axis_ref = _calculate_axis_reference(str(partition), zcollection)
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
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    distributed,
    request,
):
    """Test the synchronization of a view."""
    tested_fs = request.getfixturevalue(arg)
    create_test_collection(tested_fs)
    instance = convenience.create_view(str(tested_fs.view),
                                       view.ViewReference(
                                           str(tested_fs.collection),
                                           tested_fs.fs),
                                       filesystem=tested_fs.fs,
                                       distributed=distributed)
    var = meta.Variable(name='var3',
                        dtype=numpy.float64,
                        dimensions=('num_lines', 'num_pixels'))
    instance.add_variable(var, distributed=distributed)
    del instance

    zcollection = convenience.open_collection(str(tested_fs.collection),
                                              filesystem=tested_fs.fs,
                                              mode='w')
    zds = zcollection.load(filters=lambda keys: keys['year'] == 2000 and keys[
        'month'] == 1 and keys['day'] == 16)
    assert zds is not None
    dates = numpy.arange(numpy.datetime64('2000-01-16'),
                         numpy.datetime64('2000-01-16T23:59:59'),
                         numpy.timedelta64(1, 'h'))
    zds = make_dataset(
        dates.astype('M8[ns]'),
        numpy.ones((len(dates), zds.dimensions['num_pixels']),
                   dtype=numpy.int64))
    zcollection.insert(zds)
    del zcollection
    instance = convenience.open_view(str(tested_fs.view),
                                     filesystem=tested_fs.fs)
    assert instance is not None
    assert instance.is_synced(distributed=distributed) is False
    instance.sync(filters=lambda keys: True, distributed=distributed)
    zds = instance.load(distributed=distributed)
    assert zds is not None
