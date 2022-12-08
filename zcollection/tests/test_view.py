# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of views
=============
"""
from __future__ import annotations

import numpy
import pytest

from .. import collection, convenience, meta, partitioning, view
from ..view.detail import _checksum
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster
from .data import create_test_collection, create_test_dataset
from .fs import local_fs, s3, s3_base, s3_fs

# pylint: enable=unused-import


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_view(
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
    assert isinstance(instance, view.View)
    assert isinstance(str(instance), str)

    # No variable recorded, so no data can be loaded
    with pytest.raises(ValueError):
        instance.load()

    var = meta.Variable(
        name='var2',
        dtype=numpy.float64,
        dimensions=('num_lines', 'num_pixels'),
        attrs=(meta.Attribute(name='attr', value=1), ),
    )

    with pytest.raises(ValueError):
        instance.add_variable(var)

    var.name = 'var3'
    instance.add_variable(var)

    with pytest.raises(ValueError):
        instance.add_variable(var)

    instance = convenience.open_view(str(tested_fs.view),
                                     filesystem=tested_fs.fs)
    ds = instance.load()
    assert ds is not None
    assert set(ds['time'].values.astype('datetime64[D]')) == {
        numpy.datetime64('2000-01-01'),
        numpy.datetime64('2000-01-04'),
        numpy.datetime64('2000-01-07'),
        numpy.datetime64('2000-01-10'),
        numpy.datetime64('2000-01-13'),
        numpy.datetime64('2000-01-16'),
    }

    # Loading a variable existing only in the view.
    ds = instance.load(selected_variables=('var3', ))
    assert ds is not None
    assert tuple(ds.variables) == ('var3', )

    # Loading a non existing variable.
    ds = instance.load(selected_variables=('var55', ))
    assert ds is not None
    assert len(ds.variables) == 0

    # Test view loading that is no longer synchronized with the reference
    # collection.
    tested_fs.fs.rm(str(
        tested_fs.view.joinpath('year=2000', 'month=01', 'day=13')),
                    recursive=True)

    assert len(tuple(instance.partitions())) == 5
    assert len(tuple(instance.view_ref.partitions())) == 6

    ds = instance.load()
    assert ds is not None
    assert set(ds['time'].values.astype('datetime64[D]')) == {
        numpy.datetime64('2000-01-01'),
        numpy.datetime64('2000-01-04'),
        numpy.datetime64('2000-01-07'),
        numpy.datetime64('2000-01-10'),
        numpy.datetime64('2000-01-16'),
    }

    # Create a variable with the unsynchronized view
    var.name = 'var4'
    instance.add_variable(var)

    ds = instance.load()
    assert ds is not None

    def update(ds, varname):
        """Update function used for this test."""
        return {varname: ds.variables['var1'].values * 0 + 5}

    instance.update(update, 'var3')  # type: ignore

    with pytest.raises(ValueError):
        instance.update(update, 'varX')  # type: ignore

    with pytest.raises(ValueError):
        instance.update(update, 'var2')  # type: ignore

    ds = instance.load()
    assert ds is not None
    numpy.all(ds.variables['var3'].values == 5)

    indexers = instance.map(
        lambda x: slice(0, x.dimensions['num_lines'])  # type: ignore
    ).compute()
    ds1 = instance.load(indexer=indexers)
    assert ds1 is not None
    ds2 = instance.load()
    assert ds2 is not None

    assert numpy.allclose(ds1.variables['var1'].values,
                          ds2.variables['var1'].values)

    instance.drop_variable('var3')

    assert tuple(instance.partitions(filters=instance.sync())) == (str(
        tested_fs.view.joinpath('year=2000', 'month=01', 'day=13')), )

    with pytest.raises(ValueError):
        convenience.open_view(str(tested_fs.collection),
                              filesystem=tested_fs.fs)


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

    def update(ds, varname, partition_info: tuple[str, slice]):
        """Update function used for this test."""
        assert isinstance(partition_info, tuple)
        assert len(partition_info) == 2
        assert isinstance(partition_info[0], str)
        assert isinstance(partition_info[1], slice)
        assert partition_info[0] == 'num_lines'
        return {varname: ds.variables['var1'].values * 1 + 5}

    instance.update(update, 'var3', depth=1)  # type: ignore

    ds = instance.load()
    assert ds is not None
    numpy.all(ds.variables['var3'].values == 5)

    def map_func(x, partition_info: tuple[str, slice]):
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
        tmpdir):
    ds = next(create_test_dataset())
    zcollection = collection.Collection('time', ds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tmpdir))

    zcollection.insert(ds)
    partition = tmpdir / 'year=2000' / 'month=01' / 'day=01'
    sha256 = _checksum(str(partition), zcollection)
    assert isinstance(sha256, str)
    assert len(sha256) == 64
    assert sha256 == ('4bbb9253c07f36002098f8bba57151eb'
                      '0143f5fe5c634950340f3e2f1a4f51cf')
