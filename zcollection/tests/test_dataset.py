# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing datasets
================
"""
import pickle

import numpy
import pytest
import xarray
import zarr

from .. import dataset, meta
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster
# pylint enable=unused-import
from .test_variable import create_test_variable


def create_test_dataset():
    """Create a test dataset."""
    return dataset.Dataset(attrs=(dataset.Attribute(name='attr', value=1), ),
                           variables=(create_test_variable(),
                                      create_test_variable('var2')))


def test_dataset(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset creation."""
    ds = create_test_dataset()
    assert ds.dimensions == dict(x=5, y=2)
    assert ds.attrs == (dataset.Attribute(name='attr', value=1), )
    assert isinstance(str(ds), str)
    assert isinstance(repr(ds), str)
    assert ds.nbytes == 5 * 2 * 8 * 2

    var1 = create_test_variable()
    assert numpy.all(ds.variables['var1'].values == var1.values)
    assert ds.variables['var1'].metadata() == var1.metadata()
    assert numpy.all(var1.values == ds['var1'].values)
    assert var1.metadata() == ds['var1'].metadata()
    with pytest.raises(KeyError):
        var1 = ds['varX']

    var2 = create_test_variable('var2')
    assert numpy.all(ds.variables['var2'].values == var2.values)
    assert id(ds['var2']) == id(ds.variables['var2'])
    assert ds.variables['var2'].metadata() == var2.metadata()
    assert isinstance(ds.metadata(), meta.Dataset)

    other = ds.compute()
    assert isinstance(other, dataset.Dataset)

    ds_dict = ds.to_dict()
    assert isinstance(ds_dict, dict)
    assert isinstance(ds_dict['var1'], numpy.ndarray)
    assert isinstance(ds_dict['var2'], numpy.ndarray)

    ds_dict = ds.to_dict(variables=['var1'])
    assert isinstance(ds_dict, dict)
    assert isinstance(ds_dict['var1'], numpy.ndarray)
    assert 'var2' not in ds_dict


def test_dataset_dimensions_conflict(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset creation with dimensions conflict."""
    with pytest.raises(ValueError):
        dataset.Dataset([
            dataset.Variable(name='var1',
                             data=numpy.arange(10,
                                               dtype='int64').reshape(5, 2),
                             dimensions=('x', 'y'),
                             attrs=(dataset.Attribute(name='attr', value=1), ),
                             compressor=zarr.Blosc(cname='zstd', clevel=1),
                             fill_value=0,
                             filters=(zarr.Delta('int64', 'int32'),
                                      zarr.Delta('int32', 'int32'))),
            dataset.Variable(name='var2',
                             data=numpy.arange(20, dtype='int64'),
                             dimensions=('x'),
                             attrs=(dataset.Attribute(name='attr', value=1), ),
                             compressor=zarr.Blosc(cname='zstd', clevel=1),
                             fill_value=0,
                             filters=(zarr.Delta('int64', 'int32'),
                                      zarr.Delta('int32', 'int32'))),
        ])


def test_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test xarray creation."""
    ds = create_test_dataset()
    xr1 = ds.to_xarray()
    assert isinstance(xr1, xarray.Dataset)

    xr2 = dataset.Dataset.from_xarray(xr1).to_xarray()
    assert xr1 == xr2


def test_dataset_isel(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset selection."""
    ds = create_test_dataset()
    selected_ds = ds.isel(slices=dict(x=slice(0, 2)))
    assert selected_ds.dimensions == dict(x=2, y=2)
    assert selected_ds.attrs == (dataset.Attribute(name='attr', value=1), )
    assert numpy.all(
        selected_ds.variables['var1'].values == numpy.arange(4).reshape(2, 2))
    selected_ds = ds.isel(slices=dict(y=slice(0, 1)))
    assert selected_ds.dimensions == dict(x=5, y=1)
    assert numpy.all(selected_ds.variables['var1'].values == numpy.arange(
        0, 10, 2).reshape(5, 1))

    # Cannot slice on something which is not a dimension
    with pytest.raises(ValueError, match='invalid dimension'):
        ds.isel(slices=dict(z=slice(0, 1)))

    with pytest.raises(ValueError, match='invalid dimension'):
        ds.isel(slices=dict(var1=slice(0, 1)))


def test_dataset_delete(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test dataset deletion."""
    ds = create_test_dataset()

    other = ds.delete([1], 'y')
    assert numpy.all(
        other.variables['var1'].values == numpy.arange(0, 10, 2).reshape(5, 1))
    assert numpy.all(
        other.variables['var2'].values == numpy.arange(0, 10, 2).reshape(5, 1))

    other = ds.delete([0], 'x')
    assert numpy.all(other.variables['var1'].values == numpy.arange(
        10).reshape(5, 2)[1:, :])


def test_dataset_concat(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test concatenation of datasets."""
    ds1 = create_test_dataset()
    ds2 = create_test_dataset()
    ds3 = ds1.concat(ds2, 'y')

    matrix = numpy.arange(10).reshape(5, 2)
    assert numpy.all(ds3.variables['var1'].values == numpy.concatenate(
        (matrix, matrix), axis=1))

    ds3 = ds1.concat(ds2, 'x')
    assert numpy.all(ds3.variables['var1'].values == numpy.concatenate(
        (matrix, matrix), axis=0))

    with pytest.raises(ValueError):
        ds1.concat([], 'z')


def test_variable_pickle(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test pickling of variables."""
    variable = create_test_variable()
    other = pickle.loads(pickle.dumps(variable))
    assert numpy.all(variable.values == other.values)
    assert variable.attrs == other.attrs
    assert variable.compressor == other.compressor
    assert variable.dimensions == other.dimensions
    assert variable.dtype == other.dtype
    assert variable.fill_value == other.fill_value
    assert variable.filters == other.filters
    assert variable.name == other.name


def test_dataset_pickle(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test pickling of datasets."""
    ds = create_test_dataset()
    other = pickle.loads(pickle.dumps(ds))
    assert ds.attrs == other.attrs
    assert list(ds.variables) == list(other.variables)
    for varname, variable in ds.variables.items():
        assert variable.attrs == other.variables[varname].attrs
        assert variable.compressor == other.variables[varname].compressor
        assert variable.dimensions == other.variables[varname].dimensions
        assert variable.dtype == other.variables[varname].dtype
        assert variable.fill_value == other.variables[varname].fill_value
        assert variable.filters == other.variables[varname].filters
        assert variable.name == other.variables[varname].name


def test_dataset_add_variable(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test for adding a variable."""
    ds = create_test_dataset()
    var = meta.Variable('var2', numpy.int64,
                        ('x', 'y'), (dataset.Attribute('attr', 1), ),
                        zarr.Blosc(), 255, (zarr.Delta('int64', 'int64'), ))
    ds.add_variable(var)

    assert ds.variables['var2'].attrs == (dataset.Attribute(name='attr',
                                                            value=1), )
    assert ds.variables['var2'].dtype == numpy.int64
    assert ds.variables['var2'].dimensions == ('x', 'y')
    assert ds.variables['var2'].compressor == zarr.Blosc()
    assert ds.variables['var2'].filters == (zarr.Delta('int64', 'int64'), )
    assert ds.variables['var2'].fill_value == 255
    assert ds.variables['var2'].name == 'var2'
    assert ds.variables['var2'].shape == (5, 2)
    assert numpy.ma.allequal(
        ds.variables['var2'].values,
        numpy.ma.masked_equal(numpy.full((5, 2), 255, 'int64'), 255))

    other = ds.select_vars(['var1'])
    assert list(other.variables) == ['var1']

    data = numpy.ones((5, 2), 'int64')
    ds.drops_vars('var2')
    assert 'var2' not in ds.variables

    var = meta.Variable('var2', numpy.int64,
                        ('x', 'y'), (dataset.Attribute('attr', 1), ),
                        zarr.Blosc(), 255, (zarr.Delta('int64', 'int64'), ))
    ds.add_variable(var, data)
    assert numpy.ma.allequal(
        ds.variables['var2'].values,
        numpy.ma.masked_equal(numpy.full((5, 2), 1, 'int64'), 255))

    data = numpy.ones((10, 2), 'int64')
    with pytest.raises(ValueError, match='Conflicting sizes'):
        ds.add_variable(var, data)

    var = meta.Variable('var2', numpy.int64,
                        ('xx', 'y'), (dataset.Attribute('attr', 1), ),
                        zarr.Blosc(), 255, (zarr.Delta('int64', 'int64'), ))
    with pytest.raises(ValueError, match='has dimension'):
        ds.add_variable(var)


def test_empty_dataset(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test empty dataset."""
    ds = dataset.Dataset([], [])
    assert ds.attrs == ()
    assert list(ds.variables) == []
    assert (str(ds)) == """<zcollection.dataset.Dataset>
  Dimensions: ()
Data variables:
    <empty>"""


def test_dataset_rename(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test renaming of datasets."""
    ds = create_test_dataset()
    ds.rename(dict(var1='var3', var2='var4'))
    assert ds.variables['var3'].name == 'var3'
    assert ds.variables['var4'].name == 'var4'
    assert 'var1' not in ds.variables
    assert 'var2' not in ds.variables

    with pytest.raises(ValueError):
        ds.rename(dict(var3='var4'))


def test_dataset_persist(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test persisting of datasets."""
    ds1 = create_test_dataset()
    ds1.persist()
    ds1.persist()
    assert ds1.variables['var1'].values is not None
    assert ds1.variables['var2'].values is not None

    ds2 = create_test_dataset()
    ds2.persist(compress=True)
    ds2.persist(compress=True)
    assert ds2.variables['var1'].values is not None
    assert ds2.variables['var2'].values is not None

    assert numpy.all(
        ds1.variables['var1'].values == ds2.variables['var1'].values)
    assert numpy.all(
        ds1.variables['var2'].values == ds2.variables['var2'].values)


def test_dataset_rechunk(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test rechunking of datasets."""
    ds1 = create_test_dataset()
    ds2 = create_test_dataset()
    ds2 = ds2.rechunk()

    assert numpy.all(
        ds1.variables['var1'].values == ds2.variables['var1'].values)
    assert numpy.all(
        ds1.variables['var2'].values == ds2.variables['var2'].values)

    ds2 = ds1.rechunk().persist(compress=True)
    assert numpy.all(
        ds1.variables['var1'].values == ds2.variables['var1'].values)
    assert numpy.all(
        ds1.variables['var2'].values == ds2.variables['var2'].values)


def test_dataset_merge(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    template = dataset.Dataset([
        dataset.Variable('var1', numpy.empty(10), ('x', ), ()),
        dataset.Variable('var2', numpy.empty(10), ('x', ), ()),
        dataset.Variable('var3', numpy.empty((10, 10)), ('x', 'y'), ()),
        dataset.Variable('var4', numpy.empty((10, 10)), ('x', 'y'), ()),
    ], [dataset.Attribute('attr1', 1),
        dataset.Attribute('attr2', 2)])
    ds2 = dataset.Dataset([
        dataset.Variable('var5', numpy.empty(10), ('x', ), ()),
        dataset.Variable('var6', numpy.empty(10), ('x', ), ()),
        dataset.Variable('var7', numpy.empty((10, 10)), ('x', 'y'), ()),
        dataset.Variable('var8', numpy.empty((10, 10)), ('x', 'y'), ()),
    ], [dataset.Attribute('attr3', 3),
        dataset.Attribute('attr4', 4)])
    ds1 = pickle.loads(pickle.dumps(template))
    ds1.merge(ds2)
    assert list(ds1.variables) == [
        'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8'
    ]
    assert ds1.dimensions == {'x': 10, 'y': 10}
    assert ds1.variables['var1'].shape == (10, )
    assert ds1.variables['var2'].shape == (10, )
    assert ds1.variables['var3'].shape == (10, 10)
    assert ds1.variables['var4'].shape == (10, 10)
    assert ds1.variables['var5'].shape == (10, )
    assert ds1.variables['var6'].shape == (10, )
    assert ds1.variables['var7'].shape == (10, 10)
    assert ds1.variables['var8'].shape == (10, 10)
    assert ds1.attrs == (dataset.Attribute('attr1',
                                           1), dataset.Attribute('attr2', 2),
                         dataset.Attribute('attr3',
                                           3), dataset.Attribute('attr4', 4))

    ds2 = dataset.Dataset([
        dataset.Variable('var5', numpy.empty(10), ('a', ), ()),
        dataset.Variable('var6', numpy.empty((10, 10)), ('a', 'b'), ()),
    ], [dataset.Attribute('attr3', 3),
        dataset.Attribute('attr4', 4)])
    ds1 = pickle.loads(pickle.dumps(template))
    ds1.merge(ds2)
    assert list(
        ds1.variables) == ['var1', 'var2', 'var3', 'var4', 'var5', 'var6']
    assert ds1.dimensions == {'x': 10, 'y': 10, 'a': 10, 'b': 10}
    assert ds1.variables['var1'].shape == (10, )
    assert ds1.variables['var2'].shape == (10, )
    assert ds1.variables['var3'].shape == (10, 10)
    assert ds1.variables['var4'].shape == (10, 10)
    assert ds1.variables['var5'].shape == (10, )
    assert ds1.variables['var6'].shape == (10, 10)

    ds2 = dataset.Dataset([
        dataset.Variable('var5', numpy.empty(10), ('z', ), ()),
    ], [dataset.Attribute('attr3', 3),
        dataset.Attribute('attr4', 4)])
    ds1 = pickle.loads(pickle.dumps(template))
    ds1.merge(ds2)
    assert list(ds1.variables) == ['var1', 'var2', 'var3', 'var4', 'var5']
    assert ds1.dimensions == {'x': 10, 'y': 10, 'z': 10}
    assert ds1.variables['var1'].shape == (10, )
    assert ds1.variables['var2'].shape == (10, )
    assert ds1.variables['var3'].shape == (10, 10)
    assert ds1.variables['var4'].shape == (10, 10)
    assert ds1.variables['var5'].shape == (10, )

    ds2 = dataset.Dataset([
        dataset.Variable('var1', numpy.empty(10), ('x', ), ()),
    ], [dataset.Attribute('attr1', 1),
        dataset.Attribute('attr2', 2)])
    ds1 = pickle.loads(pickle.dumps(template))
    with pytest.raises(ValueError):
        ds1.merge(ds2)

    ds2 = dataset.Dataset([
        dataset.Variable('var5', numpy.empty(20), ('x', ), ()),
    ], [dataset.Attribute('attr3', 3),
        dataset.Attribute('attr4', 4)])
    ds1 = pickle.loads(pickle.dumps(template))
    with pytest.raises(ValueError):
        ds1.merge(ds2)


def test_dataset_select_variables_by_dims(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    ds = dataset.Dataset([
        dataset.Variable('var1', numpy.empty(10), ('x', ), ()),
        dataset.Variable('var2', numpy.empty(10), ('y', ), ()),
        dataset.Variable('var3', numpy.empty((10, 10)), ('x', 'y'), ()),
        dataset.Variable('var4', numpy.empty(20), ('a', ), ()),
        dataset.Variable('var5', numpy.empty(20), ('b', ), ()),
        dataset.Variable('var6', numpy.empty((20, 20)), ('a', 'b'), ()),
        dataset.Variable('var7', numpy.int64(1), (), ()),
    ], [
        dataset.Attribute('attr1', 1),
    ])
    selected = ds.select_variables_by_dims(('x', ))
    assert list(selected.variables) == ['var1', 'var3']
    assert selected.dimensions == {'x': 10, 'y': 10}

    selected = ds.select_variables_by_dims(('x', 'y'))
    assert list(selected.variables) == ['var1', 'var2', 'var3']
    assert selected.dimensions == {'x': 10, 'y': 10}

    selected = ds.select_variables_by_dims(('x', 'y', 'z'))
    assert list(selected.variables) == ['var1', 'var2', 'var3']
    assert selected.dimensions == {'x': 10, 'y': 10}

    selected = ds.select_variables_by_dims(('a', ))
    assert list(selected.variables) == ['var4', 'var6']
    assert selected.dimensions == {'a': 20, 'b': 20}

    selected = ds.select_variables_by_dims(('a', 'b'))
    assert list(selected.variables) == ['var4', 'var5', 'var6']
    assert selected.dimensions == {'a': 20, 'b': 20}

    selected = ds.select_variables_by_dims(('a', 'b', 'c'))
    assert list(selected.variables) == ['var4', 'var5', 'var6']
    assert selected.dimensions == {'a': 20, 'b': 20}

    selected = ds.select_variables_by_dims(('x', 'a'))
    assert list(selected.variables) == ['var1', 'var3', 'var4', 'var6']
    assert selected.dimensions == {'x': 10, 'y': 10, 'a': 20, 'b': 20}

    selected = ds.select_variables_by_dims(tuple())
    assert list(selected.variables) == ['var7']
    assert selected.dimensions == {}

    selected = ds.select_variables_by_dims(tuple(), predicate=False)
    assert list(selected.variables) == [
        'var1', 'var2', 'var3', 'var4', 'var5', 'var6'
    ]
    assert selected.dimensions == {'x': 10, 'y': 10, 'a': 20, 'b': 20}

    selected = ds.select_variables_by_dims(('x', ), predicate=False)
    assert list(selected.variables) == ['var2', 'var4', 'var5', 'var6', 'var7']
