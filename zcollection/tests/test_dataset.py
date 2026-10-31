# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing datasets
================
"""
from typing import TYPE_CHECKING
import pickle

import dask.array.core
import dask.distributed
import numcodecs
import numpy
import pytest
import xarray

from .. import dataset, meta
from ..compressed_array import CompressedArray
from ..variable.tests.data import array, delayed_array
from .cluster import dask_client, dask_cluster  # noqa: F401

if TYPE_CHECKING:
    from ..variable import Variable

# pylint enable=unused-import


def create_delayed_dataset(factory) -> dataset.Dataset:
    """Create a test dataset."""
    return dataset.Dataset(attrs=(dataset.Attribute(name='attr', value=1), ),
                           variables=(factory(), factory('var2')))


def test_compute_preserves_filler_flags() -> None:
    """Computing preserves each variable's filler flag."""
    zds = dataset.Dataset([
        dataset.DelayedArray('provided', numpy.arange(3), ('row', )),
    ])
    zds.add_variable(meta.VariableSchema('missing',
                                         'int64',
                                         dimensions=('row', ),
                                         fill_value=-1),
                     filler=True)

    computed = zds.compute(scheduler='synchronous')

    assert {
        name: var.filler
        for name, var in computed.variables.items()
    } == {
        'provided': False,
        'missing': True,
    }
    assert all(
        isinstance(var, dataset.Array) for var in computed.variables.values())
    numpy.testing.assert_array_equal(computed['provided'].array,
                                     numpy.arange(3))
    numpy.testing.assert_array_equal(computed['missing'].array,
                                     numpy.full(3, -1))


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset(factory) -> None:
    """Test dataset creation."""
    zds = create_delayed_dataset(factory)
    assert zds.dimensions == {'x': 5, 'y': 2}
    assert zds.attrs == (dataset.Attribute(name='attr', value=1), )
    assert isinstance(str(zds), str)
    assert isinstance(repr(zds), str)
    assert zds.nbytes == 5 * 2 * 8 * 2

    var1: Variable = factory()
    assert numpy.all(zds.variables['var1'].values == var1.values)
    assert zds.variables['var1'].metadata() == var1.metadata()
    assert numpy.all(var1.values == zds['var1'].values)
    assert var1.metadata() == zds['var1'].metadata()
    assert numpy.all(zds.variables['var1'][:] == var1.values)
    with pytest.raises(KeyError):
        print(zds['varX'])

    var2: Variable = factory('var2')
    assert numpy.all(zds.variables['var2'].values == var2.values)
    assert id(zds['var2']) == id(zds.variables['var2'])
    assert zds.variables['var2'].metadata() == var2.metadata()
    assert isinstance(zds.metadata(), meta.DatasetSchema)

    other = zds.compute()
    assert isinstance(other, dataset.Dataset)

    ds_dict = zds.to_dict()
    assert isinstance(ds_dict, dict)
    assert isinstance(ds_dict['var1'], numpy.ndarray)
    assert isinstance(ds_dict['var2'], numpy.ndarray)

    ds_dict = zds.to_dict(variables=['var1'])
    assert isinstance(ds_dict, dict)
    assert isinstance(ds_dict['var1'], numpy.ndarray)
    assert 'var2' not in ds_dict

    zds = zds.compute()
    assert isinstance(zds, dataset.Dataset)
    assert isinstance(zds.var1, dataset.Array)
    assert isinstance(zds.var2, dataset.Array)


def test_dataset_dimensions_conflict() -> None:
    """Test dataset creation with dimensions conflict."""
    with pytest.raises(ValueError,
                       match=r'variable .* has conflicting dimensions'):
        dataset.Dataset([
            dataset.DelayedArray(name='var1',
                                 data=numpy.arange(10, dtype='int64').reshape(
                                     5, 2),
                                 dimensions=('x', 'y'),
                                 attrs=(dataset.Attribute(name='attr',
                                                          value=1), ),
                                 compressor=numcodecs.Blosc(cname='zstd',
                                                            clevel=1),
                                 fill_value=0,
                                 filters=(numcodecs.Delta('int64', 'int32'),
                                          numcodecs.Delta('int32', 'int32'))),
            dataset.DelayedArray(name='var2',
                                 data=numpy.arange(20, dtype='int64'),
                                 dimensions=('x'),
                                 attrs=(dataset.Attribute(name='attr',
                                                          value=1), ),
                                 compressor=numcodecs.Blosc(cname='zstd',
                                                            clevel=1),
                                 fill_value=0,
                                 filters=(numcodecs.Delta('int64', 'int32'),
                                          numcodecs.Delta('int32', 'int32'))),
        ])


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_xarray(factory) -> None:
    """Test xarray creation."""
    zds = create_delayed_dataset(factory)
    xr1 = zds.to_xarray()
    assert isinstance(xr1, xarray.Dataset)

    xr2 = dataset.Dataset.from_xarray(xr1).to_xarray()
    xarray.testing.assert_identical(xr1, xr2)


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_isel(factory) -> None:
    """Test dataset selection."""
    zds = create_delayed_dataset(factory)
    selected_zds = zds.isel(slices={'x': slice(0, 2)})
    assert selected_zds.dimensions == {'x': 2, 'y': 2}
    assert selected_zds.attrs == (dataset.Attribute(name='attr', value=1), )
    assert numpy.all(
        selected_zds.variables['var1'].values == numpy.arange(4).reshape(2, 2))
    selected_zds = zds.isel(slices={'y': slice(0, 1)})
    assert selected_zds.dimensions == {'x': 5, 'y': 1}
    assert numpy.all(selected_zds.variables['var1'].values == numpy.arange(
        0, 10, 2).reshape(5, 1))

    # Cannot slice on something which is not a dimension
    with pytest.raises(ValueError, match='invalid dimension'):
        zds.isel(slices={'z': slice(0, 1)})

    with pytest.raises(ValueError, match='invalid dimension'):
        zds.isel(slices={'var1': slice(0, 1)})


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_delete(factory) -> None:
    """Test dataset deletion."""
    zds = create_delayed_dataset(factory)

    other = zds.delete([1], 'y')
    assert numpy.all(
        other.variables['var1'].values == numpy.arange(0, 10, 2).reshape(5, 1))
    assert numpy.all(
        other.variables['var2'].values == numpy.arange(0, 10, 2).reshape(5, 1))

    other = zds.delete([0], 'x')
    assert numpy.all(other.variables['var1'].values == numpy.arange(
        10).reshape(5, 2)[1:, :])


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_concat(factory) -> None:
    """Test concatenation of datasets."""
    zds1 = create_delayed_dataset(factory)
    zds2 = create_delayed_dataset(factory)
    zds3 = zds1.concat(zds2, 'y')

    matrix = numpy.arange(10).reshape(5, 2)
    assert numpy.all(zds3.variables['var1'].values == numpy.concatenate(
        (matrix, matrix), axis=1))

    zds3 = zds1.concat(zds2, 'x')
    assert numpy.all(zds3.variables['var1'].values == numpy.concatenate(
        (matrix, matrix), axis=0))

    with pytest.raises(ValueError,
                       match='cannot concatenate an empty sequence'):
        zds1.concat([], 'z')


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_concat_holding_different_variables(factory) -> None:
    """Concatenating datasets that do not hold the same variables is an
    error."""
    zds1 = create_delayed_dataset(factory)
    zds2 = create_delayed_dataset(factory)

    with pytest.raises(ValueError, match='var2'):
        zds1.select_vars(names=['var1']).concat(zds2, 'x')

    with pytest.raises(ValueError, match='var2'):
        zds1.concat(zds2.select_vars(names=['var1']), 'x')


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_pickle(factory) -> None:
    """Test pickling of datasets."""
    zds = create_delayed_dataset(factory)
    other = pickle.loads(pickle.dumps(zds))
    assert zds.attrs == other.attrs
    assert list(zds.variables) == list(other.variables)
    for varname, variable in zds.variables.items():
        assert variable.attrs == other.variables[varname].attrs
        assert variable.compressor == other.variables[varname].compressor
        assert variable.dimensions == other.variables[varname].dimensions
        assert variable.dtype == other.variables[varname].dtype
        assert variable.fill_value == other.variables[varname].fill_value
        assert variable.filters == other.variables[varname].filters
        assert variable.name == other.variables[varname].name


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_add_variable(factory) -> None:
    """Test for adding a variable."""
    zds = create_delayed_dataset(factory)
    var = meta.VariableSchema('var2',
                              numpy.int64,
                              dimensions=('x', 'y'),
                              attrs=(dataset.Attribute('attr', 1), ),
                              compressor=numcodecs.Blosc(),
                              fill_value=255,
                              filters=(numcodecs.Delta('int64', 'int64'), ))
    zds.add_variable(var)

    assert zds.variables['var2'].attrs == (dataset.Attribute(name='attr',
                                                             value=1), )
    assert zds.variables['var2'].dtype == numpy.int64
    assert zds.variables['var2'].dimensions == ('x', 'y')
    assert zds.variables['var2'].compressor == numcodecs.Blosc()
    assert zds.variables['var2'].filters == (numcodecs.Delta('int64',
                                                             'int64'), )
    assert zds.variables['var2'].fill_value == 255
    assert zds.variables['var2'].name == 'var2'
    assert zds.variables['var2'].shape == (5, 2)
    assert numpy.ma.allequal(
        zds.variables['var2'].values,
        numpy.ma.masked_equal(numpy.full((5, 2), 255, 'int64'), 255))

    other = zds.select_vars(['var1'])
    assert list(other.variables) == ['var1']

    data = numpy.ones((5, 2), 'int64')
    zds.drops_vars('var2')
    assert 'var2' not in zds.variables

    var = meta.VariableSchema('var2',
                              numpy.int64,
                              dimensions=('x', 'y'),
                              attrs=(dataset.Attribute('attr', 1), ),
                              compressor=numcodecs.Blosc(),
                              fill_value=255,
                              filters=(numcodecs.Delta('int64', 'int64'), ))
    zds.add_variable(var, data)
    assert numpy.ma.allequal(
        zds.variables['var2'].values,
        numpy.ma.masked_equal(numpy.full((5, 2), 1, 'int64'), 255))

    data = numpy.ones((10, 2), 'int64')
    with pytest.raises(ValueError, match='Conflicting sizes'):
        zds.add_variable(var, data)

    var = meta.VariableSchema('var2',
                              numpy.int64,
                              dimensions=('xx', 'y'),
                              attrs=(dataset.Attribute('attr', 1), ),
                              compressor=numcodecs.Blosc(),
                              fill_value=255,
                              filters=(numcodecs.Delta('int64', 'int64'), ))
    with pytest.raises(ValueError, match='has dimension'):
        zds.add_variable(var)


def test_empty_dataset() -> None:
    """Test empty dataset."""
    zds = dataset.Dataset([], attrs=[])
    assert zds.attrs == ()
    assert not list(zds.variables)
    assert (str(zds)) == """<zcollection.dataset.Dataset>
  Dimensions: ()
Data variables:
    <empty>"""


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_rename(factory) -> None:
    """Test renaming of datasets."""
    zds = create_delayed_dataset(factory)
    zds.rename({'var1': 'var3', 'var2': 'var4'})
    assert zds.variables['var3'].name == 'var3'
    assert zds.variables['var4'].name == 'var4'
    assert 'var1' not in zds.variables
    assert 'var2' not in zds.variables

    with pytest.raises(ValueError, match=r'.* already exists in the dataset'):
        zds.rename({'var3': 'var4'})


@pytest.mark.parametrize(('factory', 'distributed'), [
    (array, False),
    (delayed_array, False),
    (delayed_array, True),
],
                         ids=['numpy', 'dask-local', 'dask-distributed'])
def test_dataset_persist(factory, distributed, request) -> None:
    """Test persisting of datasets."""
    client = request.getfixturevalue('dask_client') if distributed else None
    zds1 = create_delayed_dataset(factory)
    zds1.persist()
    zds1.persist()
    assert zds1.variables['var1'].values is not None
    assert zds1.variables['var2'].values is not None

    zds2 = create_delayed_dataset(factory)
    zds2.persist(compress=True)
    zds2.persist(compress=True)
    assert zds2.variables['var1'].values is not None
    assert zds2.variables['var2'].values is not None

    assert numpy.all(
        zds1.variables['var1'].values == zds2.variables['var1'].values)
    assert numpy.all(
        zds1.variables['var2'].values == zds2.variables['var2'].values)

    if client is not None:
        blocks = client.gather(
            dask.distributed.futures_of(
                [var.array for var in zds2.variables.values()]))
        assert blocks
        assert all(isinstance(block, CompressedArray) for block in blocks)
        computed = zds2.compute()
        assert not computed.delayed
        for name in zds1.variables:
            numpy.testing.assert_array_equal(computed[name].values,
                                             zds1[name].values)


def test_variable_metadata_rejects_another_type() -> None:
    """Variable metadata conversion rejects unsupported objects."""
    with pytest.raises(TypeError, match=r'Expected a meta.VariableSchema'):
        dataset.get_variable_metadata({
            'name': 'anomaly',
            'dtype': 'float32',
            'dimensions': ('num_lines', ),
        })  # type: ignore[arg-type]


@pytest.mark.parametrize('factory', [array, delayed_array])
def test_dataset_rechunk(factory) -> None:
    """Test rechunking of datasets."""
    zds1 = create_delayed_dataset(factory)
    zds2 = create_delayed_dataset(factory)
    zds2 = zds2.rechunk()

    assert numpy.all(
        zds1.variables['var1'].values == zds2.variables['var1'].values)
    assert numpy.all(
        zds1.variables['var2'].values == zds2.variables['var2'].values)

    zds2 = zds1.rechunk().persist(compress=True)
    assert numpy.all(
        zds1.variables['var1'].values == zds2.variables['var1'].values)
    assert numpy.all(
        zds1.variables['var2'].values == zds2.variables['var2'].values)


def test_dataset_merge() -> None:
    """Test merging of datasets."""
    template = dataset.Dataset(
        [
            dataset.DelayedArray('var1', numpy.empty(10), ('x', )),
            dataset.DelayedArray('var2', numpy.empty(10), ('x', )),
            dataset.DelayedArray('var3', numpy.empty((10, 10)), ('x', 'y')),
            dataset.DelayedArray('var4', numpy.empty((10, 10)), ('x', 'y')),
        ],
        attrs=[dataset.Attribute('attr1', 1),
               dataset.Attribute('attr2', 2)])
    zds2 = dataset.Dataset(
        [
            dataset.DelayedArray('var5', numpy.empty(10), ('x', )),
            dataset.DelayedArray('var6', numpy.empty(10), ('x', )),
            dataset.DelayedArray('var7', numpy.empty((10, 10)), ('x', 'y')),
            dataset.DelayedArray('var8', numpy.empty((10, 10)), ('x', 'y')),
        ],
        attrs=[dataset.Attribute('attr3', 3),
               dataset.Attribute('attr4', 4)])
    zds1 = pickle.loads(pickle.dumps(template))
    zds1.merge(zds2)
    assert list(zds1.variables) == [
        'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8'
    ]
    assert zds1.dimensions == {'x': 10, 'y': 10}
    assert zds1.variables['var1'].shape == (10, )
    assert zds1.variables['var2'].shape == (10, )
    assert zds1.variables['var3'].shape == (10, 10)
    assert zds1.variables['var4'].shape == (10, 10)
    assert zds1.variables['var5'].shape == (10, )
    assert zds1.variables['var6'].shape == (10, )
    assert zds1.variables['var7'].shape == (10, 10)
    assert zds1.variables['var8'].shape == (10, 10)
    assert zds1.attrs == (dataset.Attribute('attr1',
                                            1), dataset.Attribute('attr2', 2),
                          dataset.Attribute('attr3',
                                            3), dataset.Attribute('attr4', 4))

    zds2 = dataset.Dataset(
        [
            dataset.DelayedArray('var5', numpy.empty(10), ('a', )),
            dataset.DelayedArray('var6', numpy.empty((10, 10)), ('a', 'b')),
        ],
        attrs=[dataset.Attribute('attr3', 3),
               dataset.Attribute('attr4', 4)])
    zds1 = pickle.loads(pickle.dumps(template))
    zds1.merge(zds2)
    assert list(
        zds1.variables) == ['var1', 'var2', 'var3', 'var4', 'var5', 'var6']
    assert zds1.dimensions == {'x': 10, 'y': 10, 'a': 10, 'b': 10}
    assert zds1.variables['var1'].shape == (10, )
    assert zds1.variables['var2'].shape == (10, )
    assert zds1.variables['var3'].shape == (10, 10)
    assert zds1.variables['var4'].shape == (10, 10)
    assert zds1.variables['var5'].shape == (10, )
    assert zds1.variables['var6'].shape == (10, 10)

    zds2 = dataset.Dataset(
        [
            dataset.DelayedArray('var5', numpy.empty(10), ('z', )),
        ],
        attrs=[dataset.Attribute('attr3', 3),
               dataset.Attribute('attr4', 4)])
    zds1 = pickle.loads(pickle.dumps(template))

    zds2.merge(None)
    assert list(zds2.variables) == ['var5']

    zds1.merge(zds2)
    assert list(zds1.variables) == ['var1', 'var2', 'var3', 'var4', 'var5']
    assert zds1.dimensions == {'x': 10, 'y': 10, 'z': 10}
    assert zds1.variables['var1'].shape == (10, )
    assert zds1.variables['var2'].shape == (10, )
    assert zds1.variables['var3'].shape == (10, 10)
    assert zds1.variables['var4'].shape == (10, 10)
    assert zds1.variables['var5'].shape == (10, )

    zds2 = dataset.Dataset(
        [
            dataset.DelayedArray('var1', numpy.empty(10), ('x', )),
        ],
        attrs=[dataset.Attribute('attr1', 1),
               dataset.Attribute('attr2', 2)])
    zds1 = pickle.loads(pickle.dumps(template))
    with pytest.raises(ValueError, match=r'variable .* already exists'):
        zds1.merge(zds2)

    zds2 = dataset.Dataset(
        [
            dataset.DelayedArray('var5', numpy.empty(20), ('x', )),
        ],
        attrs=[dataset.Attribute('attr3', 3),
               dataset.Attribute('attr4', 4)])
    zds1 = pickle.loads(pickle.dumps(template))
    with pytest.raises(ValueError, match=r'dimensions .* are not identical'):
        zds1.merge(zds2)


def test_dataset_select_variables_by_dims() -> None:
    """Test selecting variables by dimensions."""
    zds = dataset.Dataset([
        dataset.DelayedArray('var1', numpy.empty(10), ('x', )),
        dataset.DelayedArray('var2', numpy.empty(10), ('y', )),
        dataset.DelayedArray('var3', numpy.empty((10, 10)), ('x', 'y')),
        dataset.DelayedArray('var4', numpy.empty(20), ('a', )),
        dataset.DelayedArray('var5', numpy.empty(20), ('b', )),
        dataset.DelayedArray('var6', numpy.empty((20, 20)), ('a', 'b')),
        dataset.DelayedArray('var7', numpy.int64(1), ()),
    ],
                          attrs=[
                              dataset.Attribute('attr1', 1),
                          ])
    selected = zds.select_variables_by_dims(('x', ))
    assert list(selected.variables) == ['var1', 'var3']
    assert selected.dimensions == {'x': 10, 'y': 10}

    selected = zds.select_variables_by_dims(('x', 'y'))
    assert list(selected.variables) == ['var1', 'var2', 'var3']
    assert selected.dimensions == {'x': 10, 'y': 10}

    selected = zds.select_variables_by_dims(('x', 'y', 'z'))
    assert list(selected.variables) == ['var1', 'var2', 'var3']
    assert selected.dimensions == {'x': 10, 'y': 10}

    selected = zds.select_variables_by_dims(('a', ))
    assert list(selected.variables) == ['var4', 'var6']
    assert selected.dimensions == {'a': 20, 'b': 20}

    selected = zds.select_variables_by_dims(('a', 'b'))
    assert list(selected.variables) == ['var4', 'var5', 'var6']
    assert selected.dimensions == {'a': 20, 'b': 20}

    selected = zds.select_variables_by_dims(('a', 'b', 'c'))
    assert list(selected.variables) == ['var4', 'var5', 'var6']
    assert selected.dimensions == {'a': 20, 'b': 20}

    selected = zds.select_variables_by_dims(('x', 'a'))
    assert list(selected.variables) == ['var1', 'var3', 'var4', 'var6']
    assert selected.dimensions == {'x': 10, 'y': 10, 'a': 20, 'b': 20}

    selected = zds.select_variables_by_dims(())
    assert list(selected.variables) == ['var7']
    assert selected.dimensions == {}

    selected = zds.select_variables_by_dims((), predicate=False)
    assert list(selected.variables) == [
        'var1', 'var2', 'var3', 'var4', 'var5', 'var6'
    ]
    assert selected.dimensions == {'x': 10, 'y': 10, 'a': 20, 'b': 20}

    selected = zds.select_variables_by_dims(('x', ), predicate=False)
    assert list(selected.variables) == ['var2', 'var4', 'var5', 'var6', 'var7']


def test_dataset_expression() -> None:
    """Test dataset expression."""
    matrix = numpy.arange(100).reshape((10, 10))
    vector = numpy.arange(10)
    zds = dataset.Dataset([
        dataset.DelayedArray('vec1', vector, ('x', )),
        dataset.DelayedArray('vec2', vector, ('y', )),
        dataset.DelayedArray('mat1', matrix, ('x', 'y')),
        dataset.DelayedArray('mat2', matrix, ('x', 'y')),
    ])
    expr = dataset.Expression('vec1 + vec2')
    assert isinstance(expr, dataset.Expression)
    assert numpy.all(expr(zds).compute() == 2 * vector)
    expr = dataset.Expression('mat1 + sin(mat2)')
    assert isinstance(expr, dataset.Expression)
    assert numpy.all(expr(zds).compute() == matrix + numpy.sin(matrix))

    with pytest.raises(NameError):
        dataset.Expression('vec1 + vec3')(zds)

    with pytest.raises(SyntaxError):
        dataset.Expression('vec1 +')(zds)


@pytest.mark.parametrize('delayed', [False, True])
def test_add_variable_holding_data(delayed) -> None:
    """A variable holding data is added with its values."""
    cls = dataset.DelayedArray if delayed else dataset.Array
    seed = dask.array.core.from_array(numpy.arange(4)) if delayed \
        else numpy.arange(4)
    zds = dataset.Dataset(variables=[cls('time', seed, ('num_lines', ))],
                          delayed=delayed)

    zds.add_variable(
        dataset.Array('measure', numpy.arange(4, dtype='float32'),
                      ('num_lines', )))

    assert numpy.array_equal(zds['measure'].values, numpy.arange(4))


def test_add_variable_rejects_extra_data_dimension() -> None:
    zds = dataset.Dataset([dataset.Array('x', numpy.arange(4), ('row', ))])
    variable = meta.VariableSchema('v', 'int64', dimensions=('row', ))

    with pytest.raises(ValueError, match='longer'):
        zds.add_variable(variable, data=numpy.ones((4, 2)))
    assert 'v' not in zds.variables
