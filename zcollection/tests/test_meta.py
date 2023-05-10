# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Metadata testing.
=================
"""
from __future__ import annotations

from typing import Any
import json
import pathlib
import pickle

import numpy
import pytest
import zarr.codecs

from .. import meta


def test_attribute() -> None:
    """Test attribute creation."""
    att = meta.Attribute('a', 23.4)
    assert isinstance(att, meta.Attribute)
    assert att.name == 'a'
    assert att.value == 23.4
    assert str(att) == "Attribute('a', 23.4)"
    # pylint: disable=comparison-with-itself
    assert att == att
    assert (att == 'X') is False
    assert att != meta.Attribute('a', '23.4')
    assert isinstance(meta.Attribute.from_config(att.get_config()),
                      meta.Attribute)

    att = meta.Attribute('a', numpy.arange(10))
    assert att == meta.Attribute('a', numpy.arange(10))

    att = meta.Attribute('a', numpy.datetime64('2000-01-01', 'us'))
    assert att == att
    # pylint: enable=comparison-with-itself


def test_dimension() -> None:
    """Test dimension creation."""
    dim = meta.Dimension('a', 12)
    assert isinstance(dim, meta.Dimension)
    assert dim.name == 'a'
    assert dim.value == 12
    assert str(dim) == "Dimension('a', 12)"
    # pylint: disable=comparison-with-itself
    assert dim == dim
    # pylint: enable=comparison-with-itself
    assert dim != meta.Dimension('a', 11)
    assert isinstance(meta.Dimension.from_config(dim.get_config()),
                      meta.Dimension)


def test_variable() -> None:
    """Test variable creation."""
    var = meta.Variable('a',
                        numpy.dtype('int16'),
                        dimensions=('a', ),
                        attrs=(meta.Attribute('x', 12), ),
                        compressor=zarr.codecs.Zlib(),
                        filters=(zarr.codecs.Delta(numpy.float64, numpy.int16),
                                 zarr.codecs.FixedScaleOffset(
                                     0, 1, numpy.int16)))
    assert isinstance(var, meta.Variable)
    assert str(var) == "Variable('a')"
    # pylint: disable=comparison-with-itself
    assert var == var
    assert (var == 2) is False
    other: meta.Variable = meta.Variable.from_config(var.get_config())
    assert var == other
    other.name = 'x'
    assert var != other
    # pylint: enable=comparison-with-itself


def test_dataset() -> None:
    """Test dataset creation."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        first: dict[str, Any] = json.load(stream)
    with root.joinpath('second_dataset.json').open(encoding='utf-8') as stream:
        second: dict[str, Any] = json.load(stream)
    ds: meta.Dataset = meta.Dataset.from_config(first)
    other: meta.Dataset = meta.Dataset.from_config(second)
    assert ds == other
    assert (ds == 2) is False
    assert (ds != other) is False
    ds.dimensions = ds.dimensions + ('dummy', )
    assert ds != other


def test_select_variables() -> None:
    """Test select_variables."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        config: dict[str, Any] = json.load(stream)
    ds: meta.Dataset = meta.Dataset.from_config(config)
    variables: set[str] = ds.select_variables(('longitude', 'latitude'))
    assert variables == {'longitude', 'latitude'}
    variables = ds.select_variables(drop_variables=('longitude', 'latitude'))
    assert set(variables) & {'longitude', 'latitude'} == set()
    variables = ds.select_variables(keep_variables=('longitude', 'latitude',
                                                    'time'),
                                    drop_variables=('time', ))
    assert variables == {'longitude', 'latitude'}


def test_search_same_dimensions_as() -> None:
    """Test search_same_dimensions_as."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        first: dict[str, Any] = json.load(stream)
    ds: meta.Dataset = meta.Dataset.from_config(first)
    other: meta.Variable = ds.search_same_dimensions_as(
        ds.variables['simulated_error_karin'])
    assert other.dimensions == ds.variables['simulated_error_karin'].dimensions

    other = meta.Variable.from_config(other.get_config())
    other.dimensions = other.dimensions + ('dummy', )
    with pytest.raises(ValueError):
        ds.search_same_dimensions_as(other)


def test_pickle() -> None:
    """Test pickling."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        data: dict[str, Any] = json.load(stream)
    ds: meta.Dataset = meta.Dataset.from_config(data)
    other: meta.Dataset = pickle.loads(pickle.dumps(ds))
    assert ds == other


def test_missing_variables() -> None:
    """Test missing_variables."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        data: dict[str, Any] = json.load(stream)
    ds: meta.Dataset = meta.Dataset.from_config(data)
    other: meta.Dataset = pickle.loads(pickle.dumps(ds))

    assert len(ds.missing_variables(other)) == 0

    del other.variables['cross_track_distance']
    del other.variables['cycle_number']

    assert set(ds.missing_variables(other)) == {
        'cross_track_distance', 'cycle_number'
    }

    other.variables['XXX'] = other.variables['longitude']
    other.variables['XXX'].name = 'XXX'
    with pytest.raises(ValueError):
        ds.missing_variables(other)


def test_add_variable() -> None:
    """Test adding a variable."""
    ds = meta.Dataset(('x', 'y'), [])
    ds.add_variable(meta.Variable('a', numpy.float64, dimensions=('x', 'y')))

    with pytest.raises(ValueError):
        ds.add_variable(
            meta.Variable('a', numpy.float64, dimensions=('x', 'y')))

    ds.add_variable(meta.Variable('b', numpy.float64, dimensions=('x', )))
    ds.add_variable(meta.Variable('c', numpy.float64, dimensions=('y', )))

    with pytest.raises(ValueError):
        ds.add_variable(
            meta.Variable('d', numpy.float64, dimensions=('a', 'y')))

    with pytest.raises(ValueError):
        ds.add_variable(
            meta.Variable('e', numpy.float64, dimensions=('a', 'b')))

    with pytest.raises(ValueError):
        ds.add_variable(meta.Variable('f', numpy.float64, dimensions=('a', )))

    ds.add_variable(meta.Variable('g', numpy.float64))


def test_select_variables_by_dims() -> None:
    """Test select_variable_by_dims."""
    ds = meta.Dataset(('a', 'b', 'x', 'y'), [])
    ds.add_variable(meta.Variable('a', numpy.float64, dimensions=('x', 'y')))
    ds.add_variable(meta.Variable('b', numpy.float64, dimensions=('x', )))
    ds.add_variable(meta.Variable('c', numpy.float64, dimensions=('y', )))
    ds.add_variable(meta.Variable('d', numpy.float64, dimensions=('a', 'y')))
    ds.add_variable(meta.Variable('e', numpy.float64, dimensions=('a', 'b')))
    ds.add_variable(meta.Variable('f', numpy.float64, dimensions=('a', )))
    ds.add_variable(meta.Variable('g', numpy.float64))

    assert ds.select_variables_by_dims(('x', 'y')) == {'a', 'b', 'c', 'd'}
    assert ds.select_variables_by_dims(('x', )) == {'a', 'b'}
    assert ds.select_variables_by_dims(('y', )) == {'a', 'c', 'd'}
    assert ds.select_variables_by_dims(('a', 'y')) == {'a', 'c', 'd', 'e', 'f'}
    assert ds.select_variables_by_dims(('a', 'b')) == {'d', 'e', 'f'}
    assert ds.select_variables_by_dims(('a', )) == {'d', 'e', 'f'}
    assert ds.select_variables_by_dims(()) == {'g'}
    assert ds.select_variables_by_dims(('z', )) == set()

    assert ds.select_variables_by_dims(('x', 'y'),
                                       predicate=False) == {'e', 'f', 'g'}
    assert ds.select_variables_by_dims(
        ('x', ), predicate=False) == {'c', 'd', 'e', 'f', 'g'}
    assert ds.select_variables_by_dims(
        ('y', ), predicate=False) == {'b', 'e', 'f', 'g'}
    assert ds.select_variables_by_dims(('a', 'y'),
                                       predicate=False) == {'b', 'g'}
    assert ds.select_variables_by_dims(
        ('a', 'b'), predicate=False) == {'a', 'b', 'c', 'g'}
    assert ds.select_variables_by_dims(
        ('a', ), predicate=False) == {'a', 'b', 'c', 'g'}
    assert ds.select_variables_by_dims(
        (), predicate=False) == {'a', 'b', 'c', 'd', 'e', 'f'}
    assert ds.select_variables_by_dims(
        ('z', ), predicate=False) == {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
