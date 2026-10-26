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

import numcodecs
import numpy
import pytest
import zarr.codecs
import zarr.codecs.numcodecs
import zarr.registry

from .. import meta


@pytest.mark.parametrize(('compressor', 'filters'), [
    (zarr.codecs.ZstdCodec(level=5, checksum=True),
     (zarr.codecs.TransposeCodec(order=(1, 0)),
      zarr.codecs.numcodecs.Delta(dtype='i8'))),
    (zarr.codecs.BloscCodec(cname='zstd', clevel=7, shuffle='bitshuffle'),
     (zarr.codecs.numcodecs.Delta(dtype='i8'), )),
    (zarr.codecs.GzipCodec(level=3),
     (numcodecs.FixedScaleOffset(0, 2, 'i8'), )),
    (numcodecs.Zlib(level=4), (numcodecs.Delta(dtype='i8'), )),
    (None, ()),
])
def test_schema_codecs_survive_json(compressor, filters) -> None:
    """JSON preserves each codec class, its parameters and filter order."""
    schema = meta.VariableSchema('values',
                                 'int64',
                                 dimensions=('row', 'pixel'),
                                 compressor=compressor,
                                 filters=filters)

    config = json.loads(json.dumps(schema.get_config()))
    restored = meta.VariableSchema.from_config(config)

    assert restored == schema
    assert type(restored.compressor) is type(compressor)
    for original, result in zip((compressor, *filters),
                                (restored.compressor, *restored.filters),
                                strict=True):
        assert type(result) is type(original)
        if original is not None:
            describe = ('to_dict' if isinstance(original, zarr.abc.codec.Codec)
                        else 'get_config')
            assert getattr(result, describe)() == getattr(original, describe)()


def test_schema_uses_the_zarr_registry_for_an_additional_codec() -> None:
    """A registered codec round-trips without a zcollection translation."""
    name = 'zcollection.tests.zstd'

    class AdditionalCodec(zarr.codecs.ZstdCodec):
        """Zstd registered under a name supplied by another package."""

        def to_dict(self):
            return {
                'name': name,
                'configuration': super().to_dict()['configuration']
            }

        @classmethod
        def from_dict(cls, data):
            return cls(**data['configuration'])

    zarr.registry.register_codec(name, AdditionalCodec)
    compressor = AdditionalCodec(level=8, checksum=True)
    schema = meta.VariableSchema('values', 'int64', compressor=compressor)

    restored = meta.VariableSchema.from_config(
        json.loads(json.dumps(schema.get_config())))

    assert isinstance(restored.compressor, AdditionalCodec)
    assert restored.compressor.to_dict() == compressor.to_dict()


def test_attribute() -> None:
    """Test attribute creation."""
    att = meta.Attribute('a', 23.4)
    assert isinstance(att, meta.Attribute)
    assert att.name == 'a'
    assert att.value == 23.4
    assert str(att) == "Attribute('a', 23.4)"

    assert att == att  # noqa: PLR0124
    assert (att == 'X') is False
    assert att != meta.Attribute('a', '23.4')
    assert isinstance(meta.Attribute.from_config(att.get_config()),
                      meta.Attribute)

    att = meta.Attribute('a', numpy.arange(10))
    assert att == meta.Attribute('a', numpy.arange(10))

    att = meta.Attribute('a', numpy.datetime64('2000-01-01', 'us'))
    assert att == att  # noqa: PLR0124


def test_dimension() -> None:
    """Test dimension creation."""
    dim = meta.Dimension(name='a', value=12)
    assert isinstance(dim, meta.Dimension)
    assert dim.name == 'a'
    assert dim.value == 12
    assert str(dim) == "Dimension('a', 12, -1)"

    assert dim == dim  # noqa: PLR0124

    assert dim != meta.Dimension('a', 11)
    assert isinstance(meta.Dimension.from_config(dim.get_config()),
                      meta.Dimension)


def test_variable() -> None:
    """Test variable creation."""
    var = meta.VariableSchema('a',
                              numpy.dtype('int16'),
                              dimensions=('a', ),
                              attrs=(meta.Attribute('x', 12), ),
                              compressor=numcodecs.Zlib(),
                              filters=(numcodecs.Delta(numpy.float64,
                                                       numpy.int16),
                                       numcodecs.FixedScaleOffset(
                                           0, 1, numpy.int16)))
    assert isinstance(var, meta.VariableSchema)
    assert str(var) == "VariableSchema('a')"

    assert var == var  # noqa: PLR0124
    assert (var == 2) is False
    other: meta.VariableSchema = meta.VariableSchema.from_config(
        var.get_config())
    assert var == other
    other.name = 'x'
    assert var != other


def test_dataset() -> None:
    """Test dataset creation."""
    root: pathlib.Path = pathlib.Path(__file__).parent

    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        first: dict[str, Any] = json.load(stream)

    with root.joinpath('second_dataset.json').open(encoding='utf-8') as stream:
        second: dict[str, Any] = json.load(stream)

    ds: meta.DatasetSchema = meta.DatasetSchema.from_config(first)
    other: meta.DatasetSchema = meta.DatasetSchema.from_config(second)
    assert ds == other
    assert (ds == 2) is False
    assert (ds != other) is False

    dim_dum = meta.Dimension(name='dummy', value=20, chunks=2)
    ds.add_dimension(dimension=dim_dum)
    assert ds != other

    with pytest.raises(TypeError, match='Dimension must be a Dimension'):
        ds.add_dimension(dimension=('dummy', 0))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match='already exists in the'):
        ds.add_dimension(dimension=dim_dum)

    assert ds.dim_chunks == {
        'num_lines': -1,
        'num_pixels': -1,
        dim_dum.name: dim_dum.chunks
    }
    assert ds.dim_size == {
        'num_lines': 0,
        'num_pixels': 0,
        dim_dum.name: dim_dum.value
    }


def test_select_variables() -> None:
    """Test select_variables."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        config: dict[str, Any] = json.load(stream)

    ds: meta.DatasetSchema = meta.DatasetSchema.from_config(config)
    variables: set[str] = ds.select_variables(('longitude', 'latitude'))
    assert variables == {'longitude', 'latitude'}

    variables = ds.select_variables(drop_variables=('longitude', 'latitude'))
    assert set(variables) & {'longitude', 'latitude'} == set()

    variables = ds.select_variables(keep_variables=('longitude', 'latitude',
                                                    'time'),
                                    drop_variables=('time', ))
    assert variables == {'longitude', 'latitude'}


def test_deprecated_config() -> None:
    """Test select_variables."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('deprecated_dataset.json').open(
            encoding='utf-8') as stream:
        config: dict[str, Any] = json.load(stream)

    ds = meta.DatasetSchema.from_deprecated_config(config)

    assert isinstance(ds, meta.DatasetSchema)

    dim_size = ds.dim_size
    dim_chunks = ds.dim_chunks

    for dim in ds.dimensions:
        assert dim_size[dim] == -1
        assert dim_chunks[dim] == -1


def test_pickle() -> None:
    """Test pickling."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        data: dict[str, Any] = json.load(stream)
    ds: meta.DatasetSchema = meta.DatasetSchema.from_config(data)
    other: meta.DatasetSchema = pickle.loads(pickle.dumps(ds))
    assert ds == other


def test_missing_variables() -> None:
    """Test missing_variables."""
    root: pathlib.Path = pathlib.Path(__file__).parent
    with root.joinpath('first_dataset.json').open(encoding='utf-8') as stream:
        data: dict[str, Any] = json.load(stream)
    ds: meta.DatasetSchema = meta.DatasetSchema.from_config(data)
    other: meta.DatasetSchema = pickle.loads(pickle.dumps(ds))

    assert len(ds.missing_variables(other)) == 0

    del other.variables['cross_track_distance']
    del other.variables['cycle_number']

    assert set(ds.missing_variables(other)) == {
        'cross_track_distance', 'cycle_number'
    }

    other.variables['XXX'] = other.variables['longitude']
    other.variables['XXX'].name = 'XXX'
    with pytest.raises(
            ValueError,
            match='The reference dataset does not define the .* variables '
            'that are defined in this dataset.'):
        ds.missing_variables(other)


def test_add_variable() -> None:
    """Test adding a variable."""
    ds = meta.DatasetSchema(dimensions=(meta.Dimension('x', 0),
                                        meta.Dimension('y', 10)),
                            variables=[])
    ds.add_variable(
        meta.VariableSchema('a', numpy.float64, dimensions=('x', 'y')))

    with pytest.raises(ValueError, match='already exists'):
        ds.add_variable(
            meta.VariableSchema(name='a',
                                dtype=numpy.float64,
                                dimensions=('x', 'y')))

    with pytest.raises(TypeError, match='Expected a VariableSchema'):
        ds.add_variable((  # type: ignore[arg-type]
            'a',
            numpy.float64,
            ('x', 'y'),
        ))

    ds.add_variable(
        meta.VariableSchema(name='b', dtype=numpy.float64, dimensions=('x', )))
    ds.add_variable(
        meta.VariableSchema(name='c', dtype=numpy.float64, dimensions=('y', )))

    with pytest.raises(
            ValueError,
            match='The new variable must use the dataset dimensions.'):
        ds.add_variable(
            meta.VariableSchema('d', numpy.float64, dimensions=('a', 'y')))

    with pytest.raises(
            ValueError,
            match='The new variable must use the dataset dimensions.'):
        ds.add_variable(
            meta.VariableSchema('e', numpy.float64, dimensions=('a', 'b')))

    with pytest.raises(
            ValueError,
            match='The new variable must use the dataset dimensions.'):
        ds.add_variable(
            meta.VariableSchema('f', numpy.float64, dimensions=('a', )))

    ds.add_variable(meta.VariableSchema('g', numpy.float64))


def test_set_for_insertion():
    var_meta = meta.VariableSchema(name='f',
                                   dtype=numpy.float64,
                                   dimensions=('a', ),
                                   attrs=(meta.Attribute('x', 12), ))
    var_insert = var_meta.set_for_insertion()

    assert not var_insert.attrs
    assert var_insert.name == var_meta.name
    assert var_insert.dtype == var_meta.dtype
    assert var_insert.dimensions == var_meta.dimensions


def test_select_variables_by_dims() -> None:
    """Test select_variable_by_dims."""
    ds = meta.DatasetSchema(dimensions=(meta.Dimension('a', 0),
                                        meta.Dimension('b', 10),
                                        meta.Dimension('x', 10),
                                        meta.Dimension('y', 10)),
                            variables=[])
    ds.add_variable(
        meta.VariableSchema('a', numpy.float64, dimensions=('x', 'y')))
    ds.add_variable(meta.VariableSchema('b', numpy.float64,
                                        dimensions=('x', )))
    ds.add_variable(meta.VariableSchema('c', numpy.float64,
                                        dimensions=('y', )))
    ds.add_variable(
        meta.VariableSchema('d', numpy.float64, dimensions=('a', 'y')))
    ds.add_variable(
        meta.VariableSchema('e', numpy.float64, dimensions=('a', 'b')))
    ds.add_variable(meta.VariableSchema('f', numpy.float64,
                                        dimensions=('a', )))
    ds.add_variable(meta.VariableSchema('g', numpy.float64))

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


@pytest.mark.parametrize(('config', 'expected'), [
    ({}, meta.DEPRECATED_FORMAT_VERSION),
    ({
        'version': '0'
    }, meta.DEPRECATED_FORMAT_VERSION),
    ({
        'version': 0
    }, meta.DEPRECATED_FORMAT_VERSION),
    ({
        'version': '2026.6.0'
    }, meta.FORMAT_VERSION),
    ({
        'version': '2026.6.1.dev3+g5943e72'
    }, meta.FORMAT_VERSION),
    ({
        'version': 1
    }, 1),
    ({
        'version': 7
    }, 7),
])
def test_format_version(config: dict[str, Any], expected: int) -> None:
    """Read the version of the layout a configuration is written in.

    Before the layout carried a version of its own, the key held the version
    of the package, which names the layout only through ``'0'``.
    """
    assert meta.format_version(config) == expected
