# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Characterization of the storage semantics
=========================================

These tests pin the behaviours that a change of storage backend alters
silently: no exception is raised, the data read back is simply wrong.

They go through the public API and the files on disk, never through the
internals, so that they keep their meaning once the storage layer is
rewritten.
"""
from __future__ import annotations

import json
import pathlib

import numpy
import pytest

import zcollection

from . import reference
from .. import fs_utils
# pylint: disable=unused-import # Need to import for fixtures
from .fs import local_fs  # noqa: F401

# pylint: enable=unused-import

#: Partitions the reference dataset is expected to produce.
PARTITIONS = (
    'year=2000/month=01/day=01',
    'year=2000/month=01/day=02',
    'year=2000/month=01/day=03',
)


@pytest.fixture
def written(local_fs):  # noqa: F811
    """Write the reference collection and return it."""
    zds = reference.make_dataset()
    collection = zcollection.create_collection('time',
                                               zds,
                                               reference.make_partitioning(),
                                               str(local_fs.collection),
                                               filesystem=local_fs.fs)
    collection.insert(zds, distributed=False)
    return collection


def test_partitions(written) -> None:
    """The partitioning splits the axis into one partition per day."""
    assert tuple(sorted(written.partitions(relative=True))) == PARTITIONS


def test_plain_variable_keeps_its_zeros(written) -> None:
    """A variable without a fill value is read back unmasked.

    Its zeros are measures. A backend that gives the variable a fill value
    of zero turns every one of them into an absence.
    """
    data = written.load(delayed=False, distributed=False)
    assert data is not None
    values = data.variables['plain'].values

    assert not numpy.ma.getmaskarray(values).any()
    assert (values == 0.0).sum() == 1
    numpy.testing.assert_array_equal(values, reference.plain_values())


def test_masked_variable_masks_only_its_fill_value(written) -> None:
    """A variable with a fill value masks that value, and nothing else."""
    data = written.load(delayed=False, distributed=False)
    assert data is not None
    values = data.variables['masked'].values

    expected = reference.masked_values()
    numpy.testing.assert_array_equal(numpy.ma.getmaskarray(values),
                                     expected == reference.FILL_VALUE)
    assert not numpy.ma.getmaskarray(values)[expected == 0.0].any()


def test_filtered_variable_round_trips_within_its_precision(written) -> None:
    """A filtered variable comes back within the precision of its filter."""
    data = written.load(delayed=False, distributed=False)
    assert data is not None
    values = data.variables['scaled'].values

    numpy.testing.assert_allclose(values,
                                  reference.scaled_values(),
                                  atol=1 / reference.SCALE)


def test_datetime_axis_round_trips(written) -> None:
    """The partitioning axis keeps its resolution and its values."""
    data = written.load(delayed=False, distributed=False)
    assert data is not None
    values = data.variables['time'].values

    assert values.dtype == reference.axis().dtype
    numpy.testing.assert_array_equal(values, reference.axis())


def test_partition_filter_selects_one_day(written) -> None:
    """A filter on the partitioning keys selects without reading data."""
    selected = tuple(written.partitions(filters='day == 2', relative=True))
    assert selected == ('year=2000/month=01/day=02', )

    data = written.load(filters='day == 2', delayed=False, distributed=False)
    assert data is not None
    numpy.testing.assert_array_equal(data.variables['time'].values,
                                     reference.axis()[4:8])


def test_configuration_is_unchanged(written) -> None:
    """The collection configuration keeps the shape it is read with."""
    path = fs_utils.join_path(written.partition_properties.dir, written.CONFIG)
    with written.fs.open(path) as stream:
        written_config = json.load(stream)

    expected = json.loads((pathlib.Path(__file__).parent /
                           'reference_collection.json').read_text())
    assert written_config == expected


def test_integer_variable_masks_only_its_fill_value(written) -> None:
    """Masking on an integer variable follows the same rule as on a float."""
    data = written.load(delayed=False, distributed=False)
    assert data is not None
    values = data.variables['counts'].values

    expected = reference.count_values()
    numpy.testing.assert_array_equal(numpy.ma.getmaskarray(values),
                                     expected == reference.COUNT_FILL_VALUE)
    assert not numpy.ma.getmaskarray(values)[expected == 0].any()


def test_filtered_variable_carries_its_fill_value(written) -> None:
    """A fill value behind a filter survives the encoding.

    :mod:`zcollection.meta` requires the fill value of a filtered variable
    to be the value the filter decodes to. This asserts that the value the
    reference uses satisfies that, and comes back exactly.
    """
    data = written.load(delayed=False, distributed=False)
    assert data is not None
    values = data.variables['scaled_masked'].values

    expected = reference.scaled_masked_values()
    numpy.testing.assert_array_equal(numpy.ma.getmaskarray(values),
                                     expected == reference.SCALED_FILL_VALUE)
    numpy.testing.assert_allclose(numpy.ma.filled(values,
                                                  reference.SCALED_FILL_VALUE),
                                  expected,
                                  atol=1 / reference.SCALE)


def test_two_dimensional_variable_round_trips(written) -> None:
    """A variable spanning two dimensions keeps its shape and its values."""
    data = written.load(delayed=False, distributed=False)
    assert data is not None
    values = data.variables['grid'].values

    assert values.shape == (reference.SIZE, reference.NUM_PIXELS)
    numpy.testing.assert_array_equal(values, reference.grid_values())


def test_immutable_variable_is_stored_beside_the_partitions(written) -> None:
    """A variable that does not span the axis is stored once."""
    assert written.have_immutable

    immutable = fs_utils.join_path(written.partition_properties.dir,
                                   zcollection.collection.IMMUTABLE)
    assert written.fs.exists(immutable)
    assert written.fs.exists(fs_utils.join_path(immutable, 'pixel'))

    data = written.load(delayed=False, distributed=False)
    assert data is not None
    numpy.testing.assert_array_equal(data.variables['pixel'].values,
                                     reference.pixel_values())


def test_a_view_reads_the_collection_and_its_own_variable(
        written, local_fs) -> None:  # noqa: F811
    """A view exposes the variables of its reference, plus its own."""
    zview = zcollection.create_view(str(local_fs.view),
                                    zcollection.ViewReference(
                                        written.partition_properties.dir,
                                        local_fs.fs),
                                    filesystem=local_fs.fs,
                                    distributed=False)

    data = zview.load(delayed=False, distributed=False)
    assert data is not None
    numpy.testing.assert_array_equal(data.variables['time'].values,
                                     reference.axis())

    own = zcollection.meta.Variable(
        name='derived',
        dtype=numpy.dtype('float64'),
        dimensions=(reference.DIMENSION, ),
        fill_value=reference.FILL_VALUE,
    )
    zview.add_variable(own, distributed=False)

    reopened = zcollection.open_view(str(local_fs.view),
                                     filesystem=local_fs.fs)
    data = reopened.load(delayed=False, distributed=False)
    assert data is not None
    assert numpy.ma.getmaskarray(data.variables['derived'].values).all()
