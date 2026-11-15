# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Reading a collection written before the metadata change
=======================================================

``deprecated_collection`` contains the reference dataset as release 2026.6.0
wrote it: no ``version`` key, dimensions as a list of names, and chunk sizes
in a separate list.

These tests verify that the current code reads the old format, warns when
opening it, and refuses writes. Migration to the current format is tested
in ``test_migration.py``.
"""
from __future__ import annotations

import json
import pathlib
import zipfile

import numpy
import pytest

import zcollection
import zcollection.view

from . import reference
from .. import fs_utils

#: The frozen collection.
ARCHIVE = pathlib.Path(__file__).parent / 'deprecated_collection.zip'


@pytest.fixture
def deprecated(tmp_path) -> str:
    """Extract the frozen collection, so that no test alters it."""
    target = tmp_path / 'deprecated_collection'
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(target)
    return str(target)


def read_config(path: str) -> dict:
    """Read the configuration of a collection."""
    config = pathlib.Path(fs_utils.join_path(path, '.zcollection'))
    return json.loads(config.read_text())


def test_the_fixture_is_in_the_old_format(deprecated) -> None:
    """The frozen collection uses the older format."""
    config = read_config(deprecated)

    assert 'version' not in config
    assert config['dataset']['dimensions'] == ['num_lines', 'num_pixels']
    assert 'chunks' in config['dataset']


def test_the_fixture_carries_no_immutable_variable(deprecated) -> None:
    """This collection has no immutable variables.

    The reference dataset uses ``time`` for its axis and ``num_lines``
    for the dimension it spans. Release 2026.6.0 selects variables
    to partition by the axis name instead of the dimension name.
    Because the names differ here, nothing is partitioned.
    """
    config = read_config(deprecated)
    names = {variable['name'] for variable in config['dataset']['variables']}

    assert 'pixel' not in names
    assert names == set(reference.make_dataset(immutable=False).variables)


def test_opening_warns_and_yields_a_read_only_collection(deprecated) -> None:
    """Opening an old collection warns and allows read-only access."""
    with pytest.warns(UserWarning, match='written by an earlier release'):
        collection = zcollection.open_collection(deprecated)

    assert collection.mode == 'r'


def test_opening_for_writing_is_refused(deprecated) -> None:
    """An old collection cannot be opened for writing before migration."""
    with pytest.warns(UserWarning, match='written by an earlier release'), \
            pytest.raises(ValueError, match='written by an earlier release'):
        zcollection.open_collection(deprecated, mode='w')


def test_values_are_unchanged(deprecated) -> None:
    """The values read from the old format are the ones written."""
    with pytest.warns(UserWarning, match='written by an earlier release'):
        collection = zcollection.open_collection(deprecated)

    data = collection.load(delayed=False, distributed=False)
    assert data is not None

    plain = data.variables['plain'].values
    assert not numpy.ma.getmaskarray(plain).any()
    numpy.testing.assert_array_equal(plain, reference.plain_values())

    masked = data.variables['masked'].values
    numpy.testing.assert_array_equal(
        numpy.ma.getmaskarray(masked),
        reference.masked_values() == reference.FILL_VALUE)

    numpy.testing.assert_allclose(data.variables['scaled'].values,
                                  reference.scaled_values(),
                                  atol=1 / reference.SCALE)

    scaled_masked = data.variables['scaled_masked'].values
    numpy.testing.assert_array_equal(
        numpy.ma.getmaskarray(scaled_masked),
        reference.scaled_masked_values() == reference.SCALED_FILL_VALUE)

    counts = data.variables['counts'].values
    numpy.testing.assert_array_equal(
        numpy.ma.getmaskarray(counts),
        reference.count_values() == reference.COUNT_FILL_VALUE)

    numpy.testing.assert_array_equal(data.variables['grid'].values,
                                     reference.grid_values())

    numpy.testing.assert_array_equal(data.variables['time'].values,
                                     reference.axis())


def test_a_view_opens_on_an_old_collection(deprecated, tmp_path) -> None:
    """A view reads partition dimensions from a collection in the old format."""
    with pytest.warns(UserWarning, match='written by an earlier release'):
        zview = zcollection.create_view(
            str(tmp_path / 'view'),
            zcollection.view.ViewReference(deprecated),
            distributed=False)

    data = zview.load(delayed=False, distributed=False)
    assert data is not None
    numpy.testing.assert_array_equal(data.variables['time'].values,
                                     reference.axis())
