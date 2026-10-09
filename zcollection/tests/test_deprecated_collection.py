# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Reading a collection written before the metadata change
=======================================================

``deprecated_collection`` holds the reference dataset as release 2026.6.0
wrote it: no ``version`` key, dimensions as a flat list of names, chunk
sizes in a list of their own.

Collections in that format exist on disk. These tests assert that the
current code still reads them, and that updating one makes it writable.
"""
from __future__ import annotations

import json
import pathlib
import zipfile

import numpy
import pytest

import zcollection

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
    """The frozen collection is what the old format looks like."""
    config = read_config(deprecated)

    assert 'version' not in config
    assert config['dataset']['dimensions'] == ['num_lines', 'num_pixels']
    assert 'chunks' in config['dataset']


def test_the_fixture_carries_no_immutable_variable(deprecated) -> None:
    """This collection holds no immutable variable.

    The reference dataset names its axis ``time`` and the dimension that
    axis spans ``num_lines``. Release 2026.6.0 selects the variables to
    partition by the name of the axis rather than by the name of the
    dimension, so inserting it there leaves nothing to partition.
    """
    config = read_config(deprecated)
    names = {variable['name'] for variable in config['dataset']['variables']}

    assert 'pixel' not in names
    assert names == set(reference.make_dataset(immutable=False).variables)


def test_opening_warns_and_yields_a_read_only_collection(deprecated) -> None:
    """An old collection opens, and says it has to be updated."""
    with pytest.warns(UserWarning, match='needs to be updated'):
        collection = zcollection.open_collection(deprecated)

    assert collection.mode == 'r'


def test_opening_for_writing_is_refused(deprecated) -> None:
    """Writing to an old collection is refused until it is updated."""
    with pytest.warns(UserWarning, match='needs to be updated'), \
            pytest.raises(ValueError, match='needs to be updated'):
        zcollection.open_collection(deprecated, mode='w')


def test_values_are_unchanged(deprecated) -> None:
    """The values read from the old format are the ones written."""
    with pytest.warns(UserWarning, match='needs to be updated'):
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


def written_today(path: pathlib.Path) -> dict:
    """Write the same dataset with the current code, and read its config."""
    zds = reference.make_dataset(immutable=False)
    collection = zcollection.create_collection('time', zds,
                                               reference.make_partitioning(),
                                               str(path / 'today'))
    collection.insert(zds, distributed=False)
    config = read_config(str(path / 'today'))
    del config['version']
    return config


def test_update_yields_the_configuration_written_today(deprecated,
                                                       tmp_path) -> None:
    """Updating an old collection gives the configuration of a new one.

    The size of the partitioning dimension stays unset: it is the one
    dimension whose size varies from one partition to the next.
    """
    zcollection.update_deprecated_collection(deprecated)

    config = read_config(deprecated)
    del config['version']
    assert config == written_today(tmp_path)


def test_update_makes_the_collection_writable(deprecated) -> None:
    """An updated collection opens for writing, with no warning."""
    zcollection.update_deprecated_collection(deprecated)

    collection = zcollection.open_collection(deprecated, mode='w')
    data = collection.load(delayed=False, distributed=False)
    assert data is not None
    numpy.testing.assert_array_equal(data.variables['time'].values,
                                     reference.axis())


def test_update_keeps_a_backup(deprecated) -> None:
    """Updating an old collection keeps the configuration it replaces."""
    zcollection.update_deprecated_collection(deprecated)

    backup = pathlib.Path(fs_utils.join_path(deprecated, '.zcollection.bak'))
    assert 'version' not in json.loads(backup.read_text())
