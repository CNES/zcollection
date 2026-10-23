# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Migration of a collection
=========================

A collection written by an earlier release is migrated by copying it into a
new one, partition by partition. These tests pin what the copy holds —
values, masks, chunking, immutable variables, the configuration of the
target — what the convenience functions report, and that the source stays as
it was.

A test that touches data starts from the frozen collection
``deprecated_collection.zip``, or from one of the variants built from it
here: a collection chunked smaller
than its shape, a collection carrying an immutable variable, a view.
"""
from __future__ import annotations

from typing import Any
import dataclasses
import hashlib
import json
import logging
import pathlib
import shutil
import warnings
import zipfile

import fsspec
import numcodecs
import numpy
import pytest
import zarr

import zcollection
import zcollection.view

from . import reference
from .. import collection, fs_utils, meta, migration, store
from ..convenience.collection import check, migrate, prune, status
from ..convenience.view import check_view, migrate_view, view_status
from ..variable.abc import FILL_VALUE
from .cluster import dask_client, dask_cluster  # noqa: F401
from .fs import local_fs, s3, s3_base, s3_fs  # noqa: F401

#: The frozen collection.
ARCHIVE = pathlib.Path(__file__).parent / 'deprecated_collection.zip'

#: Partitions the frozen collection holds, as the reports name them.
PARTITIONS = (
    'year=2000/month=01/day=01',
    'year=2000/month=01/day=02',
    'year=2000/month=01/day=03',
)

#: Chunk size given to each dimension of the multi-chunk source.
SOURCE_CHUNKS = {
    reference.DIMENSION: 3,
    reference.IMMUTABLE_DIMENSION: 2,
}

#: Metadata files of a Zarr 2 store.
ZARR_2_FILES = frozenset({'.zarray', '.zattrs', '.zgroup', '.zmetadata'})

#: Metadata file of a Zarr 3 store.
ZARR_3_FILE = 'zarr.json'

#: The local file system, for the fixtures and the tests that stay on it.
local = fs_utils.get_fs('file')

#: Variable a view holds of its own.
VIEW_VARIABLE = meta.VariableSchema(
    name='derived',
    dtype=numpy.dtype('float64'),
    dimensions=(reference.DIMENSION, ),
    fill_value=reference.FILL_VALUE,
)


@pytest.fixture
def deprecated(tmp_path) -> str:
    """Extract the frozen collection, so that no test alters it."""
    path = tmp_path / 'deprecated_collection'
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(path)
    return str(path)


@pytest.fixture
def deprecated_view(deprecated, tmp_path) -> str:
    """Write a view on the frozen collection, in the deprecated format.

    The view holds one variable of its own, filled with values read from the
    collection it references.
    """
    path = str(tmp_path / 'deprecated_view')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        zview = zcollection.create_view(path,
                                        zcollection.ViewReference(deprecated),
                                        distributed=False)
        zview.add_variable(VIEW_VARIABLE, distributed=False)
        zview.update(double_the_plain_variable,
                     VIEW_VARIABLE.name,
                     delayed=False,
                     distributed=False)

    for name in PARTITIONS:
        downgrade(fs_utils.join_path(path, name))

    local = fs_utils.get_fs('file')
    config = read_view_config(local, path)
    del config['version']
    write_view_config(local, path, config)
    return path


def double_the_plain_variable(zds, name) -> dict[str, Any]:
    """Return the values a view holds in its own variable."""
    return {name: zds.variables['plain'].values * 2.0}


def place(source: str, tested_fs) -> tuple[str, str]:
    """Copy a collection onto the filesystem under test.

    Args:
        source: The collection to copy, on the local filesystem.
        tested_fs: The filesystem under test.

    Returns:
        The path of the copy, and the path for the target of a migration.
    """
    target = f'{tested_fs.collection}-source'
    fs_utils.copy_tree(source, target, fs_utils.get_fs('file'), tested_fs.fs)
    return target, f'{tested_fs.collection}-target'


def digest(fs: fsspec.AbstractFileSystem, path: str) -> dict[str, str]:
    """Return the digest of every file of a directory.

    Args:
        fs: The filesystem the directory is held on.
        path: The directory to read.

    Returns:
        The checksum of each file, by path relative to the directory.
    """
    checksums: dict[str, str] = {}
    for root, _, files in fs_utils.fs_walk(fs, path):
        for name in files:
            item = fs_utils.join_path(root, name)
            checksums[item[len(path):]] = hashlib.sha256(
                fs.cat_file(item)).hexdigest()
    return checksums


def read_config(fs: fsspec.AbstractFileSystem, path: str) -> dict[str, Any]:
    """Read the configuration of a collection."""
    with fs.open(fs_utils.join_path(path,
                                    zcollection.Collection.CONFIG)) as stream:
        return json.load(stream)


def write_config(fs: fsspec.AbstractFileSystem, path: str,
                 config: dict[str, Any]) -> None:
    """Replace the configuration of a collection."""
    name = fs_utils.join_path(path, zcollection.Collection.CONFIG)
    with fs.open(name, 'w') as stream:
        json.dump(config, stream)
    fs.invalidate_cache(name)


def read_view_config(fs: fsspec.AbstractFileSystem,
                     path: str) -> dict[str, Any]:
    """Read the configuration of a view."""
    with fs.open(fs_utils.join_path(path, zcollection.View.CONFIG)) as stream:
        return json.load(stream)


def write_view_config(fs: fsspec.AbstractFileSystem, path: str,
                      config: dict[str, Any]) -> None:
    """Replace the configuration of a view."""
    name = fs_utils.join_path(path, zcollection.View.CONFIG)
    with fs.open(name, 'w') as stream:
        json.dump(config, stream)
    fs.invalidate_cache(name)


def point_view_at(fs: fsspec.AbstractFileSystem, path: str,
                  reference_path: str) -> None:
    """Write the paths a copied view configuration holds.

    Args:
        fs: The filesystem the view is held on.
        path: The directory of the view.
        reference_path: The collection the view references.
    """
    config = read_view_config(fs, path)
    config['base_dir'] = fs_utils.normalize_path(fs, path)
    config['view_ref'] = {
        'path': fs_utils.normalize_path(fs, reference_path),
        'fs': json.loads(fs.to_json()),
    }
    write_view_config(fs, path, config)


def checksum_of(fs: fsspec.AbstractFileSystem, partition: str) -> str:
    """Read the checksum a view holds for one of its partitions."""
    array = store.open_array(fs,
                             fs_utils.join_path(partition, '.checksum'),
                             mode='r')
    return array.attrs['checksum']


def unconsolidate(fs: fsspec.AbstractFileSystem, partition: str) -> None:
    """Drop the consolidated metadata of a Zarr 3 group."""
    name = fs_utils.join_path(partition, ZARR_3_FILE)
    with fs.open(name) as stream:
        metadata: dict[str, Any] = json.load(stream)
    del metadata['consolidated_metadata']
    with fs.open(name, 'w') as stream:
        json.dump(metadata, stream)
    fs.invalidate_cache(partition)


def rewrite_array(path: pathlib.Path, chunks: dict[str, int]) -> None:
    """Write a Zarr 2 array again, with the given chunking.

    The codecs, the fill value, the attributes and the values are the ones
    the array holds.

    Args:
        path: The directory of the array, on the local filesystem.
        chunks: The chunk size of each dimension.
    """
    metadata: dict[str, Any] = json.loads((path / '.zarray').read_text())
    attrs: dict[str, Any] = json.loads((path / '.zattrs').read_text())
    shape = tuple(metadata['shape'])
    values = zarr.open_array(store=str(path), mode='r')[...]
    compressor = metadata['compressor']

    shutil.rmtree(path)
    array: zarr.Array = zarr.create_array(
        store=str(path),
        shape=shape,
        chunks=tuple(
            min(chunks[name], size) for name, size in zip(
                attrs['_ARRAY_DIMENSIONS'], shape, strict=True)),
        dtype=metadata['dtype'],
        zarr_format=2,
        compressors=numcodecs.get_codec(compressor) if compressor else None,
        filters=[
            numcodecs.get_codec(item) for item in metadata['filters'] or []
        ],
        fill_value=metadata['fill_value'],
        attributes=attrs)
    array[...] = values


def rechunk(partition: str,
            chunks: dict[str, int],
            names: tuple[str, ...] | None = None) -> None:
    """Write the arrays of a Zarr 2 group again, with the given chunking.

    Args:
        partition: The directory of the group, on the local filesystem.
        chunks: The chunk size of each dimension.
        names: The arrays to write again. All of them when it is None.
    """
    directory = pathlib.Path(partition)
    for item in sorted(directory.iterdir()):
        if item.is_dir() and (names is None or item.name in names):
            rewrite_array(item, chunks)
    zarr.consolidate_metadata(str(directory), zarr_format=2)


def downgrade(partition: str) -> None:
    """Write the arrays of a view partition in the Zarr 2 format.

    Args:
        partition: The directory of the partition, on the local filesystem.
    """
    for path in sorted(item for item in pathlib.Path(partition).iterdir()
                       if item.is_dir()):
        array: zarr.Array = zarr.open_array(store=str(path), mode='r')
        attrs = dict(array.attrs)
        attrs['_ARRAY_DIMENSIONS'] = list(array.metadata.dimension_names or ())
        shape, dtype = array.shape, array.dtype
        fill_value, values = array.fill_value, array[...]

        shutil.rmtree(path)
        downgraded: zarr.Array = zarr.create_array(store=str(path),
                                                   shape=shape,
                                                   chunks=shape,
                                                   dtype=dtype,
                                                   zarr_format=2,
                                                   compressors=None,
                                                   filters=[],
                                                   fill_value=fill_value,
                                                   attributes=attrs)
        downgraded[...] = values


def add_immutable(path: str) -> None:
    """Write an immutable variable beside the partitions of a collection.

    The frozen collection carries none.

    Args:
        path: The directory of the collection, on the local filesystem.
    """
    values: numpy.ndarray = reference.pixel_values()
    immutable = fs_utils.join_path(path, collection.IMMUTABLE)
    array: zarr.Array = zarr.create_array(
        store=immutable,
        name='pixel',
        shape=values.shape,
        chunks=values.shape,
        dtype=values.dtype,
        zarr_format=2,
        compressors=None,
        filters=[],
        fill_value=None,
        attributes={'_ARRAY_DIMENSIONS': [reference.IMMUTABLE_DIMENSION]})
    array[...] = values
    zarr.consolidate_metadata(immutable, zarr_format=2)

    local = fs_utils.get_fs('file')
    config = read_config(local, path)
    config['dataset']['variables'].append({
        'attrs': [['units', '1']],
        'compressor':
        None,
        'dimensions': [reference.IMMUTABLE_DIMENSION],
        'dtype':
        values.dtype.str,
        'fill_value':
        None,
        'filters': [],
        'name':
        'pixel',
    })
    write_config(local, path, config)


def read_collection(path: str,
                    fs: fsspec.AbstractFileSystem) -> zcollection.Dataset:
    """Read every variable of a collection."""
    zcol = zcollection.open_collection(path, filesystem=fs)
    zds = zcol.load(delayed=False, distributed=False)
    assert zds is not None
    return zds


def assert_datasets_equal(expected, actual) -> None:
    """Assert two datasets hold the same variables, values and attributes.

    Args:
        expected: The dataset read from the source.
        actual: The dataset read from the target.
    """
    assert set(actual.variables) == set(expected.variables)
    assert attributes_of(actual.attrs) == attributes_of(expected.attrs)

    for name, variable in expected.variables.items():
        values = actual.variables[name].values
        assert values.dtype == variable.values.dtype
        assert attributes_of(actual.variables[name].attrs) == attributes_of(
            variable.attrs)

        mask = numpy.ma.getmaskarray(variable.values)
        numpy.testing.assert_array_equal(numpy.ma.getmaskarray(values), mask)
        numpy.testing.assert_array_equal(
            numpy.ma.getdata(values)[~mask],
            numpy.ma.getdata(variable.values)[~mask])


def attributes_of(attrs) -> dict[str, Any]:
    """Return the attributes of a dataset or of a variable, by name."""
    return dict(item.get_config() for item in attrs)


def test_the_package_exports_the_entry_points() -> None:
    """The convenience functions, ``Outcome`` and ``Status`` are attributes
    of the package."""
    assert zcollection.migrate is migrate
    assert zcollection.status is status
    assert zcollection.check is check
    assert zcollection.prune is prune
    assert zcollection.migrate_view is migrate_view
    assert zcollection.Outcome is migration.Outcome
    assert zcollection.Status is migration.Status


def test_the_reports_hold_the_fields_of_the_contract() -> None:
    """An outcome holds three tuples, a status three others."""
    assert tuple(
        item.name
        for item in dataclasses.fields(migration.Outcome)) == ('done',
                                                               'skipped',
                                                               'failed')
    assert tuple(
        item.name
        for item in dataclasses.fields(migration.Status)) == ('migrated',
                                                              'missing',
                                                              'pruned', 'lost')


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_the_values_of_every_variable_survive_the_copy(fs, request,
                                                       deprecated) -> None:
    """The target holds the values, the masks and the attributes of the
    source."""
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)

    outcome = migrate(source,
                      target,
                      source_filesystem=tested_fs.fs,
                      target_filesystem=tested_fs.fs,
                      distributed=False)

    assert outcome.done == PARTITIONS
    assert outcome.skipped == ()
    assert outcome.failed == ()

    with pytest.warns(UserWarning, match='written by an earlier release'):
        expected = read_collection(source, tested_fs.fs)
    assert_datasets_equal(expected, read_collection(target, tested_fs.fs))


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_the_target_is_a_writable_zarr_3_collection(fs, request,
                                                    deprecated) -> None:
    """The target holds Zarr 3 stores, and takes a variable written to it."""
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)

    migrate(source,
            target,
            source_filesystem=tested_fs.fs,
            target_filesystem=tested_fs.fs,
            distributed=False)

    names = {
        name
        for _, _, files in fs_utils.fs_walk(tested_fs.fs, target)
        for name in files
    }
    assert ZARR_3_FILE in names
    assert names & ZARR_2_FILES == set()

    with warnings.catch_warnings():
        warnings.filterwarnings('error',
                                message='.*written by an earlier release')
        zcol = zcollection.open_collection(target,
                                           mode='w',
                                           filesystem=tested_fs.fs)

    zcol.add_variable(meta.VariableSchema(name='added',
                                          dtype=numpy.dtype('float64'),
                                          dimensions=(reference.DIMENSION, ),
                                          fill_value=reference.FILL_VALUE),
                      distributed=False)
    zds = zcol.load(delayed=False, distributed=False)
    assert zds is not None
    assert numpy.ma.getmaskarray(zds.variables['added'].values).all()


def test_the_target_is_configured_as_a_collection_written_today(
        deprecated, tmp_path) -> None:
    """The copy carries the configuration a fresh write produces.

    The size of the partitioning dimension stays unset: it is the one
    dimension whose size varies from one partition to the next.
    """
    target = str(tmp_path / 'target')
    today = str(tmp_path / 'today')
    zds = reference.make_dataset(immutable=False)
    written = zcollection.create_collection('time', zds,
                                            reference.make_partitioning(),
                                            today)
    written.insert(zds, distributed=False)

    migrate(deprecated, target, distributed=False)

    assert read_config(local, target) == read_config(local, today)


def test_a_target_configured_otherwise_is_refused(deprecated,
                                                  tmp_path) -> None:
    """A target whose configuration disagrees with the source stops the run."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    local = fs_utils.get_fs('file')
    config = read_config(local, target)
    config['dataset']['dimensions'] = [[
        name, -1 if name == reference.DIMENSION else 5, chunks
    ] for name, _, chunks in config['dataset']['dimensions']]
    write_config(local, target, config)

    # The contract names the type of the error.
    with pytest.raises(ValueError):  # noqa: PT011
        migrate(deprecated, target, distributed=False)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_a_filter_migrates_only_the_partitions_it_selects(
        fs, request, deprecated) -> None:
    """A filter selects the partitions written, and the status names the
    rest missing."""
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)

    outcome = migrate(source,
                      target,
                      source_filesystem=tested_fs.fs,
                      target_filesystem=tested_fs.fs,
                      filters='day == 2',
                      distributed=False)

    assert outcome.done == (PARTITIONS[1], )
    assert not tested_fs.fs.exists(fs_utils.join_path(target, PARTITIONS[0]))

    report = status(source,
                    target,
                    source_filesystem=tested_fs.fs,
                    target_filesystem=tested_fs.fs)
    assert report.migrated == (PARTITIONS[1], )
    assert report.missing == (PARTITIONS[0], PARTITIONS[2])
    assert report.pruned == ()


def test_selected_partitions_migrates_only_the_partitions_it_names(
        deprecated, tmp_path) -> None:
    """Only the partitions the list names are written."""
    target = str(tmp_path / 'target')

    outcome = migrate(deprecated,
                      target,
                      selected_partitions=[PARTITIONS[0], PARTITIONS[2]],
                      distributed=False)

    assert outcome.done == (PARTITIONS[0], PARTITIONS[2])
    assert status(deprecated, target).missing == (PARTITIONS[1], )


def test_a_status_on_an_empty_target_reports_every_partition_missing(
        deprecated, tmp_path) -> None:
    """Before the first migration, the source holds everything missing."""
    report = status(deprecated, str(tmp_path / 'target'))

    assert report.migrated == ()
    assert report.missing == PARTITIONS
    assert report.pruned == ()


def test_a_status_prints_the_counts_of_each_group(deprecated,
                                                  tmp_path) -> None:
    """A status reads as one line per value of the first partitioning key."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, filters='day == 2', distributed=False)

    assert str(status(deprecated, target)).splitlines() == [
        'year=2000 : 1 migrated, 2 missing',
    ]


def test_a_status_counts_as_pruned_what_the_source_no_longer_holds(
        deprecated, tmp_path) -> None:
    """A partition the target alone holds is counted apart."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)
    prune(deprecated, target, filters='day == 1', distributed=False)

    report = status(deprecated, target)
    assert report.migrated == (PARTITIONS[1], PARTITIONS[2])
    assert report.missing == ()
    assert report.pruned == (PARTITIONS[0], )
    assert '1 pruned' in str(report)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_a_second_migration_changes_nothing(fs, request, deprecated) -> None:
    """A complete partition is skipped and left as it was written."""
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)

    migrate(source,
            target,
            source_filesystem=tested_fs.fs,
            target_filesystem=tested_fs.fs,
            distributed=False)
    written = digest(tested_fs.fs, target)

    outcome = migrate(source,
                      target,
                      source_filesystem=tested_fs.fs,
                      target_filesystem=tested_fs.fs,
                      distributed=False)

    assert outcome.done == ()
    assert outcome.skipped == PARTITIONS
    assert outcome.failed == ()
    assert digest(tested_fs.fs, target) == written


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_an_unconsolidated_partition_is_written_again(fs, request,
                                                      deprecated) -> None:
    """An interrupted partition is written again, and equals an
    uninterrupted copy.

    The interruption leaves the partition holding every variable but one,
    and its group metadata unconsolidated.
    """
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)
    uninterrupted = f'{tested_fs.collection}-uninterrupted'

    for item in (uninterrupted, target):
        migrate(source,
                item,
                source_filesystem=tested_fs.fs,
                target_filesystem=tested_fs.fs,
                distributed=False)

    interrupted = fs_utils.join_path(target, PARTITIONS[1])
    tested_fs.fs.rm(fs_utils.join_path(interrupted, 'plain'), recursive=True)
    unconsolidate(tested_fs.fs, interrupted)

    outcome = migrate(source,
                      target,
                      source_filesystem=tested_fs.fs,
                      target_filesystem=tested_fs.fs,
                      distributed=False)

    assert outcome.done == (PARTITIONS[1], )
    assert outcome.skipped == (PARTITIONS[0], PARTITIONS[2])
    assert digest(tested_fs.fs, target) == digest(tested_fs.fs, uninterrupted)


@pytest.mark.parametrize('distributed', [False, True])
def test_a_partition_that_raises_does_not_stop_the_run(
        dask_client,  # noqa: F811
        deprecated,
        tmp_path,
        distributed) -> None:
    """A partition the copy stops on is recorded with its error as text,
    and the others are written.

    The partition that raises holds no consolidated metadata, and reading
    a Zarr 2 group requires it.
    """
    target = str(tmp_path / 'target')
    pathlib.Path(deprecated, PARTITIONS[1], '.zmetadata').unlink()

    outcome = migrate(deprecated, target, distributed=distributed)

    assert outcome.done == (PARTITIONS[0], PARTITIONS[2])
    assert len(outcome.failed) == 1
    partition, error = outcome.failed[0]
    assert partition == PARTITIONS[1]
    assert isinstance(error, str)
    assert error != ''
    assert status(deprecated, target).missing == (PARTITIONS[1], )


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_prune_removes_a_migrated_partition_and_refuses_the_others(
        fs, request, deprecated) -> None:
    """The source loses the partitions the target holds complete.

    The files of the collection itself stay.
    """
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)
    migrate(source,
            target,
            source_filesystem=tested_fs.fs,
            target_filesystem=tested_fs.fs,
            filters='day == 2',
            distributed=False)

    outcome = prune(source,
                    target,
                    source_filesystem=tested_fs.fs,
                    target_filesystem=tested_fs.fs,
                    distributed=False)

    assert outcome.done == (PARTITIONS[1], )
    assert tuple(item for item, _ in outcome.failed) == (PARTITIONS[0],
                                                         PARTITIONS[2])
    assert not tested_fs.fs.exists(fs_utils.join_path(source, PARTITIONS[1]))
    assert tested_fs.fs.exists(fs_utils.join_path(source, PARTITIONS[0]))
    assert tested_fs.fs.exists(
        fs_utils.join_path(source, zcollection.Collection.CONFIG))


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_check_reads_every_value_as_equal(fs, request, deprecated) -> None:
    """A copy read back holds the values of the source."""
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)
    migrate(source,
            target,
            source_filesystem=tested_fs.fs,
            target_filesystem=tested_fs.fs,
            distributed=False)

    outcome = check(source,
                    target,
                    source_filesystem=tested_fs.fs,
                    target_filesystem=tested_fs.fs,
                    distributed=False)

    assert outcome.done == PARTITIONS
    assert outcome.skipped == ()
    assert outcome.failed == ()


def test_check_names_the_variables_the_target_changed(deprecated,
                                                      tmp_path) -> None:
    """A partition whose values were changed is reported with the variables
    that differ."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    local = fs_utils.get_fs('file')
    for name in ('masked', 'grid'):
        array = store.open_array(
            local, fs_utils.join_path(target, PARTITIONS[2], name))
        array[...] = numpy.ones(array.shape, dtype=array.dtype)

    outcome = check(deprecated, target, distributed=False)

    assert outcome.done == (PARTITIONS[0], PARTITIONS[1])
    assert len(outcome.failed) == 1
    partition, names = outcome.failed[0]
    assert partition == PARTITIONS[2]
    assert 'masked' in names
    assert 'grid' in names


def test_check_skips_a_partition_absent_from_the_target(deprecated,
                                                        tmp_path) -> None:
    """A partition the target does not hold is left uncompared."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, filters='day == 2', distributed=False)

    outcome = check(deprecated, target, distributed=False)

    assert outcome.done == (PARTITIONS[1], )
    assert outcome.skipped == (PARTITIONS[0], PARTITIONS[2])
    assert outcome.failed == ()


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_the_immutable_variables_are_copied(fs, request, deprecated) -> None:
    """The immutable group is written, and read from the target."""
    add_immutable(deprecated)
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)

    outcome = migrate(source,
                      target,
                      source_filesystem=tested_fs.fs,
                      target_filesystem=tested_fs.fs,
                      distributed=False)

    assert outcome.done == (collection.IMMUTABLE, *PARTITIONS)
    assert tested_fs.fs.exists(
        fs_utils.join_path(target, collection.IMMUTABLE, 'pixel', ZARR_3_FILE))

    zds = read_collection(target, tested_fs.fs)
    numpy.testing.assert_array_equal(zds.variables['pixel'].values,
                                     reference.pixel_values())

    again = migrate(source,
                    target,
                    source_filesystem=tested_fs.fs,
                    target_filesystem=tested_fs.fs,
                    distributed=False)
    assert again.skipped == (collection.IMMUTABLE, *PARTITIONS)
    assert again.done == ()


def test_the_chunking_of_the_source_is_preserved(deprecated, tmp_path) -> None:
    """A source chunked smaller than its shape keeps its chunking in the
    target."""
    for name in PARTITIONS:
        rechunk(fs_utils.join_path(deprecated, name), SOURCE_CHUNKS)
    target = str(tmp_path / 'target')

    migrate(deprecated, target, distributed=False)

    local = fs_utils.get_fs('file')
    partition = fs_utils.join_path(target, PARTITIONS[0])
    assert store.open_array(
        local, fs_utils.join_path(partition, 'grid'),
        mode='r').chunks == (SOURCE_CHUNKS[reference.DIMENSION],
                             SOURCE_CHUNKS[reference.IMMUTABLE_DIMENSION])
    assert store.open_array(
        local, fs_utils.join_path(partition, 'time'),
        mode='r').chunks == (SOURCE_CHUNKS[reference.DIMENSION], )


def test_variables_that_disagree_on_a_chunk_size_are_reported_failed(
        deprecated, tmp_path) -> None:
    """A partition holding two chunk sizes for one dimension is not
    written."""
    for name in PARTITIONS:
        rechunk(fs_utils.join_path(deprecated, name), SOURCE_CHUNKS)
    rechunk(fs_utils.join_path(deprecated, PARTITIONS[1]), {
        reference.DIMENSION: 2,
        reference.IMMUTABLE_DIMENSION: 2,
    },
            names=('grid', ))
    target = str(tmp_path / 'target')

    outcome = migrate(deprecated, target, distributed=False)

    assert outcome.done == (PARTITIONS[0], PARTITIONS[2])
    assert tuple(item for item, _ in outcome.failed) == (PARTITIONS[1], )


def test_the_source_is_unchanged(deprecated, tmp_path) -> None:
    """Migrating, checking and reading a status write nothing to the source."""
    local = fs_utils.get_fs('file')
    target = str(tmp_path / 'target')
    before = digest(local, deprecated)

    migrate(deprecated, target, distributed=False)
    assert digest(local, deprecated) == before

    check(deprecated, target, distributed=False)
    assert digest(local, deprecated) == before

    status(deprecated, target)
    assert digest(local, deprecated) == before


def test_a_migration_reads_and_writes_on_two_filesystems(deprecated) -> None:
    """Each side of a migration is read through the filesystem given for it."""
    target = '/migrated-collection'
    memory: fsspec.AbstractFileSystem = fsspec.filesystem('memory')
    try:
        outcome = migrate(deprecated,
                          target,
                          source_filesystem='file',
                          target_filesystem='memory',
                          distributed=False)

        assert outcome.done == PARTITIONS
        with pytest.warns(UserWarning, match='written by an earlier release'):
            expected = read_collection(deprecated, fs_utils.get_fs('file'))
        assert_datasets_equal(expected, read_collection(target, memory))
    finally:
        if memory.exists(target):
            memory.rm(target, recursive=True)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_a_view_migrates_and_keeps_its_checksum(fs, request, deprecated,
                                                deprecated_view) -> None:
    """The migrated view points at the reference and holds its checksum."""
    tested_fs = request.getfixturevalue(fs)
    source, target = place(deprecated, tested_fs)
    view_source = f'{tested_fs.view}-source'
    view_target = f'{tested_fs.view}-target'
    fs_utils.copy_tree(deprecated_view, view_source, fs_utils.get_fs('file'),
                       tested_fs.fs)
    point_view_at(tested_fs.fs, view_source, source)

    migrate(source,
            target,
            source_filesystem=tested_fs.fs,
            target_filesystem=tested_fs.fs,
            distributed=False)
    outcome = migrate_view(view_source,
                           view_target,
                           reference=target,
                           source_filesystem=tested_fs.fs,
                           target_filesystem=tested_fs.fs,
                           distributed=False)

    assert outcome.done == PARTITIONS

    config = read_view_config(tested_fs.fs, view_target)
    assert config['view_ref']['path'] == fs_utils.normalize_path(
        tested_fs.fs, target)
    assert config['base_dir'] == fs_utils.normalize_path(
        tested_fs.fs, view_target)

    for name in PARTITIONS:
        assert checksum_of(tested_fs.fs, fs_utils.join_path(
            view_target,
            name)) == checksum_of(tested_fs.fs,
                                  fs_utils.join_path(view_source, name))

    zview = zcollection.open_view(view_target, filesystem=tested_fs.fs)
    assert zview.is_synced(distributed=False)
    zds = zview.load(delayed=False, distributed=False)
    assert zds is not None
    numpy.testing.assert_array_equal(zds.variables[VIEW_VARIABLE.name].values,
                                     reference.plain_values() * 2.0)


def test_a_view_migrates_keeping_the_path_it_references(
        deprecated, deprecated_view, tmp_path) -> None:
    """A migration without a reference keeps the path the source holds."""
    target = str(tmp_path / 'view_target')

    migrate_view(deprecated_view, target, distributed=False)

    local = fs_utils.get_fs('file')
    config = read_view_config(local, target)
    assert config['view_ref']['path'] == fs_utils.normalize_path(
        local, deprecated)


def test_check_compares_the_immutable_variables(deprecated, tmp_path) -> None:
    """The immutable group is compared, and its differences are reported.

    Nothing else in a migration reads the copy of an immutable variable: it
    is written once, and the partitions hold nothing of it.
    """
    add_immutable(deprecated)
    local = fs_utils.get_fs('file')
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    outcome = check(deprecated, target, distributed=False)
    assert outcome.done == (collection.IMMUTABLE, *PARTITIONS)
    assert not outcome.failed

    array = store.open_array(
        local, fs_utils.join_path(target, collection.IMMUTABLE, 'pixel'))
    array[...] = reference.pixel_values() + 1

    outcome = check(deprecated, target, distributed=False)
    assert outcome.failed == ((collection.IMMUTABLE, 'pixel'), )
    assert outcome.done == PARTITIONS


def test_check_names_the_arrays_the_source_holds_no_chunk_of(
        deprecated, tmp_path, caplog) -> None:
    """An array the source holds no chunk of is named in a warning.

    Zarr reads it as the fill value on both sides, so comparing the values
    cannot see it.
    """
    local = fs_utils.get_fs('file')
    target = str(tmp_path / 'target')

    # The chunk alone, the metadata of the array staying.
    local.rm(fs_utils.join_path(deprecated, PARTITIONS[0], 'masked', '0'))
    # The whole array.
    local.rm(fs_utils.join_path(deprecated, PARTITIONS[1], 'grid'),
             recursive=True)

    migrate(deprecated, target, distributed=False)
    with caplog.at_level(logging.WARNING, logger='zcollection.migration'):
        outcome = check(deprecated, target, distributed=False)

    assert outcome.done == PARTITIONS
    assert not outcome.failed
    named: str = ' '.join(item.getMessage() for item in caplog.records)
    assert 'masked' in named
    assert 'grid' in named


def test_the_deprecated_source_emits_no_warning(deprecated, tmp_path) -> None:
    """The four functions read a collection written by an earlier release
    without warning."""
    target = str(tmp_path / 'target')

    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        migrate(deprecated, target, distributed=False)
        status(deprecated, target)
        check(deprecated, target, distributed=False)
        prune(deprecated, target, distributed=False)


def fill_the_masked_variable(partition: str) -> None:
    """Write nothing but its fill value into the ``masked`` array.

    The reference collection holds no such array, and one is needed to see
    a chunk the copy leaves out.

    Args:
        partition: The directory of the partition.
    """
    array = store.open_array(local, fs_utils.join_path(partition, 'masked'))
    array[...] = reference.FILL_VALUE


def test_a_chunk_sits_beside_the_metadata_of_its_array(deprecated,
                                                       tmp_path) -> None:
    """An array names a chunk with the ``c`` prefix and a dot."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    held = sorted(
        item.name
        for item in pathlib.Path(target, PARTITIONS[0], 'grid').iterdir())
    assert held == ['c.0.0', ZARR_3_FILE]


def test_a_chunk_holding_nothing_but_the_fill_value_is_not_written(
        deprecated, tmp_path) -> None:
    """Its array keeps its metadata, and reads back its values and masks."""
    for name in PARTITIONS:
        fill_the_masked_variable(fs_utils.join_path(deprecated, name))
    source = read_collection(deprecated, local)
    target = str(tmp_path / 'target')

    migrate(deprecated, target, distributed=False)

    written = pathlib.Path(target, PARTITIONS[0], 'masked')
    assert sorted(item.name for item in written.iterdir()) == [ZARR_3_FILE]

    values = read_collection(target, local).variables['masked'].values
    assert numpy.ma.getmaskarray(values).all()
    numpy.testing.assert_array_equal(
        numpy.ma.getdata(values),
        numpy.ma.getdata(source.variables['masked'].values))


def test_a_status_counts_as_lost_what_neither_collection_holds(
        deprecated, tmp_path) -> None:
    """A partition pruned and then damaged is named, not passed over."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)
    prune(deprecated, target, filters='day == 1', distributed=False)
    unconsolidate(local, fs_utils.join_path(target, PARTITIONS[0]))

    report = status(deprecated, target)

    assert report.lost == (PARTITIONS[0], )
    assert report.pruned == ()
    assert '1 lost' in str(report)


def test_a_second_view_migration_holds_what_it_wrote(deprecated_view,
                                                     tmp_path) -> None:
    """A partition the target holds in full is not written again."""
    target = str(tmp_path / 'view-target')
    first = migrate_view(deprecated_view, target, distributed=False)
    assert first.done == PARTITIONS

    second = migrate_view(deprecated_view, target, distributed=False)

    assert second.done == ()
    assert second.skipped == PARTITIONS
    assert check_view(deprecated_view, target,
                      distributed=False).done == PARTITIONS


def test_prune_refuses_a_target_that_is_not_the_copy(deprecated,
                                                     tmp_path) -> None:
    """Another collection does not stand for the copy of this one."""
    other = str(tmp_path / 'other')
    zds = reference.make_dataset(immutable=False)
    zds = zds.select_vars(names=['time', 'plain'])
    zcollection.create_collection('time', zds, reference.make_partitioning(),
                                  other).insert(zds, distributed=False)

    with pytest.raises(ValueError, match='is not the copy of'):
        prune(deprecated, other, distributed=False)

    assert local.exists(fs_utils.join_path(deprecated, PARTITIONS[0]))


def test_prune_removes_what_both_the_filter_and_the_list_name(
        deprecated, tmp_path) -> None:
    """The two ways of selecting bound each other, as they do elsewhere."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    outcome = prune(deprecated,
                    target,
                    filters='day == 2',
                    selected_partitions=PARTITIONS,
                    distributed=False)

    assert outcome.done == (PARTITIONS[1], )
    assert local.exists(fs_utils.join_path(deprecated, PARTITIONS[0]))
    assert local.exists(fs_utils.join_path(deprecated, PARTITIONS[2]))


def test_a_view_status_names_what_the_source_view_no_longer_holds(
        deprecated_view, tmp_path) -> None:
    """A partition the copy holds alone is named, not passed over."""
    target = str(tmp_path / 'view-target')
    migrate_view(deprecated_view, target, distributed=False)
    local.rm(fs_utils.join_path(deprecated_view, PARTITIONS[0]),
             recursive=True)

    report = view_status(deprecated_view, target)

    assert report.migrated == PARTITIONS[1:]
    assert report.pruned == (PARTITIONS[0], )
    assert report.missing == ()
    assert report.lost == ()


def test_check_names_a_variable_whose_attributes_changed(deprecated,
                                                         tmp_path) -> None:
    """The copy carries the attributes over, so the check reads them."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    array = store.open_array(
        local, fs_utils.join_path(target, PARTITIONS[0], 'counts'))
    array.attrs['units'] = 'furlongs'
    store.consolidate(local, fs_utils.join_path(target, PARTITIONS[0]))

    outcome = check(deprecated, target, distributed=False)

    assert outcome.failed == ((PARTITIONS[0], 'counts'), )
    assert outcome.done == PARTITIONS[1:]


def test_check_names_a_variable_whose_chunking_changed(deprecated,
                                                       tmp_path) -> None:
    """The copy keeps the chunking of the source, so the check reads it."""
    for name in PARTITIONS:
        rechunk(fs_utils.join_path(deprecated, name), SOURCE_CHUNKS)
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    partition = fs_utils.join_path(target, PARTITIONS[0])
    held = store.open_array(local, fs_utils.join_path(partition, 'plain'))
    again = store.create_array(local,
                               partition,
                               name='plain',
                               shape=held.shape,
                               chunks=(1, ),
                               dtype=held.dtype,
                               fill_value=held.fill_value,
                               dimensions=('num_lines', ),
                               attributes=dict(held.attrs),
                               overwrite=True)
    again[...] = held[...]
    store.consolidate(local, partition)

    outcome = check(deprecated, target, distributed=False)

    assert outcome.failed == ((PARTITIONS[0], 'plain'), )


def test_check_names_an_array_short_of_one_chunk_of_several(
        deprecated, tmp_path, caplog) -> None:
    """The count of the chunks held is read against the grid of the array."""
    partition = fs_utils.join_path(deprecated, PARTITIONS[0])
    rechunk(partition, SOURCE_CHUNKS)
    held = sorted(pathlib.Path(partition, 'plain').glob('[0-9]*'))
    assert len(held) > 1
    held[-1].unlink()

    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)
    with caplog.at_level(logging.WARNING, logger='zcollection.migration'):
        check(deprecated, target, distributed=False)

    named = ' '.join(item.getMessage() for item in caplog.records)
    assert 'plain' in named


def test_check_view_names_the_variables_the_target_changed(
        deprecated_view, tmp_path) -> None:
    """A partition whose values were changed is reported with its variable."""
    target = str(tmp_path / 'view-target')
    migrate_view(deprecated_view, target, distributed=False)

    array = store.open_array(
        local, fs_utils.join_path(target, PARTITIONS[0], VIEW_VARIABLE.name))
    array[...] = array[...] + 1

    outcome = check_view(deprecated_view, target, distributed=False)

    assert outcome.failed == ((PARTITIONS[0], VIEW_VARIABLE.name), )
    assert outcome.done == PARTITIONS[1:]


def test_check_names_the_attributes_of_a_group_that_changed(
        deprecated, tmp_path) -> None:
    """The copy carries the attributes of the dataset over too."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    partition = fs_utils.join_path(target, PARTITIONS[0])
    store.write_group_attributes(local, partition, {'title': 'another'})
    store.consolidate(local, partition)

    outcome = check(deprecated, target, distributed=False)

    assert outcome.failed == ((PARTITIONS[0], 'attributes'), )


def test_check_names_a_variable_whose_fill_value_changed(deprecated,
                                                         tmp_path) -> None:
    """A fill value is read even where it masks no element.

    ``plain`` declares none and ``counts`` declares one, so the two sides
    of the comparison are both read here.
    """
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    partition = fs_utils.join_path(target, PARTITIONS[0])
    for name, value in (('plain', -1234.0), ('counts', -98)):
        array = store.open_array(local, fs_utils.join_path(partition, name))
        array.attrs[FILL_VALUE] = value
    store.consolidate(local, partition)

    outcome = check(deprecated, target, distributed=False)

    assert outcome.failed == ((PARTITIONS[0], 'counts, plain'), )


def test_a_selection_that_holds_no_partition_writes_nothing(
        deprecated, tmp_path) -> None:
    """A filter matching nothing leaves every field of the outcome empty."""
    target = str(tmp_path / 'target')

    outcome = migrate(deprecated,
                      target,
                      filters='day == 99',
                      distributed=False)

    assert outcome == migration.Outcome((), (), ())
    assert status(deprecated, target).missing == PARTITIONS


def test_check_view_skips_a_partition_the_target_does_not_hold(
        deprecated_view, tmp_path) -> None:
    """A view the copy has not been written yet compares nothing."""
    target = str(tmp_path / 'view-target')

    outcome = check_view(deprecated_view, target, distributed=False)

    assert outcome.skipped == PARTITIONS
    assert outcome.done == ()
    assert outcome.failed == ()


def test_check_names_a_variable_the_target_masks_otherwise(
        deprecated, tmp_path) -> None:
    """Two variables holding one absence apart are read as differing."""
    target = str(tmp_path / 'target')
    migrate(deprecated, target, distributed=False)

    array = store.open_array(
        local, fs_utils.join_path(target, PARTITIONS[0], 'masked'))
    assert numpy.ma.getmaskarray(
        read_collection(deprecated, local).variables['masked'].values).any()
    array[...] = numpy.ones(array.shape, dtype=array.dtype)

    outcome = check(deprecated, target, distributed=False)

    assert outcome.failed == ((PARTITIONS[0], 'masked'), )


def test_a_filter_migrates_only_the_view_partitions_it_selects(
        deprecated_view, tmp_path) -> None:
    """A view is selected the way a collection is."""
    target = str(tmp_path / 'view-target')

    outcome = migrate_view(deprecated_view,
                           target,
                           filters='day == 2',
                           distributed=False)

    assert outcome.done == (PARTITIONS[1], )
    assert not local.exists(fs_utils.join_path(target, PARTITIONS[0]))
    assert view_status(deprecated_view,
                       target).missing == (PARTITIONS[0], PARTITIONS[2])

    checked = check_view(deprecated_view,
                         target,
                         filters='day == 2',
                         distributed=False)
    assert checked.done == (PARTITIONS[1], )
