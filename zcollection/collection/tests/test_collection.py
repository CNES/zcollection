# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of the collections
=======================
"""
from __future__ import annotations

from typing import cast
import concurrent.futures
import copy
import datetime
import io
import json
import logging
import multiprocessing
import time
from unittest import mock

import dask.distributed
import fsspec
import numcodecs
import numpy
import pytest
import zarr.codecs
import zarr.codecs.numcodecs

from ... import (
    PartitionInfo,
    collection,
    convenience,
    dask_utils,
    dataset,
    execution,
    merging,
    meta,
    partitioning,
    storage,
    store,
    sync,
    variable,
)
from ...tests import reference
from ...tests.cluster import dask_client, dask_cluster  # noqa: F401
from ...tests.data import (
    DELTA,
    END_DATE,
    FILE_SYSTEM_DATASET,
    START_DATE,
    create_test_collection,
    create_test_dataset,
    create_test_dataset_with_fillvalue,
    fail_after,
    record_lock,
    with_process_sync,
)
from ...tests.fixture import dask_arrays, numpy_arrays  # noqa: F401
from ...tests.fs import local_fs  # noqa: F401

_SHORT_FILLVALUE_ROWS = slice(8, 18)


def _update_test_dataset(create_test_data, delayed: bool,
                         distributed: bool) -> dataset.Dataset:
    """Return a dataset with the date span used by update tests.

    Fill-value data has 61 dates when ``delayed`` and ``distributed`` are
    both true. Other fill-value cases use ten dates across two months.
    Data without fill values keeps its original size.
    """
    if create_test_data is not create_test_dataset_with_fillvalue:
        return next(create_test_data(delayed=delayed))

    full = delayed and distributed
    rows = None if full else _SHORT_FILLVALUE_ROWS
    zds = next(create_test_data(delayed=delayed, rows=rows))
    dates = zds.variables['time'].values
    assert dates.shape[0] == (61 if full else 10)
    if not full:
        months = dates.astype('datetime64[M]')
        assert months[0] != months[-1]
    return zds


@pytest.fixture(params=['local_fs', 's3_fs'])
def mutation_collection(request):
    """Write a collection with partitioned and immutable variables."""
    tested_fs = request.getfixturevalue(request.param)
    zds = reference.make_dataset().select_vars(['time', 'plain', 'pixel'])
    zcol = collection.Collection(
        axis='time',
        ds=zds.metadata(),
        partition_handler=reference.make_partitioning(),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs,
        mode='w',
    )
    zcol.insert(zds, distributed=False)
    return zcol


def collection_files(zcol) -> dict[str, bytes]:
    """Read every file of a collection, including its partition index."""
    return {
        name: zcol.fs.cat_file(name)
        for name in zcol.fs.find(zcol.partition_properties.dir)
    }


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_hands_the_merge_callable_the_representation_asked_for(
        local_fs, distributed) -> None:  # noqa: F811
    """``delayed`` sets the representation the merge callable receives."""
    original = reference.make_dataset()
    zcol = convenience.create_collection('time',
                                         original,
                                         reference.make_partitioning(),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs)
    zcol.insert(original, distributed=False)
    incoming = reference.make_dataset(delayed=True)

    def merge(existing_ds: dataset.Dataset, inserted_ds: dataset.Dataset,
              axis: str, partitioning_dim: str, **kwargs) -> dataset.Dataset:
        assert not existing_ds.delayed
        assert not inserted_ds.delayed
        return inserted_ds

    zcol.insert(incoming,
                merge_callable=merge,
                delayed=False,
                distributed=distributed)

    loaded = zcol.load(delayed=False, distributed=False)
    assert loaded is not None


def ten_daily_partitions() -> dataset.Dataset:
    """Return a dataset spreading over ten daily partitions."""
    time = numpy.datetime64(
        '2000-01-01', 'ns') + numpy.arange(40) * numpy.timedelta64(6, 'h')
    return dataset.Dataset([
        dataset.Array('time', time, ('num_lines', )),
        dataset.Array('value', numpy.ones(40), ('num_lines', )),
    ],
                           delayed=False)


def fail_on_the_fourth_day(existing_ds: dataset.Dataset,
                           inserted_ds: dataset.Dataset, axis: str,
                           partitioning_dim: str, **kwargs) -> dataset.Dataset:
    """Fail on the partition of 2000-01-04, and keep the inserted data."""
    if inserted_ds['time'].values[0] == numpy.datetime64('2000-01-04', 'ns'):
        raise ValueError('merge failed')
    return inserted_ds


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize(('stop_on_error', 'completed', 'not_attempted'), [
    (True, range(3), range(4, 10)),
    (False, [0, 1, 2, 4, 5, 6, 7, 8, 9], ()),
])
def test_insert_reports_each_partition(
        local_fs,  # noqa: F811
        distributed,
        stop_on_error,
        completed,
        not_attempted) -> None:
    """A failed merge gives the partitions completed, failed and not
    attempted, in paths relative to the collection."""
    inserted = ten_daily_partitions()
    zcol = convenience.create_collection('time',
                                         inserted,
                                         partitioning.Date(('time', ), 'D'),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs)
    names = zcol.insert(inserted, distributed=distributed)
    assert names == tuple(f'year=2000/month=01/day={day:02d}'
                          for day in range(1, 11))

    with pytest.raises(execution.PartitionErrors) as info:
        zcol.insert(inserted,
                    merge_callable=fail_on_the_fourth_day,
                    npartitions=1,
                    distributed=distributed,
                    stop_on_error=stop_on_error)

    assert info.value.completed == tuple(names[index] for index in completed)
    assert list(info.value.failed) == [names[3]]
    assert 'merge failed' in info.value.failed[names[3]]
    assert info.value.not_attempted == tuple(names[index]
                                             for index in not_attempted)


def keep_inserted(existing_ds: dataset.Dataset, inserted_ds: dataset.Dataset,
                  axis: str, partitioning_dim: str,
                  **kwargs) -> dataset.Dataset:
    """Keep the inserted data, checking the keyword arguments."""
    assert kwargs == {'zds': 'merge argument', 'fs': 'merge argument'}
    return inserted_ds


def test_insert_passes_its_keyword_arguments_to_the_merge_callable(
        local_fs) -> None:  # noqa: F811
    """A keyword argument of insert reaches the merge callable, also when an
    internal argument of the insertion has the same name."""
    inserted = ten_daily_partitions()
    zcol = convenience.create_collection('time',
                                         inserted,
                                         partitioning.Date(('time', ), 'D'),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs)
    zcol.insert(inserted, distributed=False)

    assert len(
        zcol.insert(inserted,
                    merge_callable=keep_inserted,
                    distributed=False,
                    zds='merge argument',
                    fs='merge argument')) == 10


class RecordingSync(sync.Sync):
    """A synchronizer that reports whether it is entered."""

    def __init__(self) -> None:
        self.held = False

    def __enter__(self) -> bool:
        self.held = True
        return True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.held = False

    def is_locked(self) -> bool:
        return self.held


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_writes_the_immutable_variables_under_the_lock(
        local_fs, distributed) -> None:  # noqa: F811
    """The immutable variables are written while the collection holds its
    lock."""
    original = reference.make_dataset()
    zcol = convenience.create_collection('time',
                                         original,
                                         reference.make_partitioning(),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs,
                                         synchronizer=RecordingSync())
    write = storage.write_zarr_group_missing
    held: list[bool] = []

    def record(*args, **kwargs):
        held.append(zcol.synchronizer.held)
        return write(*args, **kwargs)

    with mock.patch.object(storage, 'write_zarr_group_missing', record):
        zcol.insert(original, distributed=distributed)

    assert held == [True]


@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('npartitions', [None, 2, 0])
def test_insert_of_an_empty_dataset_writes_nothing(
        local_fs, npartitions, distributed) -> None:  # noqa: F811
    """Inserting a dataset without rows writes no partition and asks for no
    Dask client."""
    inserted = ten_daily_partitions()
    zcol = convenience.create_collection('time',
                                         inserted,
                                         partitioning.Date(('time', ), 'D'),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs)

    with mock.patch.object(dask_utils,
                           'get_client',
                           side_effect=AssertionError):
        written = zcol.insert(inserted.isel({'num_lines': slice(0, 0)}),
                              npartitions=npartitions,
                              distributed=distributed)

    assert written == ()
    assert tuple(zcol.partitions()) == ()


def test_insert_refuses_a_delayed_merge_of_materialized_data(
        local_fs) -> None:  # noqa: F811
    """``delayed`` set to True needs a dataset of Dask arrays."""
    original = reference.make_dataset()
    zcol = convenience.create_collection('time',
                                         original,
                                         reference.make_partitioning(),
                                         str(local_fs.collection),
                                         filesystem=local_fs.fs)

    with pytest.raises(ValueError, match='delayed is True'):
        zcol.insert(original, delayed=True, distributed=False)


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('delayed', [False, True])
@pytest.mark.parametrize('distributed', [False, True])
def test_partial_insert_preserves_filler_flags_for_merging(
        fs, request, delayed, distributed) -> None:
    """A merge identifies missing variables and retains their stored values."""
    tested_fs = request.getfixturevalue(fs)
    original = reference.make_dataset().select_vars(
        ['time', 'plain', 'counts'])
    zcol = convenience.create_collection('time',
                                         original,
                                         reference.make_partitioning(),
                                         str(tested_fs.collection),
                                         filesystem=tested_fs.fs)
    zcol.insert(original, distributed=False)
    incoming = reference.make_dataset(delayed=delayed).select_vars(
        ['time', 'plain'])
    incoming['plain'].array = incoming['plain'].array + 100

    def merge(existing_ds: dataset.Dataset, inserted_ds: dataset.Dataset,
              axis: str, partitioning_dim: str, **kwargs) -> dataset.Dataset:
        assert existing_ds.delayed is delayed
        assert inserted_ds.delayed is delayed
        assert {
            name: var.filler
            for name, var in inserted_ds.variables.items()
        } == {
            'time': False,
            'plain': False,
            'counts': True,
        }
        return dataset.Dataset(existing_ds[name] if var.filler else var
                               for name, var in inserted_ds.variables.items())

    zcol.insert(incoming, merge_callable=merge, distributed=distributed)

    loaded = zcol.load(delayed=False, distributed=False)
    assert loaded is not None
    numpy.testing.assert_array_equal(loaded['plain'].array,
                                     reference.plain_values() + 100)
    numpy.testing.assert_array_equal(loaded['counts'].array,
                                     reference.count_values())


@pytest.mark.parametrize('method', ['update_immutable', 'drop_dimension'])
def test_readonly_refuses_schema_and_immutable_updates(mutation_collection,
                                                       method) -> None:
    """A refused write preserves the schema, arrays and partition index."""
    zcol = mutation_collection
    readonly = convenience.open_collection(zcol.partition_properties.dir,
                                           filesystem=zcol.fs,
                                           mode='r')
    schema = copy.deepcopy(readonly.metadata)
    before = collection_files(zcol)
    args = (('pixel',
             reference.pixel_values() + 1) if method == 'update_immutable' else
            (reference.IMMUTABLE_DIMENSION, ))

    with pytest.raises(io.UnsupportedOperation):
        getattr(readonly, method)(*args)

    assert readonly.metadata == schema
    assert collection_files(zcol) == before


@pytest.mark.parametrize(('dimension', 'variable_name'), [
    (reference.DIMENSION, 'time'),
    (reference.IMMUTABLE_DIMENSION, 'pixel'),
])
def test_drop_dimension_refuses_used_dimensions(mutation_collection, dimension,
                                                variable_name) -> None:
    """A dimension used by any variable remains in memory and on disk."""
    zcol = mutation_collection
    schema = copy.deepcopy(zcol.metadata)
    before = collection_files(zcol)

    with pytest.raises(ValueError, match=variable_name):
        zcol.drop_dimension(dimension)

    assert zcol.metadata == schema
    assert collection_files(zcol) == before
    reopened = convenience.open_collection(zcol.partition_properties.dir,
                                           filesystem=zcol.fs)
    assert reopened.metadata == schema


def test_drop_dimension_accepts_a_dimension_after_its_variable_is_removed(
        mutation_collection) -> None:
    """A dimension becomes removable after its last variable is dropped."""
    zcol = mutation_collection
    zcol.drop_variable('pixel')
    zcol.drop_dimension(reference.IMMUTABLE_DIMENSION)

    reopened = convenience.open_collection(zcol.partition_properties.dir,
                                           filesystem=zcol.fs)
    assert reference.IMMUTABLE_DIMENSION not in reopened.metadata.dimensions
    assert reopened.load(delayed=False, distributed=False) is not None


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('invalid', [
    'unknown',
    'immutable_dtype',
    'immutable_size',
    'immutable_dimensions',
    'immutable_fill',
    'partition_dtype',
    'partition_dimensions',
])
def test_insert_validates_all_variables_before_writing(fs, request,
                                                       invalid) -> None:
    """An invalid variable leaves the immutable store and index intact."""
    tested_fs = request.getfixturevalue(fs)
    declared = reference.make_dataset().select_vars(['time', 'grid', 'pixel'])
    zcol = convenience.create_collection('time',
                                         declared,
                                         reference.make_partitioning(),
                                         str(tested_fs.collection),
                                         filesystem=tested_fs.fs)
    incoming = copy.deepcopy(declared)
    if invalid == 'unknown':
        incoming.add_variable(dataset.Array('unknown', numpy.array(42), ()))
    elif invalid == 'immutable_dtype':
        incoming['pixel'].array = incoming['pixel'].array.astype('float64')
    elif invalid == 'immutable_size':
        incoming = dataset.Dataset([
            incoming['time'],
            dataset.Array('pixel', numpy.arange(4, dtype='int16'),
                          (reference.IMMUTABLE_DIMENSION, )),
        ])
    elif invalid == 'immutable_dimensions':
        incoming['pixel'].dimensions = ('other_pixels', )
        incoming = dataset.Dataset(incoming.variables.values())
    elif invalid == 'immutable_fill':
        incoming['pixel'].fill_value = -1
    elif invalid == 'partition_dtype':
        incoming['grid'].array = incoming['grid'].array.astype('int16')
    else:
        incoming.variables['grid'] = incoming['grid'].duplicate(
            incoming['grid'].array.T,
            dimensions=(reference.IMMUTABLE_DIMENSION, reference.DIMENSION))
    before = collection_files(zcol)

    with pytest.raises(ValueError, match=r'unknown|invalid'):
        zcol.insert(incoming, distributed=False)

    assert collection_files(zcol) == before
    assert tuple(zcol.partitions()) == ()


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('invalid', ['unknown', 'dtype'])
def test_insert_validates_immutable_only_data(fs, request, invalid) -> None:
    """An immutable-only insertion follows the collection schema."""
    tested_fs = request.getfixturevalue(fs)
    declared = reference.make_dataset()
    zcol = convenience.create_collection('time',
                                         declared,
                                         reference.make_partitioning(),
                                         str(tested_fs.collection),
                                         filesystem=tested_fs.fs)
    incoming = (dataset.Dataset([
        dataset.Array('unknown', numpy.array(42), ())
    ]) if invalid == 'unknown' else dataset.Dataset([
        declared['pixel'].duplicate(declared['pixel'].array.astype('float64'))
    ]))
    before = collection_files(zcol)

    with pytest.raises(ValueError, match=r'unknown|invalid'):
        zcol.insert(incoming, distributed=False)

    assert collection_files(zcol) == before


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_insert_refuses_reassigning_a_variables_dimension(fs, request) -> None:
    """Equal dimension sizes do not allow changing a variable's dimension."""
    tested_fs = request.getfixturevalue(fs)
    declared = reference.make_dataset().select_vars(
        ['time', 'plain', 'pixel']).isel({reference.DIMENSION: slice(0, 3)})
    zcol = convenience.create_collection('time',
                                         declared,
                                         reference.make_partitioning(),
                                         str(tested_fs.collection),
                                         filesystem=tested_fs.fs)
    incoming = dataset.Dataset([
        declared['time'],
        declared['plain'].duplicate(
            dimensions=(reference.IMMUTABLE_DIMENSION, )),
        declared['pixel'],
    ])
    before = collection_files(zcol)

    with pytest.raises(ValueError, match='invalid dimensions'):
        zcol.insert(incoming, distributed=False)

    assert collection_files(zcol) == before


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('immutable_only', [False, True])
def test_insert_immutables_uses_the_declared_encoding(fs, request,
                                                      immutable_only) -> None:
    """An immutable takes its encoding once and keeps its stored values."""
    tested_fs = request.getfixturevalue(fs)
    incoming = reference.make_dataset().select_vars(['time', 'grid', 'pixel'])
    declared = copy.deepcopy(incoming)
    declared['pixel'].compressor = zarr.codecs.GzipCodec(level=3)
    declared['pixel'].filters = (zarr.codecs.numcodecs.Delta(dtype='i2'), )
    declared['pixel'].fill_value = -99
    schema = declared.metadata()
    schema.dimensions[reference.IMMUTABLE_DIMENSION].chunks = 1
    zcol = collection.Collection('time',
                                 schema,
                                 reference.make_partitioning(),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs,
                                 mode='w')
    if immutable_only:
        incoming = incoming.select_vars('pixel')

    written = zcol.insert(incoming, distributed=False)

    assert bool(written) is not immutable_only
    array = store.open_array(zcol.fs, f'{zcol.immutable_path}/pixel')
    assert array.chunks == (1, )
    assert cast('zarr.abc.codec.BytesBytesCodec', array.compressors[0]
                ).to_dict() == declared['pixel'].compressor.to_dict()
    assert cast(
        'zarr.abc.codec.ArrayArrayCodec',
        array.filters[0]).to_dict() == declared['pixel'].filters[0].to_dict()
    loaded = zcol.load(delayed=False, distributed=False)
    assert loaded is not None
    assert loaded['pixel'].fill_value == -99
    numpy.testing.assert_array_equal(loaded['pixel'].values,
                                     reference.pixel_values())
    assert incoming['pixel'].fill_value is None
    incoming['pixel'].array = incoming['pixel'].array + 10
    zcol.insert(incoming, distributed=False)
    loaded = zcol.load(delayed=False, distributed=False)
    assert loaded is not None
    numpy.testing.assert_array_equal(loaded['pixel'].values,
                                     reference.pixel_values())


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('delayed', [False, True])
@pytest.mark.parametrize('vocabulary', ['native', 'numcodecs'])
def test_codec_schema_survives_loading_and_copying_a_collection(
        fs, request, delayed, vocabulary) -> None:
    """Loaded codecs retain their parameters and values in a new collection."""
    tested_fs = request.getfixturevalue(fs)
    zds = reference.make_dataset(delayed=delayed).select_vars(
        ['time', 'grid', 'pixel'])
    zds['grid'].fill_value = reference.FILL_VALUE
    if vocabulary == 'native':
        zds['grid'].compressor = zarr.codecs.ZstdCodec(level=5, checksum=True)
        zds['grid'].filters = (zarr.codecs.numcodecs.FixedScaleOffset(
            offset=0, scale=4, dtype='f8', astype='i8'), )
    else:
        zds['grid'].compressor = numcodecs.Zstd(level=5)
        zds['grid'].filters = (numcodecs.FixedScaleOffset(offset=0,
                                                          scale=4,
                                                          dtype='f8',
                                                          astype='i8'), )
    zds['pixel'].compressor = zds['grid'].compressor
    source = convenience.create_collection('time',
                                           zds,
                                           reference.make_partitioning(),
                                           str(tested_fs.collection),
                                           filesystem=tested_fs.fs)
    source.insert(zds, distributed=False)
    loaded = source.load(delayed=delayed, distributed=False)
    assert loaded is not None
    schema = loaded.metadata()
    restored = meta.DatasetSchema.from_config(
        json.loads(json.dumps(schema.get_config())))
    assert restored == schema

    target = convenience.create_collection('time',
                                           loaded,
                                           reference.make_partitioning(),
                                           f'{tested_fs.collection}-copy',
                                           filesystem=tested_fs.fs)
    target.insert(loaded, distributed=False)
    reopened = convenience.open_collection(target.partition_properties.dir,
                                           filesystem=tested_fs.fs)
    copied = reopened.load(delayed=delayed, distributed=False)
    assert copied is not None
    for name in loaded.variables:
        original_compressor = loaded[name].compressor
        copied_compressor = copied[name].compressor
        if original_compressor is None:
            assert copied_compressor is None
        else:
            assert copied_compressor is not None
            assert copied_compressor.to_dict() == original_compressor.to_dict()
        assert [item.to_dict() for item in copied[name].filters
                ] == [item.to_dict() for item in loaded[name].filters]
        numpy.testing.assert_array_equal(copied[name].values,
                                         loaded[name].values)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_native_filter_rejects_a_fill_value_it_changes(fs, request) -> None:
    """Native filters must preserve the declared fill value."""
    tested_fs = request.getfixturevalue(fs)
    zds = reference.make_dataset().select_vars(['time', 'grid'])
    zds['grid'].fill_value = 1.1
    zds['grid'].filters = (zarr.codecs.numcodecs.FixedScaleOffset(
        offset=0, scale=1, dtype='f8', astype='i8'), )

    with pytest.raises(ValueError, match='filters read back'):
        convenience.create_collection('time',
                                      zds,
                                      reference.make_partitioning(),
                                      str(tested_fs.collection),
                                      filesystem=tested_fs.fs)

    assert not tested_fs.fs.exists(str(tested_fs.collection))


def test_constructor_rejects_an_unsized_secondary_dimension(
        local_fs) -> None:  # noqa: F811
    """Collection construction checks the sizes of secondary dimensions."""
    schema = reference.make_dataset().metadata()
    schema.dimensions['num_pixels'].value = -1

    with pytest.raises(ValueError,
                       match="'num_pixels' is declared without a size"):
        collection.Collection(
            axis='time',
            ds=schema,
            partition_handler=reference.make_partitioning(),
            partition_base_dir=str(local_fs.collection),
            filesystem=local_fs.fs,
            mode='w',
        )


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_collection_creation(
    fs,
    request,
) -> None:
    """Test the creation of a collection."""
    tested_fs = request.getfixturevalue(fs)
    zds = next(create_test_dataset(False))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    assert isinstance(str(zcol), str)
    assert zcol.have_immutable is False
    assert zcol.load(distributed=False) is None

    serialized = collection.Collection.from_config(
        path=str(tested_fs.collection),
        filesystem=tested_fs.fs,
    )
    assert serialized.axis == zcol.axis
    assert serialized.metadata == zcol.metadata
    assert serialized.partition_properties == zcol.partition_properties
    assert serialized.partitioning.get_config(
    ) == zcol.partitioning.get_config()

    with pytest.raises(ValueError,
                       match=r'zarr collection not found at path .*'):
        collection.Collection.from_config(str(tested_fs.collection.parent))

    with pytest.raises(ValueError,
                       match=r'The variable .* is not defined in the dataset'):
        collection.Collection('time_tai', zds.metadata(),
                              partitioning.Date(('time', ), 'D'),
                              str(tested_fs.collection))


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_collection_written_in_a_newer_layout(fs, request) -> None:
    """A collection written in a layout this code does not know is refused."""
    tested_fs = request.getfixturevalue(fs)
    zds = next(create_test_dataset(False))
    collection.Collection(axis='time',
                          ds=zds.metadata(),
                          partition_handler=partitioning.Date(('time', ), 'D'),
                          partition_base_dir=str(tested_fs.collection),
                          filesystem=tested_fs.fs)

    config = collection.Collection._config(str(tested_fs.collection))
    with tested_fs.fs.open(config) as stream:
        data = json.load(stream)
    data['version'] = meta.FORMAT_VERSION + 1
    with tested_fs.fs.open(config, mode='w') as stream:
        json.dump(data, stream)

    with pytest.raises(ValueError, match='written in version'):
        collection.Collection.from_config(path=str(tested_fs.collection),
                                          filesystem=tested_fs.fs)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_collection_holding_a_package_version(fs, request) -> None:
    """A collection whose version key holds a package version opens.

    Before the layout carried a version of its own, the key held the version
    of the package that wrote the collection. Such collections name the
    current layout.
    """
    tested_fs = request.getfixturevalue(fs)
    zds = next(create_test_dataset(False))
    collection.Collection(axis='time',
                          ds=zds.metadata(),
                          partition_handler=partitioning.Date(('time', ), 'D'),
                          partition_base_dir=str(tested_fs.collection),
                          filesystem=tested_fs.fs)

    config = collection.Collection._config(str(tested_fs.collection))
    with tested_fs.fs.open(config) as stream:
        data = json.load(stream)
    data['version'] = '2026.6.1.dev3+g5943e72'
    with tested_fs.fs.open(config, mode='w') as stream:
        json.dump(data, stream)

    zcol = collection.Collection.from_config(path=str(tested_fs.collection),
                                             mode='w',
                                             filesystem=tested_fs.fs)
    assert zcol.version == meta.FORMAT_VERSION
    assert zcol.mode == 'w'

    with pytest.raises(
            ValueError,
            match=r'The partitioning key .* is not defined in the dataset.'):
        collection.Collection('time', zds.metadata(),
                              partitioning.Date(('time_tai', ), 'D'),
                              str(tested_fs.collection))

    with pytest.raises(ValueError, match=r'he mode .* is not supported'):
        collection.Collection(
            axis='time',
            ds=zds.metadata(),
            mode='X',  # type: ignore[arg-type]
            partition_handler=partitioning.Date(('time', ), 'D'),
            partition_base_dir=str(tested_fs.collection),
            filesystem=tested_fs.fs)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert(  # noqa: PLR0915
    dask_client,  # noqa: F811
    arrays_type,
    distributed,
    fs,
    request,
    tmp_path,
) -> None:
    """Test the insertion of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    datasets = list(create_test_dataset(delayed=False))
    zcol = collection.Collection('time',
                                 datasets[0].metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs,
                                 synchronizer=sync.ProcessSync(
                                     str(tmp_path / 'lock.lck')))

    indices = numpy.arange(0, len(datasets))
    rng = numpy.random.default_rng(42)
    rng.shuffle(indices)
    for idx in indices:
        zcol.insert(datasets[idx],
                    merge_callable=merging.merge_time_series,
                    distributed=distributed)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    # Adding same datasets once more (should not change anything)
    for idx in indices[:5]:
        zcol.insert(datasets[idx], distributed=distributed)

    assert list(zcol.partitions()) == sorted(zcol.partitions())

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    values = data.variables['time'].values
    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))

    values = data.variables['var1'].values
    numpy.testing.assert_array_equal(
        values,
        numpy.vstack((numpy.arange(values.shape[0]), ) * values.shape[1]).T)

    values = data.variables['var2'].values
    numpy.testing.assert_array_equal(
        values,
        numpy.vstack((numpy.arange(values.shape[0]), ) * values.shape[1]).T)

    data = zcol.load(delayed=delayed,
                     filters='year == 2020',
                     distributed=distributed)
    assert data is None

    data = zcol.load(delayed=delayed,
                     filters='year == 2000',
                     distributed=distributed)
    assert data is not None
    assert data.variables['time'].shape[0] == 61

    data = zcol.load(delayed=delayed,
                     filters='year == 2000 and month == 4',
                     distributed=distributed)
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[M]') == numpy.datetime64('2000-04-01'))

    data = zcol.load(delayed=delayed,
                     filters='year == 2000 and month == 4 and day == 15',
                     distributed=distributed)
    assert data is not None
    dates = data.variables['time'].values
    assert numpy.all(
        dates.astype('datetime64[D]') == numpy.datetime64('2000-04-15'))

    data = zcol.load(
        delayed=delayed,
        filters='year == 2000 and month == 4 and day in range(5, 25)',
        distributed=distributed)
    assert data is not None
    data = zcol.load(delayed=delayed,
                     filters=lambda keys: datetime.date(2000, 4, 5) <= datetime
                     .date(keys['year'], keys['month'], keys['day']
                           ) <= datetime.date(2000, 4, 24),
                     distributed=distributed)
    assert data is not None
    dates = data.variables['time'].values.astype('datetime64[D]')
    assert dates.min() == numpy.datetime64('2000-04-06')
    assert dates.max() == numpy.datetime64('2000-04-24')

    for path_relative, path_absolute in zcol.iterate_on_records():
        assert isinstance(path_relative, str)
        assert isinstance(path_absolute, str)
        assert path_relative in path_absolute

    zcol = convenience.open_collection(str(tested_fs.collection),
                                       mode='r',
                                       filesystem=tested_fs.fs)
    zds = zcol.load(delayed=delayed,
                    selected_variables=['var1'],
                    distributed=distributed)
    assert zds is not None
    assert 'var1' in zds.variables
    assert 'var2' not in zds.variables

    zds = zcol.load(delayed=delayed,
                    selected_variables=[],
                    distributed=distributed)
    assert zds is not None
    assert len(zds.variables) == 0

    zds = zcol.load(delayed=delayed,
                    selected_variables=['varX'],
                    distributed=distributed)
    assert zds is not None
    assert len(zds.variables) == 0


@pytest.mark.parametrize(('fs', 'create_test_data'), FILE_SYSTEM_DATASET)
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_update(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    create_test_data,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = _update_test_dataset(create_test_data, delayed, distributed)
    zcol = collection.Collection('time',
                                 zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds, distributed=distributed)
    assert len(tuple(zcol.partitions())) == zds.variables['time'].shape[0]

    def update(zds: dataset.Dataset, shift: int = 3):
        """Update function used for this test."""
        return {'var2': zds.variables['var1'].values * -1 + shift}

    zcol.update(
        update,
        delayed=delayed,  # type: ignore[arg-type]
        distributed=distributed)

    data = zcol.load(distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 3,
                          rtol=0)

    zcol.update(
        update,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        variables=('var2', ),
        depth=1,
        shift=5)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 5,
                          rtol=0)

    # Test case if the selected variables do not contain the variable
    # to update.
    zcol.update(
        update,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        selected_variables=['var1'],
        depth=1,
        shift=5)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 5,
                          rtol=0)

    def update_with_info(zds: dataset.Dataset, partition_info=None, shift=3):
        """Update function used for this test."""
        assert partition_info is not None
        assert partition_info.dimension == 'num_lines'
        return {'var2': zds.variables['var1'].values * -1 + shift}

    zcol.update(
        update_with_info,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        depth=1,
        shift=10)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1 + 10,
                          rtol=0)

    def update_and_trim(zds: dataset.Dataset, partition_info=None):
        """Update function used for this test."""
        assert partition_info is not None
        assert partition_info.dimension == 'num_lines'
        zds = zds.isel({partition_info.dimension: partition_info.indices})
        return {'var2': zds.variables['var1'].values * -1}

    zcol.update(
        update_and_trim,  # type: ignore[arg-type]
        delayed=delayed,
        distributed=distributed,
        trim=False,
        depth=1)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None
    assert numpy.allclose(data.variables['var2'].values,
                          data.variables['var1'].values * -1,
                          rtol=0)

    def invalid_var_name(zds: dataset.Dataset):
        """Update function used to test if the user wants to update a non-
        existent variable name."""
        return {'var99': zds.variables['var1'].values * -1 + 3}

    with pytest.raises(ValueError, match='Unknown variables'):
        zcol.update(
            invalid_var_name,  # type: ignore[arg-type]
            distributed=distributed)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_list_partitions(
    dask_client,  # noqa: F811
    arg,
    request,
) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, delayed=False)

    all_partitions = list(zcol.partitions())
    assert len(all_partitions) == 6

    def full_path(partition):
        return zcol.fs.sep.join((zcol.partition_properties.dir, partition))

    selected_partitions = ['year=2000/month=01/day=01']
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert partitions == list(map(full_path, selected_partitions))

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=01'
    ]
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert partitions == list(map(full_path, selected_partitions[:1]))

    selected_partitions = ['year=2000/month=01/day=02']
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert not partitions

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=02',
        'year=2000/month=01/day=07'
    ]
    partitions = list(zcol.partitions(selected_partitions=selected_partitions))
    assert partitions == list(
        map(full_path, [selected_partitions[0], selected_partitions[2]]))

    indexer = zcol.map(
        lambda x: slice(0, x.dimensions['num_lines'])  # type: ignore[arg-type]
    ).compute()[3:]

    selected_partitions = [
        'year=2000/month=01/day=01', 'year=2000/month=01/day=02',
        'year=2000/month=01/day=13'
    ]
    partitions = list(
        zcol.partitions(indexer=indexer,
                        selected_partitions=selected_partitions))
    assert partitions == list(map(full_path, selected_partitions[-1:]))


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_selected_partitions_on_load_and_update(
    arg,
    request,
) -> None:
    """``selected_partitions`` bounds what is read and what is written."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, delayed=False, distributed=False)
    one_day = ['year=2000/month=01/day=01']

    part = zcol.load(selected_partitions=one_day, distributed=False)
    whole = zcol.load(distributed=False)
    assert part is not None
    assert whole is not None
    assert part.dimensions['num_lines'] < whole.dimensions['num_lines']

    def mark(zds: dataset.Dataset):
        """Mark every value of the partition."""
        return {'var2': zds.variables['var2'].values * 0 - 1}

    zcol.update(mark, selected_partitions=one_day, distributed=False)

    part = zcol.load(selected_partitions=one_day, distributed=False)
    whole = zcol.load(distributed=False)
    assert part is not None
    assert whole is not None
    assert numpy.all(part.variables['var2'].values == -1)
    assert not numpy.all(whole.variables['var2'].values == -1)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_drop_partitions(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs,
                                  delayed=False,
                                  distributed=distributed)

    all_partitions = list(zcol.partitions())
    assert 'month=01' in [
        item.split(zcol.fs.sep)[-2] for item in all_partitions
    ]

    zcol.drop_partitions(filters='year == 2000 and month==1 and day<13',
                         distributed=distributed)
    partitions = list(zcol.partitions())
    part_ge_13 = [
        int(item.split(zcol.fs.sep)[-1][-2:]) >= 13 for item in partitions
    ]
    assert all(part_ge_13)

    npartitions = len(partitions)

    if arg == 's3_fs':
        # Partitions cannot be dropped using a timedelta with S3
        with pytest.raises(NotImplementedError):
            zcol.drop_partitions(timedelta=datetime.timedelta(days=1),
                                 distributed=distributed)
        return

    zcol.drop_partitions(timedelta=datetime.timedelta(days=1),
                         distributed=distributed)
    partitions = list(zcol.partitions())
    assert len(partitions) == npartitions

    zcol.drop_partitions(timedelta=datetime.timedelta(0),
                         distributed=distributed)
    partitions = list(zcol.partitions())
    assert len(partitions) == 0

    zcol = convenience.open_collection(str(tested_fs.collection),
                                       mode='r',
                                       filesystem=tested_fs.fs)
    assert zcol.is_readonly()
    with pytest.raises(io.UnsupportedOperation):
        zcol.drop_partitions(distributed=distributed)


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('stop_on_error', [True, False])
def test_drop_partitions_reports_each_partition(
        local_fs,  # noqa: F811
        monkeypatch,
        distributed,
        stop_on_error) -> None:
    """After a failed removal, ``PartitionErrors`` gives the partitions
    completed, failed and not attempted, and the partitions listed are the
    ones the store holds."""
    zcol = create_test_collection(local_fs, distributed=False)
    names = tuple(zcol.partitions(relative=True))
    fail_after(monkeypatch, collection._drop_partition, names[2])

    with pytest.raises(execution.PartitionErrors) as info:
        zcol.drop_partitions(npartitions=1,
                             distributed=distributed,
                             stop_on_error=stop_on_error)

    remaining = names[3:] if stop_on_error else ()
    assert info.value.completed == names[:2] + (
        () if stop_on_error else names[3:])
    assert list(info.value.failed) == [names[2]]
    assert 'task failed' in info.value.failed[names[2]]
    assert info.value.not_attempted == remaining
    assert zcol._index._read() is not None
    assert tuple(zcol.partitions(relative=True)) == remaining


@pytest.fixture
def five_hours_ahead_of_utc(monkeypatch):
    """Set the local time zone five hours ahead of UTC."""
    monkeypatch.setenv('TZ', 'Etc/GMT-5')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.usefixtures('five_hours_ahead_of_utc')
@pytest.mark.parametrize('aware', [True, False])
def test_drop_partitions_measures_the_age_across_time_zones(
        local_fs, monkeypatch, aware) -> None:  # noqa: F811
    """The age of a partition does not depend on the local time zone, nor
    on whether the file system gives the time in UTC or in local time."""
    zcol = create_test_collection(local_fs, distributed=False)
    names = tuple(zcol.partitions(relative=True))
    created = (datetime.datetime.now(datetime.UTC) -
               datetime.timedelta(minutes=30))
    if not aware:
        created = created.astimezone().replace(tzinfo=None)
    monkeypatch.setattr(zcol.fs, 'created', lambda path: created)

    assert zcol.drop_partitions(timedelta=datetime.timedelta(hours=1),
                                distributed=False) == ()
    assert zcol.drop_partitions(timedelta=datetime.timedelta(minutes=10),
                                distributed=False) == names


@pytest.mark.usefixtures('dask_client')
@pytest.mark.parametrize('distributed', [False, True])
def test_drop_partitions_holds_the_lock(
        local_fs,  # noqa: F811
        monkeypatch,
        tmp_path,
        distributed) -> None:
    """The collection holds its lock while it removes the partitions."""
    zcol = with_process_sync(
        create_test_collection(local_fs, distributed=False), tmp_path)
    held = record_lock(monkeypatch, zcol)
    zcol.drop_partitions(distributed=distributed)

    assert held == [True, True]
    assert not zcol.is_locked()


def test_drop_partitions_skips_an_absent_partition(
        local_fs) -> None:  # noqa: F811
    """A partition the index lists but the store does not hold is left out
    of the paths ``drop_partitions`` returns."""
    zcol = create_test_collection(local_fs, distributed=False)
    names = tuple(zcol.partitions(relative=True))
    local_fs.fs.rm(next(zcol.partitions()), recursive=True)

    dropped = zcol.drop_partitions(distributed=False)

    assert dropped == names[1:]
    assert tuple(zcol.partitions()) == ()


@pytest.mark.parametrize('distributed', [False, True])
def test_drop_partitions_without_a_selected_partition_asks_for_no_client(
        local_fs, distributed) -> None:  # noqa: F811
    """When the filters select no partition, ``drop_partitions`` asks for no
    Dask client and accepts any npartitions."""
    zcol = create_test_collection(local_fs, distributed=False)

    with mock.patch.object(dask_utils,
                           'get_client',
                           side_effect=AssertionError):
        dropped = zcol.drop_partitions(filters='year == 1900',
                                       npartitions=0,
                                       distributed=distributed)

    assert dropped == ()


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_drop_variable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the dropping of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs,
                                  delayed=False,
                                  distributed=distributed)

    with pytest.raises(ValueError,
                       match=r'The variable .* is part of the partitioning'):
        zcol.drop_variable('time', distributed=distributed)
    zcol.drop_variable('var1', distributed=distributed)

    with pytest.raises(ValueError, match='does not exist in the collection'):
        zcol.drop_variable('var1', distributed=distributed)

    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None
    assert 'var1' not in zds.variables

    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='r',
                                       filesystem=tested_fs.fs)

    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None
    assert 'var1' not in zds.variables


def test_add_variable_rejects_another_type(local_fs) -> None:  # noqa: F811
    """Collection.add_variable rejects objects without variable metadata."""
    zcol = collection.Collection(
        axis='time',
        ds=reference.make_dataset().select_vars(['time']).metadata(),
        partition_handler=reference.make_partitioning(),
        partition_base_dir=str(local_fs.collection),
        filesystem=local_fs.fs,
        mode='w',
    )

    invalid_variable = {
        'name': 'anomaly',
        'dtype': 'float32',
        'dimensions': ('num_lines', ),
    }

    with pytest.raises(TypeError, match=r'Expected a meta.VariableSchema'):
        zcol.add_variable(invalid_variable)  # type: ignore[arg-type]


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_add_variable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test adding a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs,
                                  delayed=False,
                                  distributed=distributed)

    # Variable already exists
    new = meta.VariableSchema(name='time',
                              dtype=numpy.dtype('float64'),
                              dimensions=('num_lines', ))
    with pytest.raises(ValueError, match='already exists'):
        zcol.add_variable(new, distributed=distributed)

    # Variable have unknown dimension.
    new = meta.VariableSchema(name='x',
                              dtype=numpy.dtype('float64'),
                              dimensions=('num_lines', 'x'))
    with pytest.raises(ValueError, match='must use the dataset dimensions'):
        zcol.add_variable(new, distributed=distributed)

    new = meta.VariableSchema(
        name='var3',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    zcol.add_variable(new, distributed=distributed)

    assert new.name in zcol.metadata.variables

    # Testing the configuration update by reopening the collection
    zcol = collection.Collection.from_config(path=str(tested_fs.collection),
                                             filesystem=tested_fs.fs)

    assert new.name in zcol.metadata.variables

    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None
    values = zds.variables['var3'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.all(values.mask)  # type: ignore[arg-type]


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_variable_immutable(  # noqa: PLR0915
        dask_client,  # noqa: F811
        arg,
        request) -> None:
    """Test the adding of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, delayed=False)

    # Unknown dimension
    known_dimensions, _ = zcol.dimensions_properties()
    dim = 'num_pixels'

    new_arr = meta.VariableSchema(
        name='var_immutable',
        dtype=numpy.dtype('int16'),
        dimensions=(dim, 'x'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )

    with pytest.raises(ValueError, match='must use the dataset dimensions'):
        zcol.add_variable(variable=new_arr, distributed=False)

    assert new_arr.name not in zcol.metadata.variables

    new = meta.VariableSchema(
        name='var_immutable',
        dtype=numpy.dtype('int16'),
        dimensions=(dim, ),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )

    zcol.add_variable(variable=new, distributed=False)

    assert new.name in zcol.metadata.variables

    # Invalid dimension size
    with pytest.raises(ValueError, match='with an invalid size'):
        zcol.update_immutable(
            name=new.name,
            data=numpy.arange(known_dimensions[dim] * 2, dtype='int16'),
        )

    # Invalid name
    with pytest.raises(ValueError, match='does not exist'):
        zcol.update_immutable(
            name='invalid name',
            data=numpy.arange(known_dimensions[dim] * 2, dtype='int16'),
        )

    data = zcol.load(selected_variables=[new_arr.name],
                     delayed=False,
                     distributed=False)

    assert data is not None
    assert numpy.array_equal(
        data[new_arr.name].values,
        numpy.full(shape=known_dimensions[dim], fill_value=new.fill_value))

    arr_data = numpy.arange(known_dimensions[dim], dtype='int16')
    zcol.update_immutable(name=new.name, data=arr_data)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)
    assert new_arr.name in zcol.metadata.variables

    data = zcol.load(selected_variables=[new_arr.name],
                     delayed=False,
                     distributed=False)

    assert data is not None
    assert numpy.array_equal(data[new_arr.name].values, arr_data)
    assert data[new_arr.name].attrs == new_arr.attrs
    assert data[new_arr.name].fill_value == new_arr.fill_value

    # Adding missing dimension
    new_dim = meta.Dimension(name='x', value=2, chunks=1)
    zcol.add_dimension(dimension=new_dim)

    new_data = numpy.arange(known_dimensions[dim] * new_dim.value,
                            dtype='int16').reshape(known_dimensions[dim],
                                                   new_dim.value)

    new_x = variable.Array(name='var_immutable_x',
                           data=new_data,
                           dimensions=(dim, new_dim.name))

    zcol.add_variable(variable=new_x, distributed=False)
    zcol.update_immutable(name=new_x.name, data=new_x.values)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)
    assert new_x.name in zcol.metadata.variables

    data = zcol.load(selected_variables=[new_x.name],
                     delayed=True,
                     distributed=True)

    assert data is not None
    new_data_r = data[new_x.name]

    assert numpy.array_equal(new_data_r.values, new_x.values)
    assert new_data_r.attrs == new_x.attrs
    assert new_data_r.fill_value == new_x.fill_value
    assert new_data_r.data.chunksize[1] == new_dim.chunks

    # Updating immutable with the update function
    with pytest.raises(ValueError, match='Immutable variables'):
        zcol.update(
            lambda zds: {new_x.name: 5},  # type: ignore[arg-type]
            distributed=False,
        )

    zcol.drop_variable(variable=new_x.name)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)
    assert new_arr.name in zcol.metadata.variables
    assert new_x.name not in zcol.metadata.variables

    zcol.drop_variable(variable=new_arr.name)

    assert new_arr.name not in zcol.metadata.variables

    data = zcol.load(selected_variables=[new_x.name],
                     delayed=True,
                     distributed=True)

    assert data is not None

    assert new_x.name not in data.variables
    assert new_arr.name not in data.variables


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
def test_dimension_add_drop(
        dask_client,  # noqa: F811
        arg,
        request) -> None:
    """Test the adding of a variable."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, delayed=False)

    dim_dum = meta.Dimension(name='dummy', value=20, chunks=2)
    zcol.add_dimension(dimension=dim_dum)

    with pytest.raises(ValueError, match='already exists in the'):
        zcol.add_dimension(dimension=dim_dum)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)

    assert dim_dum.name in zcol.metadata.dimensions
    assert dim_dum.name in zcol.metadata.dim_size
    assert dim_dum.name in zcol.metadata.dim_chunks

    zcol.drop_dimension(dimension=dim_dum.name)

    # Reopening collection
    zcol = convenience.open_collection(path=str(tested_fs.collection),
                                       mode='w',
                                       filesystem=tested_fs.fs)

    assert dim_dum.name not in zcol.metadata.dimensions
    assert dim_dum.name not in zcol.metadata.dim_size
    assert dim_dum.name not in zcol.metadata.dim_chunks


@pytest.mark.parametrize(('fs', 'create_test_data'), FILE_SYSTEM_DATASET)
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_add_update(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    create_test_data,
    request,
) -> None:
    """Test the adding and updating of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = _update_test_dataset(create_test_data, delayed, distributed)
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds, distributed=distributed)
    assert len(tuple(zcol.partitions())) == zds.variables['time'].shape[0]

    new1 = meta.VariableSchema(name='var3',
                               dtype=numpy.dtype('float64'),
                               dimensions=('num_lines', 'num_pixels'),
                               attrs=(dataset.Attribute(name='attr',
                                                        value=1), ),
                               fill_value=1000000.5)

    new2 = meta.VariableSchema(
        name='var4',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    zcol.add_variable(new1, distributed=distributed)
    zcol.add_variable(new2, distributed=distributed)

    data = zcol.load(delayed=delayed, distributed=distributed)
    assert data is not None

    # Invalid update function
    with pytest.raises(TypeError, match='must be a callable'):
        zcol.update(1, new1.name, delayed=delayed)  # type: ignore[arg-type]

    def update_1(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 201.5}

    def update_2(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values // 5}

    zcol.update(update_1, new1.name, delayed=delayed)  # type: ignore[arg-type]
    zcol.update(update_2, new2.name, delayed=delayed)  # type: ignore[arg-type]

    if not (delayed and distributed):
        # The values loaded eagerly predate the updates.
        data = zcol.load(delayed=False, distributed=distributed)
        assert data is not None

    assert numpy.allclose(data.variables[new1.name].values,
                          data.variables['var1'].values * 201.5,
                          rtol=0)
    assert numpy.allclose(data.variables[new2.name].values,
                          data.variables['var1'].values // 5,
                          rtol=0)


@pytest.mark.parametrize(('fs', 'create_test_data'), FILE_SYSTEM_DATASET)
def test_update_with_immutable(
    fs,
    create_test_data,
    request,
) -> None:
    """Test the inclusion of immutable variables in the dataset provided to the
    update callback function."""
    tested_fs = request.getfixturevalue(fs)
    zds = _update_test_dataset(create_test_data,
                               delayed=False,
                               distributed=False)
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds, distributed=False)
    assert len(tuple(zcol.partitions())) == zds.variables['time'].shape[0]

    new_var = meta.VariableSchema(
        name='var4',
        dtype=numpy.dtype('int16'),
        dimensions=('num_lines', 'num_pixels'),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )
    new_im = meta.VariableSchema(
        name='var_immutable',
        dtype=numpy.dtype('int16'),
        dimensions=('num_pixels', ),
        fill_value=32267,
        attrs=(dataset.Attribute(name='attr', value=4), ),
    )

    zcol.add_variable(new_var, distributed=False)
    zcol.add_variable(variable=new_im, distributed=False)

    # Immutable variables are included
    def update_1(_zds):
        """Update function used for this test."""
        assert new_im.name in _zds.variables

        return {new_var.name: _zds.variables['var1'].values * 201.5}

    zcol.update(
        update_1,  # type: ignore[arg-type]
        delayed=False,
        distributed=False,
    )

    # Immutable variables are not included if excluded
    def update_2(_zds):
        """Update function used for this test."""
        assert new_im.name not in _zds.variables

        return {new_var.name: _zds.variables['var1'].values * 201.5}

    zcol.update(
        update_2,  # type: ignore[arg-type]
        selected_variables=['var1'],
        delayed=False,
        distributed=False,
    )


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_fillvalue(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    request,
) -> None:
    """Test the management of masked values."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    rows = None if delayed else _SHORT_FILLVALUE_ROWS
    zcol = create_test_collection(tested_fs,
                                  delayed=delayed,
                                  with_fillvalue=True,
                                  fillvalue_rows=rows)

    # Load the dataset written with masked values in the collection and
    # compare it to the original dataset.
    data = zcol.load(delayed=delayed)
    assert data is not None

    zds = next(create_test_dataset_with_fillvalue(delayed=delayed, rows=rows))
    assert data.variables['time'].shape[0] == (61 if delayed else 10)
    assert len(tuple(zcol.partitions())) == zds.variables['time'].shape[0]
    if rows is not None:
        months = zds.variables['time'].values.astype('datetime64[M]')
        assert months[0] != months[-1]

    values = data.variables['var1'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    mask = numpy.ma.getmaskarray(values)
    assert mask.any()
    assert (~mask).any()
    assert numpy.ma.allclose(zds.variables['var1'].values, values)

    values = data.variables['var2'].values
    assert isinstance(values, numpy.ma.MaskedArray)
    assert numpy.ma.allclose(zds.variables['var2'].values, values)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_degraded_tests(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the degraded functionality."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, distributed=distributed)

    fake_ds = next(create_test_dataset())
    fake_ds.variables['var3'] = fake_ds.variables['var1']
    fake_ds.variables['var3'].name = 'var3'

    with pytest.raises(ValueError, match='is unknown'):
        zcol.insert(fake_ds, distributed=distributed)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_with_missing_variable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test of the insertion of a dataset in which a variable is missing.

    This happens, for example, when a variable is not acquired, but
    created by an algorithm.
    """
    tested_fs = request.getfixturevalue(arg)
    zds = next(create_test_dataset_with_fillvalue())
    zcol = convenience.create_collection(
        axis='time',
        ds=zds,
        partition_handler=partitioning.Date(('time', ), 'M'),
        partition_base_dir=str(tested_fs.collection),
        filesystem=tested_fs.fs)
    zcol.insert(
        ds=zds,
        merge_callable=merging.merge_time_series,  # type: ignore[arg-type]
        distributed=distributed)

    zds = next(create_test_dataset_with_fillvalue())

    zds_v1 = zds.variables['var1']
    zds.drops_vars('var1')
    zcol.insert(zds, distributed=distributed)

    data = zcol.load()
    assert data is not None
    assert zds_v1.fill_value is not None
    assert numpy.ma.allequal(
        data.variables['var1'].values,
        numpy.ma.masked_equal(
            x=numpy.full(shape=zds_v1.shape,
                         fill_value=zds_v1.fill_value,
                         dtype=zds_v1.dtype),
            value=zds_v1.fill_value,
        ))


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_insert_missing_dimensions(fs, request) -> None:
    """Test the insertion of data with missing dimensions and variables."""
    tested_fs = request.getfixturevalue(fs)
    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='grid',
                data=numpy.arange(4 * 360 * 180).reshape(4, 360, 180),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection(axis='time',
                                 ds=zds_reference.metadata(),
                                 partition_handler=partitioning.Date(
                                     variables=('time', ), resolution='D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    # Adding a new dimension and a variable with this dimension so we
    # can test the insertion of a dataset not containing all dimensions
    new_dim = meta.Dimension(name='x', value=2, chunks=1)
    new_var = meta.VariableSchema(name='new_var',
                                  dtype=numpy.dtype('int8'),
                                  dimensions=(zcol.dimension, new_dim.name),
                                  fill_value=0)

    zcol.add_dimension(dimension=new_dim)
    zcol.add_variable(variable=new_var, distributed=False)

    zcol.insert(ds=zds_reference, distributed=False)

    data = zcol.load(distributed=False)

    assert data is not None
    assert set(data.variables) == {'time', 'grid', new_var.name}
    assert set(data.dimensions) == {'lat', 'lon', 'time', new_dim.name}

    #: pylint: disable=protected-access
    data_ins = zcol._set_ds_for_insertion(ds=data)
    assert set(data_ins.variables) == {'time', 'grid', new_var.name}
    assert set(data_ins.dimensions) == {'lat', 'lon', 'time', new_dim.name}


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_failed(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    request,
) -> None:
    """Test the insertion of a dataset in which the insertion failed."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zds = next(create_test_dataset(delayed=delayed))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    partitions = list(zcol.partitioning.split_dataset(zds, 'time'))

    # Create a file in the directory where the dataset should be written. This
    # should cause the insertion to fail.
    sep = zcol.fs.sep
    one_directory = sep.join((zcol.partition_properties.dir, ) +
                             partitions[0][0])
    zcol.fs.makedirs(sep.join(one_directory.split(sep)[:-1]))
    zcol.fs.touch(one_directory)

    with pytest.raises(execution.PartitionErrors):
        zcol.insert(zds, distributed=distributed)

    # Because the insert failed, the partition that was supposed to be created
    # was deleted.
    assert not zcol.fs.exists(one_directory)
    zcol.insert(zds, distributed=distributed)


def test_insert_invalid_dimensions(request, ) -> None:
    """Test the insertion of a dataset with invalid dimensions properties."""
    tested_fs = request.getfixturevalue('local_fs')
    delayed = False
    zds = next(create_test_dataset(delayed=delayed))
    zcol = collection.Collection('time',
                                 zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    # Swapping dimensions names to create invalid ones
    ds = zds.to_xarray().rename({
        'num_lines': 'num_pixels',
        'num_pixels': 'num_lines'
    })

    with pytest.raises(ValueError, match='Inserted dimension '):
        zcol.insert(ds)


def test_insert_invalid_variable(request, ) -> None:
    """Test the insertion of a dataset containing a variable with invalid
    properties."""
    tested_fs = request.getfixturevalue('local_fs')
    delayed = False
    zds = next(create_test_dataset(delayed=delayed))
    zds_x = copy.deepcopy(zds)

    # Changing variable dtype and fill value
    zds_x['var1'].fill_value = 10
    zds_x['var1'].array = zds_x['var1'].array.astype('int8')

    zcol = collection.Collection(axis='time',
                                 ds=zds_x.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    with pytest.raises(ValueError, match='has invalid dtype'):
        zcol.insert(zds)

    # Fixing dtype
    zds['var1'].array = zds['var1'].array.astype('int8')

    # A variable naming a value the collection does not declare is refused.
    zds['var1'].fill_value = 20

    with pytest.raises(ValueError, match='has invalid fill_value'):
        zcol.insert(zds)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_map_partition(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcol = create_test_collection(tested_fs, delayed=delayed)

    result = zcol.map(
        lambda x: x.variables['var1'].values * 2,  # type: ignore[arg-type]
        delayed=delayed)
    for item in result.compute():
        folder = zcol.fs.sep.join(
            (zcol.partition_properties.dir,
             zcol.partitioning.join(item[0], zcol.fs.sep)))
        zds = storage.open_zarr_group(folder, zcol.fs, delayed=False)
        assert numpy.allclose(item[1], zds.variables['var1'].values * 2)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
@pytest.mark.parametrize('distributed', [False, True])
def test_indexer(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    distributed,
    request,
) -> None:
    """Test the update of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcol = create_test_collection(tested_fs,
                                  delayed=delayed,
                                  distributed=distributed)

    indexers = zcol.map(lambda x: slice(  # type: ignore[arg-type]
        0, x.dimensions['num_lines'])).compute()
    zds1 = zcol.load(indexer=indexers,
                     delayed=delayed,
                     distributed=distributed)
    assert zds1 is not None
    zds2 = zcol.load(delayed=delayed, distributed=distributed)
    assert zds2 is not None

    assert numpy.allclose(zds1.variables['var1'].values,
                          zds2.variables['var1'].values)


def test_variables() -> None:
    """Test the listing of the variables in a collection."""
    fs = fsspec.filesystem('memory')
    zds = next(create_test_dataset(delayed=False))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir='/',
                                 filesystem=fs)
    variables = zcol.variables()
    assert isinstance(variables, tuple)
    assert len(variables) == 3
    assert variables[0].name == 'time'
    assert variables[1].name == 'var1'
    assert variables[2].name == 'var2'

    variables = zcol.variables(('time', ))
    assert len(variables) == 1
    assert variables[0].name == 'time'


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('arrays_type', ['dask_arrays', 'numpy_arrays'])
def test_map_overlap(
    dask_client,  # noqa: F811
    fs,
    arrays_type,
    request,
) -> None:
    """Test the map overlap method."""
    tested_fs = request.getfixturevalue(fs)
    delayed = request.getfixturevalue(arrays_type)
    zcol = create_test_collection(tested_fs, delayed=delayed)

    def func(zds: dataset.Dataset, partition_info: PartitionInfo):
        assert partition_info.dimension == 'num_lines'
        var = zds.variables['var1']
        slices = [slice(None)] * len(var.dimensions)
        slices[var.dimensions.index(
            partition_info.dimension)] = partition_info.indices
        return var.values[tuple(slices)] * 2

    result = zcol.map_overlap(
        func,  # type: ignore[arg-type]
        delayed=delayed,
        depth=1)

    for partition, data in result.compute():
        folder = zcol.fs.sep.join(
            (zcol.partition_properties.dir,
             zcol.partitioning.join(partition, zcol.fs.sep)))
        zds = storage.open_zarr_group(folder, zcol.fs, delayed=False)
        assert numpy.allclose(data, zds.variables['var1'].values * 2)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_immutable(
    dask_client,  # noqa: F811
    arg,
    distributed,
    request,
) -> None:
    """Test the insertion of a dataset with variables that are immutable
    relative to the partitioning."""
    tested_fs = request.getfixturevalue(arg)
    rng = numpy.random.default_rng(42)

    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='lon',
                data=numpy.arange(0, 360, 1),
                dimensions=('lon', ),
            ),
            dataset.Array(
                name='lat',
                data=numpy.arange(-90, 90, 1),
                dimensions=('lat', ),
            ),
            dataset.Array(
                name='grid',
                data=rng.random((4, 360, 180)),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection('time',
                                 zds_reference.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    assert zcol.have_immutable
    assert tested_fs.fs.exists(zcol.immutable_path)

    zcol.insert(zds_reference, distributed=distributed)

    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None

    assert numpy.all(
        zds.variables['grid'].values == zds_reference.variables['grid'].values)
    assert numpy.all(
        zds.variables['time'].values == zds_reference.variables['time'].values)
    assert numpy.all(
        zds.variables['lon'].values == zds_reference.variables['lon'].values)
    assert numpy.all(
        zds.variables['lat'].values == zds_reference.variables['lat'].values)

    def update(zds: dataset.Dataset, varname: str) -> dict[str, numpy.ndarray]:
        """Update function used for this test."""
        return {varname: zds.variables['grid'].values * -1}

    zcol.update(
        update,  # type: ignore[arg-type]
        delayed=False,
        distributed=distributed,
        varname='grid')
    zds = zcol.load(delayed=False, distributed=distributed)
    assert zds is not None

    assert numpy.all(zds.variables['grid'].values ==
                     zds_reference.variables['grid'].values * -1)
    assert numpy.all(
        zds.variables['time'].values == zds_reference.variables['time'].values)
    assert numpy.all(
        zds.variables['lon'].values == zds_reference.variables['lon'].values)
    assert numpy.all(
        zds.variables['lat'].values == zds_reference.variables['lat'].values)

    new_variable = meta.VariableSchema(
        'new_var',
        numpy.float64,
        dimensions=('time', 'lon', 'lat'),
        attrs=(meta.Attribute('units', 'm'), ),
    )
    zcol.add_variable(new_variable, distributed=distributed)
    zcol.update(
        update,  # type: ignore[arg-type]
        distributed=distributed,
        varname='new_var')
    zds = zcol.load(distributed=distributed)
    assert zds is not None
    assert numpy.all(zds.variables['new_var'].values ==
                     zds_reference.variables['grid'].values)

    new_variable = meta.VariableSchema(
        'new_var2',
        numpy.float64,
        dimensions=('another_dim', ),
        attrs=(meta.Attribute('units', 'm'), ),
    )
    with pytest.raises(
            ValueError,
            match=r'The new variable must use the dataset dimensions.'):
        zcol.add_variable(new_variable, distributed=distributed)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_insert_immutable_only(fs, request, caplog) -> None:
    """Test insertion of data only containing immutable variables."""
    tested_fs = request.getfixturevalue(fs)
    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='lat',
                data=numpy.arange(-90, 90, 1),
                dimensions=('lat', ),
            ),
            dataset.Array(
                name='grid',
                data=numpy.arange(4 * 360 * 180).reshape(4, 360, 180),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection(axis='time',
                                 ds=zds_reference.metadata(),
                                 partition_handler=partitioning.Date(
                                     variables=('time', ), resolution='D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    assert zcol.have_immutable
    assert zcol.load(distributed=False) is None

    caplog.set_level(logging.DEBUG)
    partitions = zcol.insert(ds=zds_reference.select_vars(names='lat'),
                             distributed=False)

    assert "Writing Zarr variable 'lat'" in caplog.text
    assert partitions == ()

    data = zcol.load(distributed=False)

    assert data is not None
    assert zcol.dimension not in data.variables
    assert numpy.array_equal(data['lat'].values, zds_reference['lat'].values)

    caplog.clear()
    zcol.insert(ds=zds_reference, distributed=False)

    assert "Writing Zarr variable 'lat'" not in caplog.text

    data = zcol.load(distributed=False)

    assert data is not None
    assert numpy.array_equal(data['time'].values, zds_reference['time'].values)
    assert numpy.array_equal(data['grid'].values, zds_reference['grid'].values)
    assert numpy.array_equal(data['lat'].values, zds_reference['lat'].values)


@pytest.mark.parametrize('arg', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_copy_collection(
        dask_client,  # noqa: F811
        arg,
        distributed,
        request,
        tmp_path) -> None:
    """Test the dropping of a dataset."""
    tested_fs = request.getfixturevalue(arg)
    zcol = create_test_collection(tested_fs, distributed=distributed)

    target = str(tmp_path / 'copy')
    zcopy = zcol.copy(target,
                      filesystem=fsspec.filesystem('file'),
                      distributed=distributed)

    ds_before_copy = zcol.load(distributed=distributed)
    ds_after_copy = zcopy.load(distributed=distributed)

    assert ds_before_copy is not None
    assert ds_after_copy is not None

    assert numpy.all(ds_before_copy.variables['var1'].values ==
                     ds_after_copy.variables['var1'].values)
    assert numpy.all(ds_before_copy.variables['var2'].values ==
                     ds_after_copy.variables['var2'].values)


def _insert(zds: dataset.Dataset, base_dir: str, lock_file: str,
            scheduler_file: str) -> None:
    client = dask.distributed.Client(scheduler_file=scheduler_file)
    zcol = collection.Collection.from_config(
        base_dir, mode='w', synchronizer=sync.ProcessSync(lock_file))
    zcol.insert(zds, merge_callable=merging.merge_time_series)
    client.close()


def test_concurrent_insert(
    dask_client,  # noqa: F811
    tmp_path,
) -> None:
    """Test the insertion of a dataset."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset(delayed=False))
    zds = datasets[0]
    lock_file = str(tmp_path / 'lock.lck')
    synchronizer = sync.ProcessSync(lock_file)
    base_dir = str(tmp_path / 'test')
    zcol = collection.Collection(
        axis='time',
        ds=zds.metadata(),
        partition_handler=partitioning.Date(('time', ), 'D'),
        partition_base_dir=base_dir,
        filesystem=fs,
        synchronizer=synchronizer,
    )

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=32, mp_context=multiprocessing.get_context('spawn'))
    futures = []

    assert zcol.is_locked() is False

    indices = list(numpy.arange(0, len(datasets)))
    rng = numpy.random.default_rng()
    rng.shuffle(indices)
    futures = [
        pool.submit(_insert, datasets[ix], base_dir, lock_file,
                    dask_client.scheduler_file) for ix in indices
    ]

    def update(zds: dataset.Dataset, shift: int = 3):
        """Update function used for this test."""
        return {'var2': zds.variables['var1'].values * -1 + shift}

    launch_update = True
    for item in concurrent.futures.as_completed(futures):
        assert item.exception() is None
        if launch_update:
            zcol.update(update)  # type: ignore[arg-type]
            launch_update = False

    data = zcol.load()
    assert data is not None
    values = data.variables['time'].values

    assert numpy.all(values == numpy.arange(START_DATE, END_DATE, DELTA))


def test_partition_modified(
        dask_client,  # noqa: F811
        tmp_path):
    """Test the loading of a variable that has been modified since its
    creation."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset())
    zds = datasets[0]
    base_dir = str(tmp_path / 'test')
    zcol = collection.Collection('time',
                                 zds.metadata(),
                                 partitioning.Date(('time', ), 'D'),
                                 base_dir,
                                 filesystem=fs)
    zcol.insert(zds)

    last_month = zcol.load(filters=lambda keys: keys['year'] == 2000 and keys[
        'month'] == 1 and keys['day'] == 16)
    assert last_month is not None

    zds = zcol.load()
    assert zds is not None

    def new_shape(
        var: variable.Variable,
        selected_dim: str,
        new_size: int,
    ) -> tuple[int, ...]:
        """Compute the new shape of a variable."""
        return tuple(
            new_size if dim == selected_dim else size
            for dim, size in zip(var.dimensions, var.shape, strict=True))

    dim, size = 'num_lines', last_month.dimensions['num_lines'] * 25
    last_month = dataset.Dataset(
        [
            variable.DelayedArray(
                name,
                numpy.resize(var.array.compute(), new_shape(var, dim, size)),
                var.dimensions,
                attrs=var.attrs,
                compressor=var.compressor,
                fill_value=var.fill_value,
                filters=var.filters,
            ) for name, var in last_month.variables.items()
        ],
        attrs=zds.attrs,
    )
    zcol.insert(last_month)

    with pytest.raises(RuntimeError, match='Try to re-load'):
        _ = zds['var2'].values

    with pytest.raises(RuntimeError, match='Try to re-load'):
        _ = zds['time'].values


def test_add_variable_leaves_a_vanished_partition(
        local_fs,  # noqa: F811
) -> None:
    """A partition the store no longer holds is not written to.

    The index names it, so add_variable reaches it. Writing there would
    build a partition holding the one variable.
    """
    zcol = create_test_collection(local_fs, distributed=False)
    partitions = tuple(zcol.partitions())
    vanished = partitions[1]
    local_fs.fs.rm(vanished, recursive=True)
    local_fs.fs.invalidate_cache()

    zcol.add_variable(meta.VariableSchema(name='var3',
                                          dtype='float64',
                                          dimensions=('num_lines', )),
                      distributed=False)

    assert not local_fs.fs.exists(vanished)
    assert 'var3' in zcol.metadata.variables
    assert local_fs.fs.exists(
        local_fs.fs.sep.join((partitions[0], 'var3', 'zarr.json')))


def test_validate_partitions_reports_a_vanished_partition(
        local_fs,  # noqa: F811
) -> None:
    """A partition the store no longer holds is reported, and unindexed."""
    zcol = create_test_collection(local_fs, distributed=False)
    partitions = tuple(zcol.partitions())
    vanished = partitions[1]
    local_fs.fs.rm(vanished, recursive=True)
    local_fs.fs.invalidate_cache()

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        invalid = zcol.validate_partitions(distributed=False, fix=True)

    assert invalid == (zcol._relative_path(vanished), )
    assert tuple(zcol.partitions()) == tuple(item for item in partitions
                                             if item != vanished)


@pytest.mark.parametrize('distributed', [False, True])
def test_invalid_partitions(
        dask_client,  # noqa: F811
        distributed,
        tmp_path) -> None:
    """Test the validate_partitions function."""
    fs = fsspec.filesystem('file')
    datasets = list(create_test_dataset())
    zds = datasets.pop(0)
    zds.concat(datasets, 'num_lines')
    base_dir = str(tmp_path / 'test')
    zcol = collection.Collection(
        axis='time',
        ds=zds.metadata(),
        partition_handler=partitioning.Date(variables=('time', ),
                                            resolution='D'),
        partition_base_dir=base_dir,
        filesystem=fs,
    )
    zcol.insert(zds)
    # A chunk whose values all equal the fill value is not written, so only
    # the partitions holding one can be corrupted.
    partitions = tuple(item for item in zcol.partitions()
                       if fs.exists(fs.sep.join((item, 'var2', 'c.0.0'))))
    rng = numpy.random.default_rng()
    choices = rng.choice(len(partitions), size=2, replace=False)
    for idx in choices:
        var2 = fs.sep.join((partitions[idx], 'var2', 'c.0.0'))
        with fs.open(var2, 'wb') as file:
            file.write(b'invalid')

    with pytest.raises(ValueError, match='When changing to a larger dtype'):
        _ = zcol.load(delayed=False, distributed=distributed)

    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        invalid_partitions = zcol.validate_partitions(distributed=distributed)

    assert len(invalid_partitions) == 2
    assert sorted(invalid_partitions) == sorted(
        zcol._relative_path(partitions[ix]) for ix in choices)
    with pytest.warns(RuntimeWarning, match='Invalid partition'):
        zcol.validate_partitions(fix=True, distributed=distributed)

    assert zcol.load() is not None

    # Filters excludes all partitions
    assert not zcol.validate_partitions(filters='day==36',
                                        distributed=distributed)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('distributed', [False, True])
def test_insert_with_chunks(
    dask_client,  # noqa: F811
    fs,
    distributed,
    request,
    tmp_path,
) -> None:
    """Test the insertion of a dataset."""
    tested_fs = request.getfixturevalue(fs)
    datasets = list(create_test_dataset(delayed=False))[:2]

    ds_meta = datasets[0].metadata()
    chunk_size = 5
    ds_meta.dimensions['num_pixels'].chunks = chunk_size

    zcol = collection.Collection('time',
                                 ds_meta,
                                 partitioning.Date(('time', ), 'M'),
                                 str(tested_fs.collection),
                                 filesystem=tested_fs.fs,
                                 synchronizer=sync.ProcessSync(
                                     str(tmp_path / 'lock.lck')))

    # First insertion
    zcol.insert(datasets[0],
                merge_callable=merging.merge_time_series,
                distributed=distributed)

    # Not setting distributed to False when loading otherwise we won't have any chunk
    data = zcol.load()

    assert data is not None
    assert data.variables['var1'].data.chunksize[1] == chunk_size
    assert data.variables['var2'].data.chunksize[1] == chunk_size

    # Insertion with merge
    zcol.insert(datasets[1],
                merge_callable=merging.merge_time_series,
                distributed=distributed)
    data = zcol.load()

    assert data is not None
    # Insertion properties are kept
    assert data.variables['var1'].data.chunksize[1] == chunk_size
    assert data.variables['var2'].data.chunksize[1] == chunk_size


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_with_empty_collection(fs, request):
    """Test the behavior when working with an empty collection."""
    tested_fs = request.getfixturevalue(fs)
    zds_reference = dataset.Dataset(
        variables=[
            dataset.Array(
                name='time',
                data=numpy.arange(numpy.datetime64('2000-01-01'),
                                  numpy.datetime64('2000-01-05'),
                                  numpy.timedelta64(1, 'D')),
                dimensions=('time', ),
            ),
            dataset.Array(
                name='grid',
                data=numpy.arange(4 * 360 * 180).reshape(4, 360, 180),
                dimensions=('time', 'lon', 'lat'),
            ),
        ],
        attrs=(dataset.Attribute('history', 'Created for testing'), ),
    )
    zcol = collection.Collection(axis='time',
                                 ds=zds_reference.metadata(),
                                 partition_handler=partitioning.Date(
                                     variables=('time', ), resolution='D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs,
                                 mode='w')

    var = meta.VariableSchema(name='var2',
                              dtype=numpy.float64,
                              dimensions=('time', 'lat'))

    zcol.add_variable(variable=var, distributed=False)

    assert var.name in zcol.metadata.variables

    def update(zds, varname):
        """Update function used for this test."""
        return {varname: zds.variables['var1'].values * 0 + 5}

    with pytest.warns(Warning, match='function is not applied'):
        zcol.update(update, varname=var.name, distributed=False)


def test_insert_writes_the_declared_encoding(
    dask_client,  # noqa: F811
    request,
) -> None:
    """A partition is written with the codecs the collection declares.

    The compressor and the filters belong to the collection, so every
    partition is stored with them.
    """
    tested_fs = request.getfixturevalue('local_fs')
    zds = next(create_test_dataset(delayed=False))
    declared = copy.deepcopy(zds)
    declared['var1'].compressor = numcodecs.Blosc(cname='zstd', clevel=5)
    declared['var1'].filters = (numcodecs.FixedScaleOffset(0, 10, 'i8',
                                                           'i2'), )

    zcol = collection.Collection(axis='time',
                                 ds=declared.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds)

    partition = next(iter(zcol.partitions()))
    path = tested_fs.fs.sep.join((partition, 'var1', 'zarr.json'))
    with tested_fs.fs.open(path) as stream:
        codecs = [item['name'] for item in json.load(stream)['codecs']]

    assert 'blosc' in codecs
    assert 'numcodecs.fixedscaleoffset' in codecs


def test_insert_takes_the_declared_fill_value(
    dask_client,  # noqa: F811
    request,
) -> None:
    """A variable naming no fill value takes the declared one."""
    tested_fs = request.getfixturevalue('local_fs')
    zds = next(create_test_dataset(delayed=False))
    declared = copy.deepcopy(zds)
    declared['var1'].fill_value = 10

    zcol = collection.Collection(axis='time',
                                 ds=declared.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    assert zds['var1'].fill_value is None
    zcol.insert(zds)

    loaded = zcol.load()
    assert loaded is not None
    assert loaded['var1'].fill_value == 10


def test_load_a_selection_that_matches_no_partition(
    dask_client,  # noqa: F811
    request,
) -> None:
    """Naming the partitions to read and matching none of them reads nothing.

    The collection holds an immutable variable.
    """
    tested_fs = request.getfixturevalue('local_fs')
    reference = next(create_test_dataset(delayed=False))
    zds = dataset.Dataset(variables=[
        *reference.variables.values(),
        variable.Array(name='lon',
                       data=numpy.arange(25, dtype='float32'),
                       dimensions=('num_pixels', )),
    ])

    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'M'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)
    zcol.insert(zds)
    assert zcol.have_immutable

    assert zcol.load(filters='year == 1800') is None
    assert zcol.load(selected_partitions=['year=1800/month=01']) is None

    loaded = zcol.load()
    assert loaded is not None
    assert 'lon' in loaded.variables


def test_insert_returns_the_paths_a_selection_takes(request) -> None:
    """What an insertion wrote names the partitions it wrote.

    A reader of the result feeds it back to ``selected_partitions``, which
    reads paths relative to the collection.
    """
    tested_fs = request.getfixturevalue('local_fs')
    zds = next(create_test_dataset(delayed=False))
    zcol = collection.Collection(axis='time',
                                 ds=zds.metadata(),
                                 partition_handler=partitioning.Date(
                                     ('time', ), 'D'),
                                 partition_base_dir=str(tested_fs.collection),
                                 filesystem=tested_fs.fs)

    written = zcol.insert(zds, distributed=False)

    assert written
    assert set(written) == set(zcol.partitions(relative=True))

    read = zcol.load(selected_partitions=written[:1],
                     delayed=False,
                     distributed=False)
    assert read is not None
    assert read.variables['time'].values.size < zds.variables[
        'time'].values.size
