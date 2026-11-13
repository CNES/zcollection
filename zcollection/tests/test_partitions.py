# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the index of the partitions
===================================

A listing that reads the index sees what the store holds only as long as
the two agree. These tests pin what writes the index, and what happens
when it is gone.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import subprocess
import sys
from unittest import mock

import fsspec
import numpy
import pytest

from .. import collection, execution, meta, partitioning, partitions
from ..convenience import create_view, open_collection
from ..fs_utils import join_path
from ..view import ViewReference
from .data import create_test_collection, create_test_dataset
# pylint: disable=unused-import # Need to import for fixtures
from .fs import local_fs  # noqa: F401

# pylint: enable=unused-import

#: Each partitioning, with the path of one of its partitions.
PARTITIONINGS = [
    (partitioning.Date(('time', ), 'D'), 'year=2000/month=01/day=05'),
    (partitioning.Sequence(('cycle', 'pass')), 'cycle=2/pass=7'),
    (partitioning.GroupedSequence(variables=('cycle', ), size=5), 'cycle=10'),
]


@pytest.mark.parametrize('keys', [
    [[]],
    [[2000, 1]],
    [[2000, 1, 1, 0]],
    [[2000, 1, '1']],
    [[2000, 1, True]],
    [[2000, 1, 1.0]],
    [[2000, 1, None]],
    [[2000, 1, 1], [2000, 1, 1]],
    [[2000, 1, 2], [2000, 1, 1]],
    {
        'key': [2000, 1, 1]
    },
])
def test_invalid_keys_are_rejected(keys) -> None:
    """Reject keys with invalid values, lengths, order or duplicates."""
    fs = fsspec.filesystem('memory')
    index = partitions.PartitionIndex(fs,
                                      '/invalid-index',
                                      PARTITIONINGS[0][0],
                                      writable=False)
    fs.pipe_file(
        index.path,
        gzip.compress(
            json.dumps({
                'version': partitions.VERSION,
                'state': 'ready',
                'names': ['year', 'month', 'day'],
                'partitions': keys,
            }).encode()))

    assert index._read() is None


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_a_corrupt_index_cannot_drop_the_collection(fs, request) -> None:
    """Dropping partitions through a corrupt index preserves the configuration."""
    tested_fs = request.getfixturevalue(fs)
    zcol = create_test_collection(tested_fs, distributed=False)
    expected = set(zcol.partitions(relative=True))
    tested_fs.fs.pipe_file(
        zcol._index.path,
        gzip.compress(
            json.dumps({
                'version': partitions.VERSION,
                'state': 'ready',
                'names': ['year', 'month', 'day'],
                'partitions': [[]],
            }).encode()))

    dropped = zcol.drop_partitions(distributed=False)

    assert set(dropped) == expected
    assert tested_fs.fs.exists(
        join_path(str(tested_fs.collection), zcol.CONFIG))
    assert tuple(zcol.partitions()) == ()


def count_listings(monkeypatch) -> dict[str, int]:
    """Count the directory listings a local file system serves."""
    counted = {'ls': 0}
    listing = fsspec.implementations.local.LocalFileSystem.ls

    def ls(self, *args, **kwargs):
        counted['ls'] += 1
        return listing(self, *args, **kwargs)

    monkeypatch.setattr(fsspec.implementations.local.LocalFileSystem, 'ls', ls)
    return counted


@pytest.mark.parametrize(('handler', 'path'), PARTITIONINGS)
def test_a_path_survives_the_index(handler, path) -> None:
    """A path parsed into a key comes back as the same path."""
    index = partitions.PartitionIndex(fsspec.filesystem('file'),
                                      '/collection',
                                      handler,
                                      writable=False)
    key, = index.keys_of((path, ))

    assert index.path_of(key) == path


def test_an_insertion_writes_the_index(local_fs) -> None:  # noqa: F811
    """An insertion names in the index the partitions it wrote."""
    zcol = create_test_collection(local_fs, distributed=False)
    written = join_path(str(local_fs.collection), partitions.FILENAME)

    assert local_fs.fs.exists(written)

    data = json.loads(gzip.decompress(local_fs.fs.cat_file(written)))

    assert data['names'] == ['year', 'month', 'day']
    assert len(data['partitions']) == len(tuple(zcol.partitions()))


def test_a_listing_reads_the_index(
        local_fs,  # noqa: F811
        monkeypatch) -> None:
    """A listing that reads the index walks no directory."""
    zcol = create_test_collection(local_fs, distributed=False)
    expected = tuple(zcol.partitions())
    counted = count_listings(monkeypatch)

    assert tuple(zcol.partitions()) == expected
    assert counted['ls'] == 0


@pytest.mark.parametrize('written', [
    None,
    b'not an index',
    gzip.compress(b'{}')[:10],
    b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03' + b'\xff' * 12,
    gzip.compress(b'{"version": 1, "names": ["cycle"], "partitions": []}'),
    gzip.compress(b'{"version": 0, "names": ["year", "month", "day"], '
                  b'"partitions": []}'),
])
def test_an_unusable_index_is_walked_without_writing(
        local_fs,  # noqa: F811
        written) -> None:
    """An unusable index triggers a directory walk without writing the index."""
    zcol = create_test_collection(local_fs, distributed=False)
    expected = tuple(zcol.partitions())
    path = join_path(str(local_fs.collection), partitions.FILENAME)

    if written is None:
        local_fs.fs.rm(path)
    else:
        local_fs.fs.pipe_file(path, written)

    assert tuple(zcol.partitions()) == expected
    if written is None:
        assert not local_fs.fs.exists(path)
    else:
        assert local_fs.fs.cat_file(path) == written


def test_a_read_only_collection_lists_without_writing(
        local_fs,  # noqa: F811
        caplog) -> None:
    """A collection that may not be written is listed without an index."""
    zcol = create_test_collection(local_fs, distributed=False)
    expected = tuple(zcol.partitions())
    path = join_path(str(local_fs.collection), partitions.FILENAME)
    local_fs.fs.rm(path)

    reader = open_collection(str(local_fs.collection),
                             mode='r',
                             filesystem=local_fs.fs)

    assert tuple(reader.partitions()) == expected
    assert not local_fs.fs.exists(path)
    assert 'listed from the store' in caplog.text


def test_a_dropped_partition_leaves_the_index(local_fs) -> None:  # noqa: F811
    """Dropping a partition takes it out of the index."""
    zcol = create_test_collection(local_fs, distributed=False)
    before = tuple(zcol.partitions())

    dropped = tuple(zcol.drop_partitions(filters='day == 1',
                                         distributed=False))

    assert len(dropped) > 0
    assert len(tuple(zcol.partitions())) == len(before) - len(dropped)


def test_reindex_reads_the_store(local_fs) -> None:  # noqa: F811
    """Indexing again names what the store holds, and nothing else."""
    zcol = create_test_collection(local_fs, distributed=False)
    before = tuple(zcol.partitions())
    local_fs.fs.rm(before[0], recursive=True)

    assert len(tuple(zcol.partitions())) == len(before)

    after = zcol.reindex()

    assert len(after) == len(before) - 1
    assert len(tuple(zcol.partitions())) == len(before) - 1


def test_an_inserted_partition_joins_the_index(local_fs) -> None:  # noqa: F811
    """An insertion keeps the index ordered, and names no partition twice."""
    zcol = create_test_collection(local_fs, distributed=False)
    datasets = create_test_dataset(delayed=False)
    next(datasets)
    zcol.insert(next(datasets), distributed=False)

    listed = tuple(zcol.partitions(relative=True))

    assert listed == tuple(sorted(
        listed, key=zcol.partitioning.parse))  # type: ignore[arg-type]
    assert len(set(listed)) == len(listed)


def test_a_read_skips_a_vanished_partition(
        local_fs,  # noqa: F811
        caplog) -> None:
    """Loading and mapping skip an absent partition and log a warning."""
    zcol = create_test_collection(local_fs, distributed=False)
    partitions = tuple(zcol.partitions())
    whole = zcol.load(distributed=False)
    assert whole is not None

    local_fs.fs.rm(partitions[1], recursive=True)
    local_fs.fs.invalidate_cache()

    partial = zcol.load(distributed=False)

    assert partial is not None
    assert len(partial['time'].values) < len(whole['time'].values)
    assert len(zcol.map(lambda zds: 1).compute()) == len(partitions) - 1
    assert 'holds no partition' in caplog.text


def test_a_read_raises_the_errors_of_its_caller(
        local_fs,  # noqa: F811
) -> None:
    """Mapping propagates FileNotFoundError raised by the supplied function."""
    zcol = create_test_collection(local_fs, distributed=False)

    def absent_elsewhere(zds) -> int:
        """Read a file that has nothing to do with the collection."""
        with open('/a-file-that-is-not-there', 'rb'):
            pass
        return 1

    with pytest.raises(FileNotFoundError):
        zcol.map(absent_elsewhere).compute()


def test_an_overlap_skips_the_vanished_neighbour_alone(
        local_fs,  # noqa: F811
) -> None:
    """An absent neighbour does not prevent mapping the current partition."""
    zcol = create_test_collection(local_fs, distributed=False)
    partitions = tuple(zcol.partitions())
    local_fs.fs.rm(partitions[3], recursive=True)
    local_fs.fs.invalidate_cache()

    mapped = zcol.map_overlap(lambda zds: 1, depth=1).compute()

    assert len(mapped) == len(partitions) - 1


def test_a_read_through_an_indexer_reads_nothing(
        local_fs,  # noqa: F811
) -> None:
    """An indexer naming only a vanished partition reads nothing."""
    zcol = create_test_collection(local_fs, distributed=False)
    vanished = tuple(zcol.partitions())[3]
    key = zcol.partitioning.parse(vanished)
    local_fs.fs.rm(vanished, recursive=True)
    local_fs.fs.invalidate_cache()

    assert zcol.load(indexer=[(key, slice(0, 1))], distributed=False) is None


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_sync_checks_existing_data_outside_the_index(fs, request) -> None:
    """Synchronizing an unindexed partition checks its existing axis."""
    tested_fs = request.getfixturevalue(fs)
    zcol = create_test_collection(tested_fs, delayed=False, distributed=False)
    zview = create_view(str(tested_fs.view),
                        ViewReference(str(tested_fs.collection), tested_fs.fs),
                        filesystem=tested_fs.fs,
                        distributed=False)
    zview.add_variable(meta.VariableSchema('derived',
                                           'int64',
                                           dimensions=('num_lines', ),
                                           fill_value=-1),
                       distributed=False)

    def fill_derived(zds, *args, **kwargs):
        """Fill the derived variable."""
        return {
            'derived': numpy.full(zds.dimensions['num_lines'],
                                  42,
                                  dtype='int64')
        }

    def replace_partition(existing_ds, inserted_ds, axis, partitioning_dim,
                          **kwargs):
        """Replace the measurements of an existing partition."""
        return inserted_ds

    zview.update(fill_derived, distributed=False)
    zview._index.write(zview._index.keys()[1:])
    replacement = zcol.load(filters='day == 1', distributed=False)
    assert replacement is not None
    replacement['time'].array[:] += numpy.timedelta64(12, 'h')
    zcol.insert(replacement,
                merge_callable=replace_partition,
                distributed=False)

    zview.sync(filters='day == 1', distributed=False)
    loaded = zview.load(filters='day == 1', distributed=False)

    assert loaded is not None
    numpy.testing.assert_array_equal(loaded['derived'].array, [-1])
    assert zview.is_synced(distributed=False)


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_sync_creates_variables_when_extending_a_view(fs, request) -> None:
    """Extending a view creates its variables in every added partition."""
    tested_fs = request.getfixturevalue(fs)
    create_test_collection(tested_fs, delayed=False, distributed=False)
    zview = create_view(str(tested_fs.view),
                        ViewReference(str(tested_fs.collection), tested_fs.fs),
                        filters='day == 1',
                        filesystem=tested_fs.fs,
                        distributed=False)
    zview.add_variable(meta.VariableSchema('derived',
                                           'int64',
                                           dimensions=('num_lines', ),
                                           fill_value=-1),
                       distributed=False)

    zview.sync(filters='day == 4', distributed=False)
    loaded = zview.load(distributed=False)

    assert loaded is not None
    numpy.testing.assert_array_equal(loaded['derived'].array, [-1, -1])


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_an_interrupted_insertion_keeps_written_partitions_visible(
        fs, request) -> None:
    """A failed insertion leaves its completed partitions visible."""
    tested_fs = request.getfixturevalue(fs)
    zcol = create_test_collection(tested_fs, delayed=False, distributed=False)
    datasets = create_test_dataset(delayed=False)
    next(datasets)
    ds = next(datasets)
    insert = collection._insert
    count = 0

    def fail_second(args, /, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('injected partition failure')
        return insert(args, **kwargs)

    with mock.patch.object(collection, '_insert', fail_second):
        with pytest.raises(execution.PartitionErrors,
                           match='injected partition failure'):
            zcol.insert(ds, distributed=False)

    reopened = open_collection(str(tested_fs.collection),
                               filesystem=tested_fs.fs)
    assert len(tuple(reopened.partitions())) == 7
    assert zcol._index._read() == zcol._index.walk()


def test_a_failed_reindex_after_a_failed_insertion_leaves_the_index_marked(
        local_fs,  # noqa: F811
        caplog) -> None:
    """A failed insertion whose reindex also fails leaves the index marked."""
    zcol = create_test_collection(local_fs, delayed=False, distributed=False)
    datasets = create_test_dataset(delayed=False)
    next(datasets)

    def fail(args, /, **kwargs):
        raise OSError('injected partition failure')

    def fail_walk(index):
        raise OSError('injected walk failure')

    with mock.patch.object(collection, '_insert', fail), \
            mock.patch.object(partitions.PartitionIndex, 'walk', fail_walk):
        with pytest.raises(execution.PartitionErrors,
                           match='injected partition failure'):
            zcol.insert(next(datasets), distributed=False)

    stored = json.loads(gzip.decompress(local_fs.fs.cat_file(
        zcol._index.path)))
    assert stored['state'] == 'updating'
    assert 'could not be rebuilt' in caplog.text


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_sync_retries_an_interrupted_index_publication(fs, request) -> None:
    """Retrying sync publishes partitions completed before its failure."""
    tested_fs = request.getfixturevalue(fs)
    zcol = create_test_collection(tested_fs, delayed=False, distributed=False)
    zview = create_view(str(tested_fs.view),
                        ViewReference(str(tested_fs.collection), tested_fs.fs),
                        filesystem=tested_fs.fs,
                        distributed=False)
    datasets = create_test_dataset(delayed=False)
    next(datasets)
    zcol.insert(next(datasets), distributed=False)
    write = partitions.PartitionIndex.write

    def fail_view(index, keys, *, updating=False):
        if index.path == zview._index.path and not updating:
            raise OSError('injected publication failure')
        return write(index, keys, updating=updating)

    with mock.patch.object(partitions.PartitionIndex, 'write', fail_view):
        with pytest.raises(OSError, match='injected publication failure'):
            zview.sync(distributed=False)

    zview.sync(distributed=False)

    assert len(tuple(zview.partitions())) == 11
    assert zview.is_synced(distributed=False)


def test_a_listing_cannot_overwrite_a_newer_index(
        local_fs,  # noqa: F811
        monkeypatch) -> None:
    """A listing concurrent with sync leaves the published index intact."""
    from ..convenience import open_view

    zcol = create_test_collection(local_fs, delayed=False, distributed=False)
    writer = create_view(str(local_fs.view),
                         ViewReference(str(local_fs.collection), local_fs.fs),
                         filesystem=local_fs.fs,
                         distributed=False)
    reader = open_view(str(local_fs.view), filesystem=local_fs.fs)
    local_fs.fs.rm(reader._index.path)
    walk = partitions.PartitionIndex.walk
    published = None

    def walk_then_sync(index):
        nonlocal published
        keys = walk(index)
        if index is reader._index:
            datasets = create_test_dataset(delayed=False)
            next(datasets)
            zcol.insert(next(datasets), distributed=False)
            writer.sync(distributed=False)
            published = local_fs.fs.cat_file(writer._index.path)
        return keys

    monkeypatch.setattr(partitions.PartitionIndex, 'walk', walk_then_sync)
    tuple(reader.partitions())

    assert local_fs.fs.cat_file(writer._index.path) == published
    assert len(tuple(writer.partitions())) == 11


def test_a_copy_writes_the_index_of_its_partitions(
        local_fs,  # noqa: F811
        caplog) -> None:
    """A copy indexes the partitions it copies."""
    zcol = create_test_collection(local_fs, distributed=False)
    selected = tuple(zcol.partitions(relative=True))[:3]
    target = local_fs.collection.parent / 'copy'

    copied = zcol.copy(str(target),
                       filters=lambda keys: zcol._index.path_of(
                           tuple(keys.values())) in selected,
                       distributed=False)

    assert tuple(copied.partitions(relative=True)) == selected
    assert copied._index._read() == copied._index.walk()
    assert 'listed from the store' not in caplog.text


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
@pytest.mark.parametrize('missing_all', [False, True])
def test_a_copy_skips_a_missing_partition(fs, missing_all, request) -> None:
    """A copy remains readable when an indexed source partition is absent."""
    tested_fs = request.getfixturevalue(fs)
    zcol = create_test_collection(tested_fs, delayed=False, distributed=False)
    paths = tuple(zcol.partitions())
    for path in paths if missing_all else paths[1:2]:
        tested_fs.fs.rm(path, recursive=True)
    tested_fs.fs.invalidate_cache()

    copied = zcol.copy(str(tested_fs.collection) + '-copy',
                       filesystem=tested_fs.fs,
                       distributed=False)
    loaded = copied.load(distributed=False)

    if missing_all:
        assert loaded is None
    else:
        assert loaded is not None
        assert len(loaded['time'].array) == 5
    assert len(tuple(copied.partitions())) == (0 if missing_all else 5)
    assert copied.validate_partitions(distributed=False) == ()


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_selecting_variables_preserves_the_rows_of_a_view(fs, request) -> None:
    """Selecting reference variables still skips an absent view partition."""
    tested_fs = request.getfixturevalue(fs)
    create_test_collection(tested_fs, delayed=False, distributed=False)
    zview = create_view(str(tested_fs.view),
                        ViewReference(str(tested_fs.collection), tested_fs.fs),
                        filesystem=tested_fs.fs,
                        distributed=False)
    zview.add_variable(meta.VariableSchema('derived',
                                           'int64',
                                           dimensions=('num_lines', )),
                       distributed=False)
    tested_fs.fs.rm(tuple(zview.partitions())[1], recursive=True)
    tested_fs.fs.invalidate_cache()

    all_variables = zview.load(distributed=False)
    time_only = zview.load(selected_variables=['time'], distributed=False)

    assert all_variables is not None
    assert time_only is not None
    numpy.testing.assert_array_equal(all_variables['time'].array,
                                     time_only['time'].array)


def test_an_insertion_survives_process_exit(local_fs) -> None:  # noqa: F811
    """A killed writer leaves completed partitions visible after reopening."""
    create_test_collection(local_fs, delayed=False, distributed=False)
    script = '''
import os
import sys
from zcollection import collection, open_collection
from zcollection.tests.data import create_test_dataset

insert = collection._insert
count = 0
def exit_second(args, /, **kwargs):
    global count
    count += 1
    if count == 2:
        os._exit(9)
    return insert(args, **kwargs)

collection._insert = exit_second
datasets = create_test_dataset(delayed=False)
next(datasets)
zcol = open_collection(sys.argv[1], mode='w')
zcol.insert(next(datasets), distributed=False)
'''
    process = subprocess.run(
        [sys.executable, '-c', script,
         str(local_fs.collection)],
        cwd=pathlib.Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        timeout=30)

    assert process.returncode == 9, process.stderr.decode()
    reopened = open_collection(str(local_fs.collection))
    loaded = reopened.load(distributed=False)

    assert loaded is not None
    assert len(loaded['time'].array) == 7


def test_a_view_keeps_the_union_of_its_selections(
        local_fs) -> None:  # noqa: F811
    """A view index preserves partitions from successive filter selections."""
    create_test_collection(local_fs, delayed=False, distributed=False)
    zview = create_view(str(local_fs.view),
                        ViewReference(str(local_fs.collection), local_fs.fs),
                        filesystem=local_fs.fs,
                        filters='day <= 7',
                        distributed=False)

    zview.sync(filters='day >= 13', distributed=False)

    days = [
        dict(zview.view_ref.partitioning.parse(path))['day']
        for path in zview.partitions()
    ]
    assert days == [1, 4, 7, 13, 16]


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_a_view_listing_only_reads_its_two_indexes(fs, request,
                                                   monkeypatch) -> None:
    """A ready view index requires no directory or existence checks."""
    tested_fs = request.getfixturevalue(fs)
    create_test_collection(tested_fs, delayed=False, distributed=False)
    zview = create_view(str(tested_fs.view),
                        ViewReference(str(tested_fs.collection), tested_fs.fs),
                        filesystem=tested_fs.fs,
                        distributed=False)
    reads = []

    def refuse(*args, **kwargs):
        raise AssertionError('unexpected directory or existence check')

    def record_read(original):

        def read(path, *args, **kwargs):
            reads.append(path)
            return original(path, *args, **kwargs)

        return read

    for filesystem in {zview.fs, zview.view_ref.fs}:
        monkeypatch.setattr(filesystem, 'cat_file',
                            record_read(filesystem.cat_file))
        monkeypatch.setattr(filesystem, 'ls', refuse)
        monkeypatch.setattr(filesystem, 'exists', refuse)

    assert len(tuple(zview.partitions())) == 6
    assert sorted(reads) == sorted(
        [zview._index.path, zview.view_ref._index.path])


def test_a_new_collection_has_an_empty_index(local_fs) -> None:  # noqa: F811
    """Creating a collection writes its empty partition index."""
    ds = next(create_test_dataset(delayed=False))
    zcol = collection.Collection('time',
                                 ds.metadata(),
                                 PARTITIONINGS[0][0],
                                 str(local_fs.collection),
                                 filesystem=local_fs.fs)

    assert local_fs.fs.exists(zcol._index.path)
    assert zcol._index._read() == ()


def test_opening_for_writing_leaves_an_existing_index_unchanged(
        local_fs,  # noqa: F811
        monkeypatch) -> None:
    """Opening for writing preserves the existing partition index."""
    zcol = create_test_collection(local_fs, delayed=False, distributed=False)
    raw = local_fs.fs.cat_file(zcol._index.path)

    def refuse(*args, **kwargs):
        raise AssertionError('opening must not publish an index')

    monkeypatch.setattr(partitions.PartitionIndex, 'write', refuse)
    reopened = open_collection(str(local_fs.collection),
                               mode='w',
                               filesystem=local_fs.fs)

    assert local_fs.fs.cat_file(reopened._index.path) == raw
    assert len(tuple(reopened.partitions())) == 6


@pytest.mark.parametrize('fs', ['local_fs', 's3_fs'])
def test_reindex_refreshes_cached_directory_listings(fs, request) -> None:
    """Reindexing observes changes made through another filesystem instance."""
    tested_fs = request.getfixturevalue(fs)
    zcol = create_test_collection(tested_fs, delayed=False, distributed=False)
    zcol._index.walk()
    external = type(tested_fs.fs)(**tested_fs.fs.storage_options,
                                  skip_instance_cache=True)
    external.rm(next(iter(zcol.partitions())), recursive=True)

    assert len(zcol.reindex()) == 5


def test_an_insertion_keeps_the_index_without_a_walk(
        local_fs,  # noqa: F811
        monkeypatch) -> None:
    """A successful insertion updates a ready index without listing directories."""
    zcol = create_test_collection(local_fs, delayed=False, distributed=False)
    datasets = create_test_dataset(delayed=False)
    next(datasets)

    def refuse(*args, **kwargs):
        raise AssertionError('unexpected index walk')

    monkeypatch.setattr(partitions.PartitionIndex, 'walk', refuse)

    zcol.insert(next(datasets), distributed=False)

    assert len(tuple(zcol.partitions())) == 11
