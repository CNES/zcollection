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

import fsspec
import pytest

from .. import partitioning, partitions
from ..convenience import open_collection
from ..fs_utils import join_path
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
    gzip.compress(b'{"version": 1, "names": ["cycle"], "partitions": []}'),
    gzip.compress(b'{"version": 0, "names": ["year", "month", "day"], '
                  b'"partitions": []}'),
])
def test_an_unusable_index_is_replaced(
        local_fs,  # noqa: F811
        written) -> None:
    """An index the collection cannot use is rebuilt and written back."""
    zcol = create_test_collection(local_fs, distributed=False)
    expected = tuple(zcol.partitions())
    path = join_path(str(local_fs.collection), partitions.FILENAME)

    if written is None:
        local_fs.fs.rm(path)
    else:
        local_fs.fs.pipe_file(path, written)

    assert tuple(zcol.partitions()) == expected
    assert local_fs.fs.exists(path)


def test_a_collection_read_only_holds_the_walk(
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
    assert 'held for this session alone' in caplog.text


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
