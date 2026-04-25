"""End-to-end Phase 1 walkthrough for ``zcollection``.

Builds a ~100 MB float32 dataset on a LocalStore, partitions it, reopens the
collection from disk, queries with a filter, and asserts bit-exact equality.

Run with::

    python -m examples.walkthrough /tmp/zc3-walkthrough
"""

from pathlib import Path
import shutil
import sys
import tempfile

import numpy

import zcollection as zc
from zcollection.partitioning import Sequence


def build_schema() -> zc.DatasetSchema:
    return (zc.Schema().with_dimension(
        'time', size=None, chunks=4096).with_dimension(
            'x_ac', size=240,
            chunks=240).with_variable('time',
                                      dtype='int64',
                                      dimensions=('time', )).with_variable(
                                          'partition',
                                          dtype='int64',
                                          dimensions=('time', )).with_variable(
                                              'ssh',
                                              dtype='float32',
                                              dimensions=('time', 'x_ac'),
                                              fill_value=numpy.float32('nan'),
                                          ).build())


def build_dataset(schema: zc.DatasetSchema,
                  n_partitions: int = 4,
                  rows_per_partition: int = 25_000) -> zc.Dataset:
    rng = numpy.random.default_rng(42)
    n = n_partitions * rows_per_partition
    time = numpy.arange(n, dtype='int64')
    partition = numpy.repeat(numpy.arange(n_partitions, dtype='int64'),
                             rows_per_partition)
    ssh = rng.standard_normal(size=(n, 240), dtype='float32')
    return zc.Dataset(
        schema=schema,
        variables={
            'time': zc.Variable(schema.variables['time'], time),
            'partition': zc.Variable(schema.variables['partition'], partition),
            'ssh': zc.Variable(schema.variables['ssh'], ssh),
        },
    )


def main(target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)

    schema = build_schema()
    ds = build_dataset(schema)
    bytes_written = ds['ssh'].to_numpy().nbytes
    print(f"dataset: {ds}  ({bytes_written/1e6:.1f} MB ssh)")

    col = zc.create_collection(
        f"file://{target}",
        schema=schema,
        axis='time',
        partitioning=Sequence(('partition', ), dimension='time'),
    )
    written = col.insert(ds)
    print(f"wrote {len(written)} partitions: {written}")

    reopened = zc.open_collection(f"file://{target}", mode='r')
    print(
        f"reopened: axis={reopened.axis} parts={list(reopened.partitions())}")

    full = reopened.query()
    assert numpy.array_equal(full['time'].to_numpy(), ds['time'].to_numpy())
    assert numpy.array_equal(full['ssh'].to_numpy(), ds['ssh'].to_numpy())
    print('bit-exact round-trip: OK')

    sub = reopened.query(filters='partition == 2')
    assert sub['partition'].to_numpy().tolist() == [2] * 25_000
    print('filter pushdown: OK')


if __name__ == '__main__':
    target = Path(sys.argv[1]) if len(
        sys.argv) > 1 else Path(tempfile.gettempdir()) / 'zc3-walkthrough'
    main(target)
