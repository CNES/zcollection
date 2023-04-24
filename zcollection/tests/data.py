# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Make test datasets
==================
"""
from __future__ import annotations

import itertools

import numpy
import zarr

from .. import collection, dataset, partitioning
from ..partitioning.tests.data import create_test_sequence

START_DATE = numpy.datetime64('2000-01-01', 'us')
END_DATE = numpy.datetime64('2000-06-30', 'us')
DELTA = numpy.timedelta64(72, 'h')
FILL_VALUE = 2147483647


def make_dataset(dates: numpy.ndarray,
                 measures: numpy.ndarray,
                 fill_value: float | None = None,
                 filters: tuple | None = None) -> dataset.Dataset:
    """Create a dataset."""
    return dataset.Dataset(
        attrs=(dataset.Attribute(name='attr', value=1), ),
        variables=(
            dataset.DelayedArray(
                name='time',
                data=dates,
                dimensions=('num_lines', ),
                attrs=(dataset.Attribute(name='attr', value=1), ),
                compressor=zarr.Blosc(),
            ),
            dataset.DelayedArray(name='var1',
                                 data=measures,
                                 dimensions=('num_lines', 'num_pixels'),
                                 attrs=(dataset.Attribute(name='attr',
                                                          value=1), ),
                                 fill_value=fill_value,
                                 filters=filters),
            dataset.DelayedArray(name='var2',
                                 data=measures,
                                 dimensions=('num_lines', 'num_pixels'),
                                 attrs=(dataset.Attribute(name='attr',
                                                          value=1), ),
                                 fill_value=fill_value,
                                 filters=filters),
        ),
    )


def create_test_dataset():
    """Create a temporal dataset."""

    dates = numpy.arange(START_DATE, END_DATE, DELTA)
    indices = numpy.arange(0, len(dates))

    for item in numpy.array_split(dates, 12):
        mask = (dates >= item[0]) & (dates <= item[-1])
        measures = numpy.vstack((indices[mask], ) * 25).T

        yield make_dataset(item, measures)


def create_test_dataset_with_fillvalue():
    """Create a dataset with a fixed scale offset filter and fill values."""

    dates = numpy.arange(START_DATE, END_DATE, DELTA)
    measures = numpy.arange(0, len(dates), dtype=numpy.float64)
    measures[measures % 2 == 0] = FILL_VALUE
    measures = numpy.vstack((measures, ) * 25).T * 1e-4

    yield make_dataset(dates,
                       measures,
                       fill_value=FILL_VALUE * 1e-4,
                       filters=(zarr.FixedScaleOffset(scale=10000,
                                                      offset=0,
                                                      dtype='<f8',
                                                      astype='i4'), ))


def create_test_collection(tested_fs, with_fillvalue=False):
    """Create a collection."""
    ds = next(create_test_dataset_with_fillvalue(
    ) if with_fillvalue else create_test_dataset())
    zcollection = collection.Collection('time',
                                        ds.metadata(),
                                        partitioning.Date(('time', ), 'D'),
                                        str(tested_fs.collection),
                                        filesystem=tested_fs.fs)
    zcollection.insert(ds)
    return zcollection


FILE_SYSTEM_DATASET = list(
    itertools.product([
        'local_fs',
        's3_fs',
    ], [
        create_test_dataset,
        create_test_dataset_with_fillvalue,
    ]))
