# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Date partitioning tests
=======================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import dataclasses
import pickle
import random
import string

import dask.array.core
import dask.local
import fsspec
import numpy
import pytest
import xarray

from .. import Date, get_codecs
from ... import dataset

if TYPE_CHECKING:
    from ...type_hints import NDArray

#: First date of the dataset to partition
START_DATE = numpy.datetime64('2000-01-06', 'ns')

#: Time step between consecutive dates in the test dataset.
TIME_DELTA = numpy.timedelta64(1, 'h')


@dataclasses.dataclass(frozen=True)
class PartitionTestData:
    """Test data for partitioning."""
    timedelta: numpy.timedelta64
    indices: slice
    resolution: str
    partitioning: Date
    dates: NDArray

    def check_partitioning(self, date: numpy.datetime64, zds: dataset.Dataset,
                           partition: tuple[str, ...]) -> None:
        """Check the partitioning of a dataset."""
        item = date.astype('datetime64[us]').item()
        assert partition == (
            f'year={item.year}',
            f'month={item.month:02d}',
            f'day={item.day:02d}',
            f'hour={item.hour:02d}',
        )[self.indices]

        folder = '/'.join(partition)
        fields = self.partitioning.parse(folder)
        parsed_date, = self.partitioning.encode(fields)
        assert parsed_date == numpy.datetime64(date).astype(
            f'datetime64[{self.resolution}]')

        expected_selection = self.dates[(self.dates >= parsed_date)
                                        & (self.dates < parsed_date +
                                           self.timedelta)]
        assert numpy.all(zds.variables['dates'].compute(
            scheduler=dask.local.get_sync) == expected_selection)

        assert fields == (
            ('year', item.year),
            ('month', item.month),
            ('day', item.day),
            ('hour', item.hour),
        )[self.indices]
        assert self.partitioning.join(fields, '/') == folder
        assert self.partitioning.join(
            self.partitioning.decode((parsed_date, )), '/') == folder


@pytest.mark.parametrize('delayed', [False, True])
def test_split_dataset(delayed: bool) -> None:
    """Test the split_dataset method."""
    splits: list[tuple[numpy.datetime64, slice,
                       Literal['Y', 'M', 'D', 'h']]] = [
                           (
                               numpy.datetime64('2001-12-31', 'Y'),
                               slice(0, 1),
                               'Y',
                           ),
                           (
                               numpy.datetime64('2000-12-31', 'M'),
                               slice(0, 2),
                               'M',
                           ),
                           (
                               numpy.datetime64('2000-12-31', 'D'),
                               slice(0, 3),
                               'D',
                           ),
                           (
                               numpy.datetime64('2000-01-31', 'h'),
                               slice(0, 4),
                               'h',
                           ),
                       ]
    for end_date, indices, resolution in splits:
        # Time delta between two partitions
        timedelta = numpy.timedelta64(1, resolution)

        # Temporal axis to split
        dates: NDArray = numpy.arange(START_DATE, end_date, TIME_DELTA)

        # Measured data
        rng = numpy.random.default_rng(42)
        observation: NDArray = rng.random(dates.size)

        # Create the dataset to split
        xds = xarray.Dataset({
            'dates':
            xarray.DataArray(dates, dims=('num_lines', )),
            'observation':
            xarray.DataArray(observation, dims=('num_lines', ))
        })

        partitioning = Date(('dates', ), resolution)
        assert len(partitioning) == len(range(indices.start, indices.stop))

        # Date of the current partition
        date = START_DATE.astype(f'datetime64[{resolution}]')

        # Build the test dataset
        zds = dataset.Dataset.from_xarray(xds)
        if not delayed:
            zds = zds.compute()

        checker = PartitionTestData(timedelta, indices, resolution,
                                    partitioning, dates)

        for partition, indexer in partitioning.split_dataset(zds, 'num_lines'):
            checker.check_partitioning(date, zds.isel(indexer), partition)
            date += timedelta


def test_construction() -> None:
    """Test constructing a Date partitioning scheme."""
    partitioning = Date(('dates', ), 'D')
    assert partitioning.resolution == 'D'
    assert partitioning.variables == ('dates', )
    assert partitioning.dtype() == (('year', 'uint16'), ('month', 'uint8'),
                                    ('day', 'uint8'))
    assert len(partitioning) == 3
    assert partitioning.get_config() == {
        'id': 'Date',
        'resolution': 'D',
        'variables': ('dates', ),
    }

    with pytest.raises(
            ValueError,
            match=r'Partitioning on dates is performed on a single variable.'):
        Date(('dates1', 'dates2'), 'D')

    with pytest.raises(ValueError,
                       match='resolution must be in: Y, M, D, h, m, s'):
        Date(('dates', ), 'W')


def test_pickle() -> None:
    """A restored Date retains its configuration and partitions dates."""
    partitioning = Date(('dates', ), resolution='h')
    other = pickle.loads(pickle.dumps(partitioning))
    zds = dataset.Dataset([
        dataset.Array(
            'dates',
            numpy.array(['2000-01-02T03', '2000-01-02T04'],
                        dtype='datetime64[h]'), ('row', )),
    ],
                          delayed=False)

    assert isinstance(other, Date)
    assert other.get_config() == partitioning.get_config()
    assert list(other.split_dataset(zds, 'row')) == [
        (('year=2000', 'month=01', 'day=02', 'hour=03'), {
            'row': slice(0, 1)
        }),
        (('year=2000', 'month=01', 'day=02', 'hour=04'), {
            'row': slice(1, 2)
        }),
    ]


RESOLUTION_DTYPE_TEST_SET = [
    ('Y', (('year', 'uint16'), )),
    ('M', (('year', 'uint16'), ('month', 'uint8'))),
    ('D', (('year', 'uint16'), ('month', 'uint8'), ('day', 'uint8'))),
    ('h', (('year', 'uint16'), ('month', 'uint8'), ('day', 'uint8'),
           ('hour', 'uint8'))),
    ('m', (('year', 'uint16'), ('month', 'uint8'), ('day', 'uint8'),
           ('hour', 'uint8'), ('minute', 'uint8'))),
    ('s', (('year', 'uint16'), ('month', 'uint8'), ('day', 'uint8'),
           ('hour', 'uint8'), ('minute', 'uint8'), ('second', 'uint8')))
]


@pytest.mark.parametrize(('resolution', 'dtype'), RESOLUTION_DTYPE_TEST_SET)
def test_config(resolution, dtype):
    """Test the configuration of the Date class."""
    partitioning = Date(variables=('dates', ), resolution=resolution)
    assert partitioning.dtype() == dtype

    config = partitioning.get_config()
    other = get_codecs(config)

    assert isinstance(other, Date)
    assert other.variables == ('dates', )
    assert other.dtype() == dtype


@pytest.mark.parametrize('delayed', [False, True])
def test_no_monotonic(delayed: bool):
    """Reject non-monotonic dates when partitioning."""
    dates: numpy.ndarray = numpy.arange(numpy.datetime64('2000-01-01', 'h'),
                                        numpy.datetime64('2000-01-02', 'h'),
                                        numpy.timedelta64(1, 'm'))
    rng = numpy.random.default_rng(42)
    rng.shuffle(dates)
    partitioning = Date(('dates', ), 'h')

    with pytest.raises(ValueError, match='is not monotonic'):
        list(
            partitioning._split({
                'dates':
                dask.array.core.from_array(  # type: ignore[arg-type]
                    dates) if delayed else dates
            }))


def test_values_must_be_datetime64():
    """Reject values that are not datetime64."""
    dates = numpy.arange(numpy.datetime64('2000-01-01', 'h'),
                         numpy.datetime64('2000-01-02', 'h'),
                         numpy.timedelta64(1, 'm'))
    partitioning = Date(('dates', ), 'h')
    dates = dates.astype('int64')
    with pytest.raises(TypeError):

        list(
            partitioning._split({
                'dates':
                dask.array.core.from_array(dates)  # type: ignore[arg-type]
            }))


@pytest.mark.parametrize(
    ('start', 'end', 'step', 'path_generator'),
    [(('2000-01-01', 'D'), ('2000-02-01', 'D'), (1, 'D'), lambda item:
      (f'year={item.year}', f'month={item.month:02d}', f'day={item.day:02d}')),
     (('2000', 'Y'), ('2005', 'Y'), (1, 'Y'), lambda item:
      (f'year={item.year}', ))])
def test_listing_partition(start, end, step, path_generator):
    """Test the listing of the partitions."""
    fs = fsspec.filesystem('memory')
    variables = ['time', 'values']
    root = '/' + ''.join(random.choices(string.ascii_letters, k=10))
    fs.mkdir(root)
    fs.open(fs.sep.join((root, '.zcollection')), 'w').close()

    partitioning = Date(('dates', ), step[1])

    start = numpy.datetime64(*start)
    end = numpy.datetime64(*end)
    step = numpy.timedelta64(*step)

    expected = []
    for date in numpy.arange(start, end, step):
        item = date.item()
        partition = fs.sep.join((root, *path_generator(item)))
        expected.append(partition)
        fs.mkdirs(partition)

        _ = {fs.mkdirs(fs.sep.join((partition, item))) for item in variables}
        _ = {
            fs.open(fs.sep.join((partition, item)), 'w').close()
            for item in ['.zattrs', '.zgroup', '.zmetadata']
        }

    assert expected == list(partitioning.list_partitions(fs, root))
