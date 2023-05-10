# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test partitioning by date.
==========================
"""
from __future__ import annotations

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
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ...type_hints import NDArray

# pylint: disable=disable=unused-argument

#: First date of the dataset to partition
START_DATE = numpy.datetime64('2000-01-06', 'ns')

#: Time delta between two partitions
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

        expected_selection = self.dates[
            (self.dates >= parsed_date)
            & (self.dates < parsed_date + self.timedelta)]
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
def test_split_dataset(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    delayed: bool,
) -> None:
    """Test the split_dataset method."""
    for end_date, indices, resolution in [
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
    ]:

        # Time delta between two partitions
        timedelta = numpy.timedelta64(1, resolution)

        # Temporal axis to split
        dates: NDArray = numpy.arange(START_DATE, end_date, TIME_DELTA)

        # Measured data
        observation: NDArray = numpy.random.rand(dates.size)  # type: ignore

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
        date = numpy.datetime64(START_DATE, resolution)

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
    """Test the construction of the Date class."""
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

    with pytest.raises(ValueError):
        Date(('dates1', 'dates2'), 'D')

    with pytest.raises(ValueError):
        Date(('dates', ), 'W')


def test_config():
    """Test the configuration of the Date class."""
    partitioning = Date(('dates', ), 'D')
    assert partitioning.dtype() == (('year', 'uint16'), ('month', 'uint8'),
                                    ('day', 'uint8'))
    config = partitioning.get_config()
    partitioning = get_codecs(config)
    assert isinstance(partitioning, Date)


def test_pickle():
    """Test the pickling of the Date class."""
    partitioning = Date(('dates', ), 'D')
    other = pickle.loads(pickle.dumps(partitioning))
    assert isinstance(other, Date)
    assert other.resolution == 'D'
    assert other.variables == ('dates', )


@pytest.mark.parametrize('delayed', [False, True])
def test_no_monotonic(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    delayed: bool,
):
    """Test that the Date partitioning raises an error if the temporal axis is
    not monotonic."""
    dates: numpy.ndarray = numpy.arange(numpy.datetime64('2000-01-01', 'h'),
                                        numpy.datetime64('2000-01-02', 'h'),
                                        numpy.timedelta64(1, 'm'))
    numpy.random.shuffle(dates)
    partitioning = Date(('dates', ), 'h')
    # pylint: disable=protected-access
    with pytest.raises(ValueError):
        arr = dask.array.core.from_array(dates) if delayed else dates
        list(partitioning._split({'dates': arr}))  # type: ignore[arg-type]
    # pylint: enable=protected-access


def test_values_must_be_datetime64(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test that the values must be datetime64."""
    dates = numpy.arange(numpy.datetime64('2000-01-01', 'h'),
                         numpy.datetime64('2000-01-02', 'h'),
                         numpy.timedelta64(1, 'm'))
    partitioning = Date(('dates', ), 'h')
    dates = dates.astype('int64')
    with pytest.raises(TypeError):
        # pylint: disable=protected-access
        list(
            partitioning._split({
                'dates':
                dask.array.core.from_array(dates)  # type: ignore[arg-type]
            }))
    # pylint: enable=protected-access


@pytest.mark.parametrize(
    'start, end, step, path_generator',
    [(('2000-01-01', 'D'), ('2000-02-01', 'D'), (1, 'D'), lambda item:
      (f'year={item.year}', f'month={item.month:02d}', f'day={item.day:02d}')),
     (('2000', 'Y'), ('2005', 'Y'), (1, 'Y'), lambda item:
      (f'year={item.year}', ))])
def test_listing_partition(start, end, step, path_generator):
    """Test the listing of the partitions."""
    fs = fsspec.filesystem('memory')
    variables = [
        'ancillary_surface_classification_flag',
        'correction_flag',
        'cross_track_angle',
        'cross_track_distance',
        'cycle_number',
        'dac',
        'depth_or_elevation',
        'distance_to_coast',
        'doppler_centroid',
        'dynamic_ice_flag',
        'geoid',
        'heading_to_coast',
        'height_cor_xover',
        'ice_conc',
        'internal_tide_hret',
        'internal_tide_sol2',
        'inv_bar_cor',
        'iono_cor_gim_ka',
        'latitude',
        'latitude_avg_ssh',
        'latitude_nadir',
        'load_tide_fes',
        'load_tide_got',
        'longitude',
        'longitude_avg_ssh',
        'longitude_nadir',
        'mean_dynamic_topography',
        'mean_dynamic_topography_uncert',
        'mean_sea_surface_cnescls',
        'mean_sea_surface_cnescls_uncert',
        'mean_sea_surface_dtu',
        'mean_sea_surface_dtu_uncert',
        'mean_wave_direction',
        'mean_wave_period_t02',
        'model_dry_tropo_cor',
        'model_wet_tropo_cor',
        'num_pt_avg',
        'obp_ref_surface',
        'ocean_tide_eq',
        'ocean_tide_fes',
        'ocean_tide_got',
        'ocean_tide_non_eq',
        'orbit_alt_rate',
        'orbit_qual',
        'pass_number',
        'phase_bias_ref_surface',
        'polarization_karin',
        'pole_tide',
        'rad_cloud_liquid_water',
        'rad_surface_type_flag',
        'rad_tmb_187',
        'rad_tmb_238',
        'rad_tmb_340',
        'rad_water_vapor',
        'rad_wet_tropo_cor',
        'rain_flag',
        'rain_rate',
        'sc_altitude',
        'sc_pitch',
        'sc_roll',
        'sc_yaw',
        'sea_state_bias_cor',
        'sea_state_bias_cor_2',
        'sig0_cor_atmos_model',
        'sig0_cor_atmos_rad',
        'sig0_karin',
        'sig0_karin_2',
        'sig0_karin_qual',
        'sig0_karin_uncert',
        'simulated_error_baseline_dilation',
        'simulated_error_karin',
        'simulated_error_orbital',
        'simulated_error_phase',
        'simulated_error_roll',
        'simulated_error_timing',
        'simulated_error_troposphere',
        'simulated_true_ssh_karin',
        'solid_earth_tide',
        'ssha_karin',
        'ssha_karin_2',
        'ssha_karin_qual',
        'ssh_karin',
        'ssh_karin_2',
        'ssh_karin_uncert',
        'swh_karin',
        'swh_karin_qual',
        'swh_karin_uncert',
        'swh_model',
        'swh_sea_state_bias',
        'time',
        'time_tai',
        'velocity_heading',
        'wind_speed_karin',
        'wind_speed_karin_2',
        'wind_speed_model_u',
        'wind_speed_model_v',
        'wind_speed_rad',
        'x_factor',
    ]
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
