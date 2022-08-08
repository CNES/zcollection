# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test partitioning by date.
==========================
"""
from typing import Iterator
import pickle

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

# pylint: disable=disable=unused-argument


def test_split_dataset(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test the split_dataset method."""
    start_date = numpy.datetime64('2000-01-06', 'us')
    delta = numpy.timedelta64(1, 'h')

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
        dates = numpy.arange(start_date, end_date, delta)

        # Measured data
        observation = numpy.random.rand(dates.size)  # type: ignore

        # Create the dataset to split
        ds = xarray.Dataset(
            dict(dates=xarray.DataArray(dates, dims=('num_lines', )),
                 observation=xarray.DataArray(observation,
                                              dims=('num_lines', ))))

        partitioning = Date(('dates', ), resolution)
        assert len(partitioning) == len(range(indices.start, indices.stop))

        # Date of the current partition
        date = numpy.datetime64(start_date, resolution)

        # Build the test dataset
        ds = dataset.Dataset.from_xarray(ds)

        iterator = partitioning.split_dataset(ds, 'num_lines')
        assert isinstance(iterator, Iterator)

        for partition, indexer in iterator:
            subset = ds.isel(indexer)

            # Cast the date to the a datetime object to extract the date
            item = date.astype('datetime64[us]').item()
            expected = (
                f'year={item.year}',
                f'month={item.month:02d}',
                f'day={item.day:02d}',
                f'hour={item.hour:02d}',
            )
            assert partition == expected[indices]

            folder = '/'.join(partition)
            fields = partitioning.parse(folder)
            parsed_date, = partitioning.encode(fields)
            assert parsed_date == numpy.datetime64(date).astype(
                f'datetime64[{resolution}]')

            expected_selection = dates[
                (dates >= parsed_date)  # type: ignore
                & (dates < parsed_date + timedelta)]  # type: ignore
            computed_selection = subset.variables['dates'].compute(
                scheduler=dask.local.get_sync)
            assert numpy.all(computed_selection == expected_selection)

            expected = (
                ('year', item.year),
                ('month', item.month),
                ('day', item.day),
                ('hour', item.hour),
            )
            assert fields == expected[indices]
            assert partitioning.join(fields, '/') == folder
            assert partitioning.join(partitioning.decode((parsed_date, )),
                                     '/') == folder

            date += timedelta


def test_construction():
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


def test_no_monotonic(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test that the Date partitioning raises an error if the temporal axis is
    not monotonic."""
    dates = numpy.arange(numpy.datetime64('2000-01-01', 'h'),
                         numpy.datetime64('2000-01-02', 'h'),
                         numpy.timedelta64(1, 'm'))
    numpy.random.shuffle(dates)
    partitioning = Date(('dates', ), 'h')
    # pylint: disable=protected-access
    with pytest.raises(ValueError):
        list(partitioning._split({'dates': dask.array.core.from_array(dates)}))
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
        list(partitioning._split({'dates': dask.array.core.from_array(dates)}))
    # pylint: enable=protected-access


def test_listing_partition():
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
    start = numpy.datetime64('2000-01-01', 'D')
    end = numpy.datetime64('2000-02-01', 'D')
    delta = numpy.timedelta64(1, 'D')

    root = '/zcollection'
    fs.mkdir(root)
    fs.open(fs.sep.join((root, '.zcollection')), 'w').close()

    expected = []
    for date in numpy.arange(start, end, delta):
        item = date.item()
        partition = fs.sep.join(
            (root, f'year={item.year}', f'month={item.month:02d}',
             f'day={item.day:02d}'))
        expected.append(partition)
        fs.mkdirs(partition)

        _ = {fs.mkdirs(fs.sep.join((partition, item))) for item in variables}

        _ = {
            fs.open(fs.sep.join((partition, item)), 'w').close()
            for item in ['.zattrs', '.zgroup', '.zmetadata']
        }

    partitioning = Date(('dates', ), 'D')
    assert expected == list(partitioning.list_partitions(fs, root))
