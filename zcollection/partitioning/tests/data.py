# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Make test data.
===============
"""
import numpy
import xarray

START_DATE = numpy.datetime64('2000-01-01', 'us')
END_DATE = numpy.datetime64('2000-06-30', 'us')
DELTA = numpy.timedelta64(72, 'h')


def create_test_sequence(repeatability, number_of_measures, number_of_cycles):
    """Creation of a data set for testing purposes."""
    pass_number = numpy.hstack([
        numpy.tile(ix + 1, number_of_measures) for j in range(number_of_cycles)
        for ix in range(repeatability)
    ])
    cycle_number = numpy.hstack([
        numpy.tile(ix + 1, repeatability * number_of_measures)
        for ix in range(number_of_cycles)
    ])
    delta = numpy.timedelta64(24 // repeatability // 2, 'h')
    time = numpy.arange(START_DATE, START_DATE + len(cycle_number) * delta,
                        delta)
    observation = numpy.random.rand(cycle_number.size)  # type: ignore
    ds = xarray.Dataset(
        dict(time=xarray.DataArray(time, dims=('num_lines', )),
             cycle_number=xarray.DataArray(cycle_number, dims=('num_lines', )),
             pass_number=xarray.DataArray(pass_number, dims=('num_lines', )),
             observation=xarray.DataArray(observation, dims=('num_lines', ))))
    return ds
