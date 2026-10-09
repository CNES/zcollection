# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Reference collection
====================

A small collection whose content exercises the storage semantics that a
change of storage backend can alter without raising: a variable with no
fill value, a variable with one, a filtered variable, and a datetime64
axis.

The dataset is deterministic, so a collection written from it can be
compared byte for byte across versions.
"""
from __future__ import annotations

import numcodecs
import numpy
import zarr

from .. import dataset, partitioning

#: First date of the reference dataset.
START_DATE = numpy.datetime64('2000-01-01', 'ns')

#: Delta between two measures.
DELTA = numpy.timedelta64(6, 'h')

#: Number of measures. Four per day over three days.
SIZE = 12

#: Fill value of the ``masked`` variable.
FILL_VALUE = -999.0

#: Scale applied by the filter of the filtered variables.
SCALE = 1000

#: Fill value of the ``scaled_masked`` variable. The filter encodes it
#: without loss, which is what :mod:`zcollection.meta` requires of a fill
#: value sitting behind a filter.
SCALED_FILL_VALUE = -10.0

#: Fill value of the ``counts`` variable. No measure carries it.
COUNT_FILL_VALUE = -99

#: Name of the partitioning dimension.
DIMENSION = 'num_lines'

#: Name of the dimension the immutable variables carry.
IMMUTABLE_DIMENSION = 'num_pixels'

#: Length of the immutable dimension.
NUM_PIXELS = 3


def axis() -> numpy.ndarray:
    """Return the values of the partitioning axis."""
    return START_DATE + numpy.arange(SIZE) * DELTA


def plain_values() -> numpy.ndarray:
    """Return values holding exact zeros and no fill value.

    A zero here is a measure, not an absence. Reading it back as a masked
    value is a corruption that raises nothing.
    """
    values = numpy.arange(SIZE, dtype='float64') - 4.0
    assert 0.0 in values
    return values


def masked_values() -> numpy.ndarray:
    """Return values holding both the fill value and exact zeros."""
    values = numpy.arange(SIZE, dtype='float64') - 4.0
    values[2] = FILL_VALUE
    values[7] = FILL_VALUE
    return values


def scaled_values() -> numpy.ndarray:
    """Return values a FixedScaleOffset filter encodes without loss."""
    return numpy.round(numpy.linspace(-1.5, 1.5, SIZE), 3)


def scaled_masked_values() -> numpy.ndarray:
    """Return filtered values holding the fill value the filter decodes to."""
    values = scaled_values().copy()
    values[1] = SCALED_FILL_VALUE
    values[9] = SCALED_FILL_VALUE
    return values


def count_values() -> numpy.ndarray:
    """Return integer values holding the fill value and exact zeros."""
    values = numpy.arange(SIZE, dtype='int32') - 3
    values[5] = COUNT_FILL_VALUE
    return values


def grid_values() -> numpy.ndarray:
    """Return values spanning the partitioning and the immutable dimension."""
    return numpy.arange(SIZE * NUM_PIXELS,
                        dtype='float64').reshape(SIZE, NUM_PIXELS)


def pixel_values() -> numpy.ndarray:
    """Return values carried by the immutable dimension alone.

    A variable that does not span the partitioning dimension is stored once,
    beside the partitions, not in each of them.
    """
    return numpy.arange(NUM_PIXELS, dtype='int16') * 10


def filter_codec() -> numcodecs.abc.Codec:
    """Return the filter the scaled variables are stored behind."""
    return numcodecs.FixedScaleOffset(offset=0,
                                      scale=SCALE,
                                      dtype='f8',
                                      astype='i4')


def make_dataset(delayed: bool = False,
                 immutable: bool = True) -> dataset.Dataset:
    """Build the reference dataset.

    Args:
        delayed: Whether to hold the values in Dask arrays.
        immutable: Whether to carry the variable that does not span the
            partitioning dimension. Release 2026.6.0 selects the variables
            to partition by the name of the axis rather than by the name of
            the dimension. It writes such a variable when the two names
            match, and fails when they differ, as they do here.

    Returns:
        The dataset.
    """
    array = dataset.DelayedArray if delayed else dataset.Array
    variables: tuple[dataset.Variable, ...] = (
        array(
            name='time',
            data=axis(),
            dimensions=(DIMENSION, ),
            attrs=(dataset.Attribute(name='units', value='ns'), ),
            compressor=zarr.Blosc(),
        ),
        array(
            name='plain',
            data=plain_values(),
            dimensions=(DIMENSION, ),
            attrs=(dataset.Attribute(name='units', value='m'), ),
        ),
        array(
            name='masked',
            data=masked_values(),
            dimensions=(DIMENSION, ),
            attrs=(dataset.Attribute(name='units', value='m'), ),
            fill_value=FILL_VALUE,
        ),
        array(
            name='scaled',
            data=scaled_values(),
            dimensions=(DIMENSION, ),
            attrs=(dataset.Attribute(name='units', value='m'), ),
            filters=(filter_codec(), ),
        ),
        array(
            name='scaled_masked',
            data=scaled_masked_values(),
            dimensions=(DIMENSION, ),
            attrs=(dataset.Attribute(name='units', value='m'), ),
            fill_value=SCALED_FILL_VALUE,
            filters=(filter_codec(), ),
        ),
        array(
            name='counts',
            data=count_values(),
            dimensions=(DIMENSION, ),
            attrs=(dataset.Attribute(name='units', value='1'), ),
            fill_value=COUNT_FILL_VALUE,
        ),
        array(
            name='grid',
            data=grid_values(),
            dimensions=(DIMENSION, IMMUTABLE_DIMENSION),
            attrs=(dataset.Attribute(name='units', value='m'), ),
        ),
    )
    if immutable:
        variables += (array(
            name='pixel',
            data=pixel_values(),
            dimensions=(IMMUTABLE_DIMENSION, ),
            attrs=(dataset.Attribute(name='units', value='1'), ),
        ), )
    return dataset.Dataset(
        attrs=(dataset.Attribute(name='title', value='reference collection'),
               dataset.Attribute(name='revision', value=1)),
        variables=variables,
    )


def make_partitioning() -> partitioning.Partitioning:
    """Return the partitioning of the reference collection: one per day."""
    return partitioning.Date(('time', ), 'D')
