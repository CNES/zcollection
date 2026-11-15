# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Merging a time series
=====================
"""
from typing import TYPE_CHECKING

import numpy

from . import period
from .. import dataset

if TYPE_CHECKING:
    from ..type_hints import NDArray


def _merge_time_series(
    existing_ds: dataset.Dataset,
    inserted_ds: dataset.Dataset,
    axis: str,
    partitioning_dim: str,
) -> dataset.Dataset:
    """Merge two time series.

    See :func:`merge_time_series` for details.
    """
    existing_axis: NDArray = existing_ds.variables[axis].values
    inserted_axis: NDArray = inserted_ds.variables[axis].values
    existing_period = period.Period(existing_axis.min(),
                                    existing_axis.max(),
                                    within=True)
    inserted_period = period.Period(inserted_axis.min(),
                                    inserted_axis.max(),
                                    within=True)

    relation: period.PeriodRelation = inserted_period.get_relation(
        existing_period)

    # The new piece is located before the existing data.
    if relation.is_before():
        return inserted_ds.concat(existing_ds, partitioning_dim)

    # The new piece is located after the existing data.
    if relation.is_after():
        return existing_ds.concat(inserted_ds, partitioning_dim)

    # The new data replaces the existing data.
    if relation.contains():
        return inserted_ds

    intersection: period.Period = inserted_period.intersection(existing_period)

    # The new piece is located before, but there is an overlap
    # between the two datasets.
    if relation.is_before_overlapping():

        indices = numpy.nonzero(
            # Compare NumPy array values with datetime64 values.
            existing_axis > inserted_period.last)[0]

        return inserted_ds.concat(
            existing_ds.isel({partitioning_dim: indices}), partitioning_dim)

    # The new piece is located after, but there is an overlap
    # between the two datasets.
    if relation.is_after_overlapping():

        indices = numpy.nonzero(
            # Compare NumPy array values with datetime64 values.
            existing_axis < intersection.begin)[0]

        return existing_ds.isel({
            partitioning_dim: indices
        }).concat(inserted_ds, partitioning_dim)

    assert relation.is_inside()
    # Compare NumPy array values with datetime64 values.
    index = numpy.nonzero(existing_axis < intersection.begin)[0]
    before: dataset.Dataset = existing_ds.isel(
        {partitioning_dim: slice(0, index[-1] + 1, None)})

    # Compare NumPy array values with datetime64 values.
    index = numpy.nonzero(existing_axis > inserted_period.last)[0]

    after: dataset.Dataset = existing_ds.isel(
        {partitioning_dim: slice(index[0], index[-1] + 1, None)})

    return before.concat((inserted_ds, after), partitioning_dim)


def merge_time_series(
    existing_ds: dataset.Dataset,
    inserted_ds: dataset.Dataset,
    axis: str,
    partitioning_dim: str,
    **kwargs,
) -> dataset.Dataset:
    """Merge two time series, replacing overlapping values with new data.

    Existing records outside the inserted time series are retained. Where
    the series overlap, values from the inserted dataset replace existing
    values.

    The figure shows the merge: column ``A`` contains inserted measurements,
    column ``B`` contains existing measurements, and column ``C`` contains
    the result. Each cell represents an hour of measurements.

    .. figure:: ../images/merge_time_series.svg
        :align: center
        :width: 50%

    Args:
        existing_ds: The existing dataset.
        inserted_ds: The inserted dataset.
        axis: The axis to merge on.
        partitioning_dim: The name of the partitioning dimension.
        kwargs: Additional options, including ``tolerance``. When specified,
            ``tolerance`` detects gaps in the inserted axis and splits the
            inserted data into segments before merging them. With ``None``,
            gaps are not checked.

    Returns:
        The merged dataset.
    """
    tolerance = kwargs.get('tolerance', None)
    index: NDArray

    # Check if the inserted dataset contains data gaps.
    if tolerance is not None:
        inserted_axis: NDArray = inserted_ds.variables[axis].values
        delta: NDArray = numpy.concatenate((
            numpy.array([0]),
            numpy.diff(numpy.roll(inserted_axis, shift=0)),
        ))
        index = numpy.concatenate((
            numpy.array([0], dtype=numpy.int64),
            numpy.nonzero(delta > tolerance)[0],
            numpy.array([inserted_axis.size], dtype=numpy.int64),
        ))
    else:
        index = numpy.array([], dtype=numpy.int64)

    if index.size > 1:
        # Split the inserted dataset at gaps before merging each segment.
        for ix in range(len(index) - 1):
            existing_ds = _merge_time_series(
                existing_ds,
                inserted_ds.isel(
                    {partitioning_dim: slice(index[ix], index[ix + 1], None)}),
                axis, partitioning_dim)
        return existing_ds

    return _merge_time_series(existing_ds, inserted_ds, axis, partitioning_dim)
