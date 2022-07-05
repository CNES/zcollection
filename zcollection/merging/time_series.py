# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Merging a time series
=====================
"""
import numpy

from . import period
from .. import dataset


def merge_time_series(
    existing_ds: dataset.Dataset,
    inserted_ds: dataset.Dataset,
    axis: str,
    partitioning_dim: str,
) -> dataset.Dataset:
    """Merge two time series together.

    Replaces only the intersection between the existing dataset and the new one,
    but also keeps the existing records if they have not been updated.

    The following figure illustrates the implemented algorithm. Column ``A``
    represents the new data and column ``B``, the data already present. The
    different cells in the columns represent the hours on the day of the
    measurements. The merge result is shown in column ``C``. It contains the
    measurements of the column ``A`` or column ``B`` if column ``A`` does not
    replace them.

    .. figure:: ../images/merge_time_series.svg
        :align: center
        :width: 50%

    Args:
        existing_ds: The existing dataset.
        inserted_ds: The inserted dataset.
        axis: The axis to merge on.
        partitioning_dim: The name of the partitioning dimension.

    Returns:
        The merged dataset.
    """
    existing_axis = existing_ds.variables[axis].values
    inserted_axis = inserted_ds.variables[axis].values
    existing_period = period.Period(existing_axis.min(),
                                    existing_axis.max(),
                                    within=True)
    inserted_period = period.Period(inserted_axis.min(),
                                    inserted_axis.max(),
                                    within=True)

    relation = inserted_period.get_relation(existing_period)

    # The new piece is located before the existing data.
    if relation.is_before():
        return inserted_ds.concat(existing_ds, partitioning_dim)

    # The new piece is located after the existing data.
    if relation.is_after():
        return existing_ds.concat(inserted_ds, partitioning_dim)

    # The new piece replace the old one
    if relation.contains():
        return inserted_ds

    intersection = inserted_period.intersection(existing_period)

    # The new piece is located before, but there is an overlap
    # between the two datasets.
    if relation.is_before_overlapping():
        # pylint: disable=comparison-with-callable
        indices = numpy.where(
            # comparison between ndarray and datetime64
            existing_axis > intersection.end())[0]  # type: ignore
        # pylint: enable=comparison-with-callable
        return inserted_ds.concat(
            existing_ds.isel({partitioning_dim: indices}), partitioning_dim)

    # The new piece is located after, but there is an overlap
    # between the two datasets.
    if relation.is_after_overlapping():
        # pylint: disable=comparison-with-callable
        indices = numpy.where(
            # comparison between ndarray and datetime64
            existing_axis < intersection.begin)[0]  # type: ignore
        # pylint: enable=comparison-with-callable
        return existing_ds.isel({
            partitioning_dim: indices
        }).concat(inserted_ds, partitioning_dim)

    assert relation.is_inside()
    # comparison between ndarray and datetime64
    index = numpy.where(existing_axis < intersection.begin)[0]  # type: ignore
    before = existing_ds.isel(
        {partitioning_dim: slice(0, index[-1] + 1, None)})

    # pylint: disable=comparison-with-callable
    # comparison between ndarray and datetime64
    index = numpy.where(existing_axis > intersection.end())[0]  # type: ignore
    # pylint: enable=comparison-with-callable
    after = existing_ds.isel(
        {partitioning_dim: slice(index[0], index[-1] + 1, None)})

    return before.concat((inserted_ds, after), partitioning_dim)
