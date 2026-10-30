# Copyright (c) 2026 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Collection schema validation."""
import numcodecs
import pytest

from ... import meta
from ..abc import _check_fill_values, _check_sizes


def test_a_dimension_declared_without_a_size_is_refused() -> None:
    """Only the partitioned dimension can have an unspecified size."""
    schema = meta.DatasetSchema(
        dimensions=[meta.Dimension('row', -1),
                    meta.Dimension('pixel', 2)],
        variables=[])
    _check_sizes(schema, 'row')
    schema.dimensions['pixel'].value = -1

    with pytest.raises(ValueError, match="'pixel' is declared without a size"):
        _check_sizes(schema, 'row')


def test_a_value_marking_an_absence_its_filters_lose_is_refused() -> None:
    """Filters must preserve the declared fill value."""
    codec = numcodecs.FixedScaleOffset(offset=0,
                                       scale=1000,
                                       dtype='f8',
                                       astype='i2')
    variable = meta.VariableSchema('value',
                                   'float64',
                                   dimensions=('row', ),
                                   fill_value=-999.0,
                                   filters=(codec, ))
    schema = meta.DatasetSchema(dimensions=[meta.Dimension('row', -1)],
                                variables=[variable])

    with pytest.raises(ValueError, match='marks an absence with'):
        _check_fill_values(schema)
