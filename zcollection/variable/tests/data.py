# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Create test variables.
======================
"""
import numcodecs
import numpy

from .. import Array, DelayedArray
from ... import meta


def array(name='var1', fill_value=0) -> Array:
    """Creates a test variable with the given name, fill value, dimensions, and
    attributes.

    Args:
        name: The name of the variable.
        fill_value: The fill value for uninitialized parts of the array.

    Returns:
        An Array object.
    """
    return Array(name=name,
                 data=numpy.arange(10, dtype='int64').reshape(5, 2),
                 dimensions=('x', 'y'),
                 attrs=(meta.Attribute(name='attr', value=1), ),
                 compressor=numcodecs.Blosc(cname='zstd', clevel=1),
                 fill_value=fill_value,
                 filters=(numcodecs.Delta('int64', 'int32'),
                          numcodecs.Delta('int32', 'int32')))


def delayed_array(name='var1', fill_value=0) -> DelayedArray:
    """Create a delayed test variable with the given name, fill value,
    dimensions, and attributes.

    Args:
        name: The name of the variable.
        fill_value: The fill value for uninitialized parts of the array.

    Returns:
        A DelayedArray object representing a lazily-evaluated test variable.
    """
    return DelayedArray(name=name,
                        data=numpy.arange(10, dtype='int64').reshape(5, 2),
                        dimensions=('x', 'y'),
                        attrs=(meta.Attribute(name='attr', value=1), ),
                        compressor=numcodecs.Blosc(cname='zstd', clevel=1),
                        fill_value=fill_value,
                        filters=(numcodecs.Delta('int64', 'int32'),
                                 numcodecs.Delta('int32', 'int32')))
