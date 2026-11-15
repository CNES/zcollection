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
    """Create a NumPy-backed test variable with fixed data and metadata.

    Args:
        name: The name of the variable.
        fill_value: The fill value for uninitialized elements.

    Returns:
        The test Array.
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
    """Create a Dask-backed test variable with fixed data and metadata.

    Args:
        name: The name of the variable.
        fill_value: The fill value for uninitialized elements.

    Returns:
        The test DelayedArray.
    """
    return DelayedArray(name=name,
                        data=numpy.arange(10, dtype='int64').reshape(5, 2),
                        dimensions=('x', 'y'),
                        attrs=(meta.Attribute(name='attr', value=1), ),
                        compressor=numcodecs.Blosc(cname='zstd', clevel=1),
                        fill_value=fill_value,
                        filters=(numcodecs.Delta('int64', 'int32'),
                                 numcodecs.Delta('int32', 'int32')))
