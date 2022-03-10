# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Typing
======

.. py:data:: NDArray

    A numpy tensor with any type.

.. py:data:: NDMaskedArray

    A numpy masked tensor with any type.

.. py:data:: ArrayLike

    Objects that can be converted to arrays

.. py:data:: NDArrayLike

    Anything that can be coerced into numpy.dtype.
"""
from typing import TYPE_CHECKING, Any, Union
import sys

import dask.array
import numpy
import numpy.typing
import packaging.version

#: An array object represents a multidimensional, homogeneous array of
#: fixed-size items.
if TYPE_CHECKING and packaging.version.Version(
        numpy.__version__) > packaging.version.Version(
            "1.20") and sys.version_info > (3, 8):
    NDArray = numpy.typing.NDArray  # pragma: no cover
    NDMaskedArray = numpy.ma.MaskedArray[Any, numpy.dtype[
        numpy.typing._generic_alias.ScalarType]]  # pragma: no cover
else:
    NDArray = numpy.ndarray  # pragma: no cover
    NDMaskedArray = numpy.ma.MaskedArray  # pragma: no cover

DTypeLike = numpy.typing.DTypeLike

ArrayLike = Union[dask.array.Array, numpy.typing.ArrayLike]
