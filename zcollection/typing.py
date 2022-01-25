# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Numpy typings
=============
"""
import distutils.version
import sys
from typing import TYPE_CHECKING, Any, Union

import dask.array
import numpy
import numpy.typing

#: An array object represents a multidimensional, homogeneous array of
#: fixed-size items.
if not TYPE_CHECKING or distutils.version.LooseVersion(
        numpy.__version__) < distutils.version.LooseVersion(
            "1.20") or sys.version_info < (3, 9):
    NDArray = numpy.ndarray  # pragma: no cover
    NDMaskedArray = numpy.ma.MaskedArray  # pragma: no cover
else:
    NDArray = numpy.typing.NDArray  # pragma: no cover
    NDMaskedArray = numpy.ma.MaskedArray[Any, numpy.dtype[
        numpy.typing._generic_alias.ScalarType]]  # pragma: no cover

#: Anything that can be coerced into numpy.dtype.
DTypeLike = numpy.typing.DTypeLike

#: dask.array.Array, numpy.typing.ArrayLike
ArrayLike = Union[dask.array.Array, numpy.typing.ArrayLike]
