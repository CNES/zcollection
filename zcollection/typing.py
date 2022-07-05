# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Typing
======
"""
from typing import Any, Protocol, Tuple, TypeVar

import numpy
import numpy.typing

# pylint: disable=invalid-name
_DType_co = TypeVar("_DType_co", covariant=True, bound="numpy.dtype[Any]")
_ScalarType_co = TypeVar("_ScalarType_co", bound=numpy.generic, covariant=True)
# pylint: enable=invalid-name

#: A numpy tensor with any type.
NDArray = numpy.typing.NDArray  # pragma: no cover

#: A numpy masked tensor with any type.
NDMaskedArray = numpy.ma.MaskedArray[
    Any, numpy.dtype[_ScalarType_co]]  # pragma: no cover

#: Anything that can be coerced into numpy.dtype.
DTypeLike = numpy.typing.DTypeLike  # pragma: no cover

#: numpy.dtype.
DType = numpy.dtype[_ScalarType_co]  # pragma: no cover


class ArrayLike(Protocol[_DType_co]):
    """Protocol for array-like objects."""

    def __array__(self) -> numpy.ndarray[Any, _DType_co]:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the array."""
        # pylint: disable=unnecessary-ellipsis
        # Make checker happy.
        ...
        # pylint: enable=unnecessary-ellipsis
