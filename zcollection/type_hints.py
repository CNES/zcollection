# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Type hints for the zcollection package.
=======================================

.. rubric:: Type aliases

.. py:data:: DType
    :canonical: DType

    Type of a numpy array.

.. py:data:: DTypeLike
    :canonical: DTypeLike

    Type of a numpy array or a string.

.. py:data:: NDArray
    :canonical: NDArray

    Type of a numpy array.

.. py:data:: NDMaskedArray

    Type of a numpy array with a mask.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar
from types import GenericAlias  # type: ignore[attr-defined]

import numpy.typing

_DType_co = TypeVar('_DType_co', covariant=True, bound='numpy.dtype[Any]')
_ScalarType_co = TypeVar('_ScalarType_co', bound=numpy.generic, covariant=True)

if TYPE_CHECKING:
    DType = numpy.dtype[_ScalarType_co]
    NDMaskedArray = numpy.ma.MaskedArray[Any, DType]  # pragma: no cover
else:
    DType = GenericAlias(numpy.dtype, (_ScalarType_co, ))
    NDMaskedArray = GenericAlias(numpy.ma.MaskedArray, (Any, DType))

NDArray: TypeAlias = numpy.typing.NDArray  # pragma: no cover
DTypeLike: TypeAlias = numpy.typing.DTypeLike  # pragma: no cover


class ArrayLike(Protocol[_DType_co]):
    """Protocol for array-like objects."""

    def __array__(self) -> NDArray:
        ...

    @property
    def dtype(self) -> DType:
        """The data type of the array."""

        # Make checker happy.
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array."""

        # Make checker happy.
        ...

    @property
    def size(self) -> int:
        """The size of the array."""

        # Make checker happy.
        ...

    def astype(self, dtype: DTypeLike) -> ArrayLike[_DType_co]:
        """Convert the array to a given type."""

        # Make checker happy.
        ...
