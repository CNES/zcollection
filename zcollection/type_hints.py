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

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

try:
    from types import GenericAlias  # type: ignore[attr-defined]
except ImportError:
    # pylint: disable=ungrouped-imports
    # For Python < 3.9 we use a backport of GenericAlias provided by
    # numpy
    # isort: off
    from numpy._typing._generic_alias import (  # type: ignore[assignment,misc]
        _GenericAlias as GenericAlias,  # yapf: disable
    )
    # isort: on
    # pylint: enable=ungrouped-imports

try:
    from typing_extensions import TypeAlias
except ImportError:
    # pylint: disable=ungrouped-imports
    # TypeAlias is defined in typing starting from 3.10
    from typing import TypeAlias  # type: ignore[attr-defined,no-redef]
    # pylint: enable=ungrouped-imports

import numpy
import numpy.typing

# pylint: disable=invalid-name
_DType_co = TypeVar('_DType_co', covariant=True, bound='numpy.dtype[Any]')
_ScalarType_co = TypeVar('_ScalarType_co', bound=numpy.generic, covariant=True)
# pylint: enable=invalid-name

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
    def shape(self) -> tuple[int, ...]:
        """The shape of the array."""
        # pylint: disable=unnecessary-ellipsis
        # Make checker happy.
        ...
        # pylint: enable=unnecessary-ellipsis
