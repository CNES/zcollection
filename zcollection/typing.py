# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Typing
======
"""
from typing import TYPE_CHECKING, Any, Protocol, Tuple, TypeVar

try:
    from types import GenericAlias  # type: ignore[attr-defined]
except ImportError:
    # pylint: disable=ungrouped-imports
    # For Python < 3.9 we use a backport of GenericAlias provided by
    # numpy
    # isort: off
    from numpy._typing._generic_alias import (  # type: ignore[misc,no-redef]
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
    #: numpy.dtype.
    DType = numpy.dtype[_ScalarType_co]
    #: A numpy masked tensor with any type.
    NDMaskedArray = numpy.ma.MaskedArray[Any, DType]  # pragma: no cover
else:
    DType = GenericAlias(numpy.dtype, (_ScalarType_co, ))
    NDMaskedArray = GenericAlias(numpy.ma.MaskedArray, (Any, DType))

#: A numpy tensor with any type.
NDArray: TypeAlias = numpy.typing.NDArray  # pragma: no cover

#: Anything that can be coerced into numpy.dtype.
DTypeLike: TypeAlias = numpy.typing.DTypeLike  # pragma: no cover


class ArrayLike(Protocol[_DType_co]):
    """Protocol for array-like objects."""

    def __array__(self) -> NDArray:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the array."""
        # pylint: disable=unnecessary-ellipsis
        # Make checker happy.
        ...
        # pylint: enable=unnecessary-ellipsis
