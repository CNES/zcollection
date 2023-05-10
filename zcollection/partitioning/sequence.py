# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning a sequence of variables
====================================
"""
from __future__ import annotations

from typing import Any, ClassVar, Iterator

import dask.array.core
import dask.array.routines
import numpy

from . import abc
from ..type_hints import ArrayLike, NDArray


def _is_monotonic(arr: NDArray) -> bool:
    """Check if the array is monotonic.

    The matrix will be sorted in the reverse order of the partitioning keys
    (column in the matrix). If the order of the matrix is unchanged, the
    different partitioning columns are monotonic.

    Args:
        arr: The array to check.

    Returns:
        True if the array is monotonic, False otherwise.
    """
    # `reversed` because `numpy.lexsort` wants the most significant key last.
    values: list[NDArray] = [
        arr[:, ix] for ix in reversed(range(arr.shape[1]))
    ]
    sort_order: NDArray = numpy.lexsort(numpy.array(values))
    return numpy.all(abc.difference(sort_order) > 0)  # type: ignore


def _unique(arr: ArrayLike, is_delayed: bool) -> tuple[NDArray, NDArray]:
    """Return unique elements and their indices.

    Args:
        arr: Array of elements.
        is_delayed: If True, the array is delayed.
    Returns:
        Tuple of unique elements and their indices.
    """
    index: NDArray
    indices: NDArray

    if is_delayed:
        index, indices = abc.unique(arr)  # type: ignore[arg-type]
        if not _is_monotonic(index):
            raise ValueError('index is not monotonic')
        return index, indices
    return abc.unique_and_check_monotony(arr)


class Sequence(abc.Partitioning):
    """Initialize a partitioning scheme for a sequence of variables.

    A sequence is a combination of variables constituting unique monotonic keys.
    For example, the orbit number (``cycle``) and the half-orbit number
    (``pass``) of a satellite.

    Args:
        variables: A list of strings representing the variables to be used for
            partitioning.
        dtype: An optional sequence of strings representing the data type used
            to store variable values in a binary representation without data
            loss. Must be one of the following allowed data types: ``int8``,
            ``int16``, ``int32``, ``int64``, ``uint8``, ``uint16``, ``uint32``,
            ``uint64``. If not provided, defaults to ``int64`` for all
            variables.

    Raises:
        ValueError: If the periodicity is not valid.

    Example:
        >>> partitioning = Sequence(["a", "b", "c"], (None, 10, 10))
    """
    #: The ID of the partitioning scheme.
    ID: ClassVar[str] = 'Sequence'

    # pylint: disable=arguments-differ
    # False positive: `self` is used in the signature.
    @staticmethod
    def _split(variables: dict[str, ArrayLike]) -> Iterator[abc.Partition]:
        """Split the variables constituting the partitioning into partitioning
        schemes."""
        index: NDArray
        indices: NDArray
        matrix: dask.array.core.Array | NDArray

        # Determine if the variables are handled by Dask.
        is_delayed: bool = any(
            isinstance(item, dask.array.core.Array)
            for item in variables.values())

        # Combines the arrays of variable values into a transposed matrix.
        matrix = dask.array.routines.vstack(tuple(
            variables.values())).transpose() if is_delayed else numpy.vstack(
                tuple(variables.values())).transpose()
        if matrix.dtype.kind not in 'iu':
            raise TypeError('The variables must be integer')

        index, indices = _unique(matrix, is_delayed)  # type: ignore[arg-type]
        indices = abc.concatenate_item(indices, matrix.shape[0])

        fields = tuple(variables.keys())
        # pylint: disable=unnecessary-lambda-assignment
        # We want to reference a lambda function, not assign it to a variable.
        if len(fields) == 1:
            concat: Any = lambda fields, keys: (fields + keys, )
        else:
            concat = lambda fields, keys: tuple(zip(fields, keys))
        # pylint: enable=unnecessary-lambda-assignment

        return ((concat(fields,
                        tuple(item)), slice(start, indices[ix + 1], None))
                for item, (ix, start) in zip(index, enumerate(indices[:-1])))
        # pylint: enable=arguments-differ

    def encode(
        self,
        partition: tuple[tuple[str, int], ...],
    ) -> tuple[int, ...]:
        """Encode a partitioning scheme to the handled values.

        Args:
            partition: The partitioning scheme to be encoded.

        Returns:
            The encoded partitioning scheme.

        Example:
            >>> partitioning = Sequence(["a", "b", "c"])
            >>> fields = partitioning.parse("a=100/b=10/c=1")
            >>> fields
            (('a', 100), ('b', 10), ('c', 1))
            >>> partitioning.encode(fields)
            (100, 10, 1)
        """
        return tuple(value
                     for _, value in self.parse(self.join(partition, '/')))

    def decode(self, values: tuple[int, ...]) -> tuple[tuple[str, int], ...]:
        """Decode a partitioning scheme.

        Args:
            values: The encoded partitioning scheme.

        Returns:
            The decoded partitioning scheme.

        Example:
            >>> partitioning = Sequence(["a", "b", "c"])
            >>> partitioning.decode((100, 10, 1))
            (('a', 100), ('b', 10), ('c', 1))
        """
        return tuple(
            (key, value) for key, value in zip(self.variables, values))
