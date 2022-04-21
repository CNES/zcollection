# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning a sequence of variables
====================================
"""
from typing import (
    ClassVar,
    Dict,
    Iterator,
    Optional,
    Sequence as _Sequence,
    Tuple,
)
import sys

import dask.array
import numpy

from . import abc
from ..typing import NDArray


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
    values = [arr[:, ix] for ix in reversed(range(arr.shape[1]))]
    sort_order = numpy.lexsort(numpy.array(values))
    return numpy.all(abc.difference(sort_order) > 0)  # type: ignore


class Sequence(abc.Partitioning):
    """Partitioning a sequence of variables.

    A sequence is a combination of variables constituting unique monotonic keys.
    For example, the orbit number (``cycle``) and the half-orbit number
    (``pass``) of a satellite.

    Args:
        variables: The sequence of variables constituting the partitioning.
        periodicity: The periodicity of each variable. The first value is
            ignored and always takes the value ``0``. The default value is
            None, which means that the periodicity is unknown
            (partitioning operates, but searching for next/previous partitions
            is not possible).
        dtype: The data type of the partitioning.

    Raises:
        ValueError: If the periodicity is not valid.

    Example:
        >>> partitioning = Sequence(["a", "b", "c"], (None, 10, 10))
    """
    __slots__ = ("periodicity", )

    #: The ID of the partitioning scheme.
    ID: ClassVar[str] = "Sequence"

    def __init__(self,
                 variables: _Sequence[str],
                 periodicity: Optional[_Sequence[int]] = None,
                 dtype: Optional[_Sequence[str]] = None) -> None:
        if periodicity is not None:
            if len(periodicity) != len(variables):
                raise ValueError(
                    "The number of variables and periodicity must "
                    "be the same.")
            if not all((item > 0) for item in periodicity[1:]):  # type: ignore
                raise ValueError("The periodicity must be positive")
            periodicity = (0, ) + tuple(periodicity[1:])
        self.periodicity: Optional[Tuple[int, ...]] = periodicity

        super().__init__(variables, dtype)

    @staticmethod
    def _split(
            variables: Dict[str, dask.array.Array]) -> Iterator[abc.Partition]:
        """Split the variables constituting the partitioning into partitioning
        schemes."""
        matrix = dask.array.vstack(tuple(variables.values())).transpose()
        if matrix.dtype.kind not in "iu":
            raise TypeError("The variables must be integer")

        index, indices = abc.unique(matrix)
        if not _is_monotonic(index):
            raise ValueError("index is not monotonic")

        indices = abc.concatenate_item(indices, matrix.shape[0])

        fields = tuple(variables.keys())
        if len(fields) == 1:
            concat = lambda fields, keys: (fields + keys, )
        else:
            concat = lambda fields, keys: tuple(zip(fields, keys))

        return ((concat(fields,
                        tuple(item)), slice(start, indices[ix + 1], None))
                for item, (ix, start) in zip(index, enumerate(indices[:-1])))

    def _before(
        self, partition_scheme: Tuple[Tuple[str, int], ...]
    ) -> Tuple[Tuple[str, int], ...]:
        """Return the previous partitioning scheme."""
        if self.periodicity is None:
            raise RuntimeError("Sequence periodicity is unknown.")
        values = list(value for _, value in partition_scheme)
        for ix in range(len(values) - 1, -1, -1):
            values[ix] -= 1
            if values[ix] >= 0:
                break
            values[ix] = self.periodicity[ix] - 1
        return tuple((variable, value)
                     for variable, value in zip(self.variables, values))

    def _after(
        self, partition_scheme: Tuple[Tuple[str, int], ...]
    ) -> Tuple[Tuple[str, int], ...]:
        """Return the next partitioning scheme."""
        if self.periodicity is None:
            raise RuntimeError("Sequence periodicity is unknown.")
        periodicity = (sys.maxsize, ) + self.periodicity[1:]
        values = list(value for _, value in partition_scheme)
        for ix in range(len(values) - 1, -1, -1):
            values[ix] += 1
            if values[ix] < periodicity[ix]:
                break
            values[ix] = 0
        return tuple((variable, value)
                     for variable, value in zip(self.variables, values))

    def encode(
        self,
        partition: Tuple[Tuple[str, int], ...],
    ) -> Tuple[int, ...]:
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
                     for _, value in self.parse(self.join(partition, "/")))

    def decode(self, values: Tuple[int, ...]) -> Tuple[Tuple[str, int], ...]:
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
