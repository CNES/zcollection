# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning a sequence of variables
====================================
"""
from typing import ClassVar, Dict, Iterator, Tuple

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
        variables:  List of variables to be used for partitioning.

    Example:
        >>> partitioning = Sequence(["a", "b", "c"])
    """
    #: The ID of the partitioning scheme.
    ID: ClassVar[str] = "Sequence"

    @staticmethod
    def _split(
            variables: Dict[str, dask.array.Array]) -> Iterator[abc.Partition]:
        """Split the variables constituting the partitioning into partitioning
        schemes."""
        matrix = dask.array.vstack(tuple(variables.values())).transpose()
        if matrix.dtype.kind != "i":
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
