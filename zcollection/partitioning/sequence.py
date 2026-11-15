# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning a sequence of variables
====================================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import dask.array.core
import dask.array.routines
import numpy

from . import abc

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence as SequenceType

    from ..type_hints import ArrayLike, NDArray

#: Minimum size of a grouped sequence.
MIN_GROUP_SIZE: int = 2


def _is_monotonic(arr: NDArray) -> bool:
    """Check whether records are ordered by the partitioning keys.

    The columns represent partitioning keys. If lexicographic sorting leaves
    their row order unchanged, the keys are monotonic.

    Args:
        arr: The array to check.

    Returns:
        Whether the rows are already in partition-key order.
    """
    # `reversed` because `numpy.lexsort` wants the most significant key last.
    values: list[NDArray] = [
        arr[:, ix] for ix in reversed(range(arr.shape[1]))
    ]
    sort_order: NDArray = numpy.lexsort(numpy.array(values))
    return bool(numpy.all(abc.difference(sort_order) > 0))


def _unique(arr: ArrayLike, is_delayed: bool) -> tuple[NDArray, NDArray]:
    """Return unique partition keys and their first indices.

    Args:
        arr: The array of partition keys.
        is_delayed: Whether the array uses Dask.

    Returns:
        The unique keys and their indices.

    Raises:
        ValueError: If the partition keys are not monotonic.
    """
    index: NDArray
    indices: NDArray

    if is_delayed:
        index, indices = abc.unique_dask(arr)  # type: ignore[arg-type]
    else:
        index, indices = abc.unique_numpy(arr)  # type: ignore[arg-type]
    if not _is_monotonic(index):
        raise ValueError('index is not monotonic')
    return index, indices


class Sequence(abc.Partitioning):
    """Partition a dataset using a sequence of variables.

    The variables define monotonic partition keys, such as the ``cycle``
    and ``pass`` numbers of a satellite.

    Args:
        variables: Names of the partitioning variables.
        dtype: Optional data types used to store partition keys. Supported
            types are ``int8``, ``int16``, ``int32``, ``int64``, ``uint8``,
            ``uint16``, ``uint32`` and ``uint64``. Defaults to ``int64``
            for every variable.

    Raises:
        TypeError: If ``dtype`` is a string instead of a sequence.
        ValueError: If no variables are given or a data type is unsupported.

    Example:
        >>> partitioning = Sequence(["a", "b", "c"],
        ...                        ("int64", "int32", "int16"))
    """
    #: The ID of the partitioning scheme.
    ID: ClassVar[str] = 'Sequence'

    def _split(self, variables: dict[str, NDArray]) -> Iterator[abc.Partition]:
        """Split monotonic integer variables into partition keys."""
        index: NDArray
        indices: NDArray
        matrix: dask.array.core.Array | NDArray

        # Check whether the variables use Dask.
        is_delayed: bool = any(
            isinstance(item, dask.array.core.Array)
            for item in variables.values())

        # Combine variable arrays into a matrix with one row per record.
        matrix = dask.array.routines.vstack(tuple(
            variables.values())).transpose() if is_delayed else numpy.vstack(
                tuple(variables.values())).transpose()
        if matrix.dtype.kind not in 'iu':
            raise TypeError('The variables must be integer')

        index, indices = _unique(matrix, is_delayed)  # type: ignore[arg-type]
        indices = abc.concatenate_item(indices, matrix.shape[0])

        fields = tuple(variables.keys())

        # Pass the lambda directly rather than assigning it a name.
        if len(fields) == 1:

            def concat(fields, keys):
                return (fields + keys, )
        else:

            def concat(fields, keys):
                return tuple(zip(fields, keys, strict=True))

        return ((concat(fields,
                        tuple(item)), slice(start, indices[ix + 1], None))
                for item, (
                    ix,
                    start) in zip(index, enumerate(indices[:-1]), strict=True))

    def encode(
        self,
        partition: tuple[tuple[str, int], ...],
    ) -> tuple[int, ...]:
        """Encode partition keys into their values.

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
        """Decode values into partition keys.

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
            (key, value)
            for key, value in zip(self.variables, values, strict=True))


class GroupedSequence(Sequence):
    """Partition a sequence by grouping values of its last variable.

    The partitioning variables define monotonic keys. Values of the last
    variable are grouped into ranges of ``size``, aligned to ``start``.

    Args:
        variables: Names of the partitioning variables.
        size: Width of each group along the last variable. Must be at least 2;
            use :class:`Sequence` for individual values.
        start: Value used to align the groups. Defaults to 0.
        dtype: An optional sequence of strings representing the data type used
            to store partition keys without data loss. Group lower bounds
            must fit the last declared type; use a signed type if they can be
            negative. Must be one of the following allowed data types: ``int8``,
            ``int16``, ``int32``, ``int64``, ``uint8``, ``uint16``, ``uint32``,
            ``uint64``. If not provided, defaults to ``int64`` for all
            variables.
    """

    __slots__ = ('_dtype', '_pattern', 'size', 'start', 'variables')

    #: The ID of the partitioning scheme.
    ID: ClassVar[str] = 'GroupedSequence'

    def __init__(
        self,
        variables: SequenceType[str],
        size: int,
        start: int = 0,
        dtype: SequenceType[str] | None = None,
    ) -> None:
        if size < MIN_GROUP_SIZE:
            raise ValueError(
                f"Parameter 'size' must be at least {MIN_GROUP_SIZE} ('{size}' "
                'was provided).')

        self.size = size
        self.start = start

        super().__init__(variables, dtype)

    def _split(
        self,
        variables: dict[str, NDArray],
    ) -> Iterator[abc.Partition]:
        """Split the last variable into groups and build partition keys."""
        last_field = list(variables)[-1]
        values = variables[last_field]
        offset = self.start % self.size
        key_dtype = self._dtype[-1]
        limits = numpy.iinfo(key_dtype)

        variables = variables.copy()
        # Shift group numbers by one to keep unsigned arithmetic nonnegative.
        variables[last_field] = values // self.size + (values % self.size
                                                       >= offset)

        for fields, indexer in super()._split(variables):
            # Calculate bounds as Python integers to preserve uint64 precision.
            group_number = int(fields[-1][1])
            lower_bound = (group_number - 1) * self.size + offset
            if not limits.min <= lower_bound <= limits.max:
                raise ValueError(
                    f"The lower bound {lower_bound} of {last_field!r} "
                    f"does not fit the declared {key_dtype} partition key "
                    'dtype.')
            yield (*fields[:-1], (last_field, lower_bound)), indexer
