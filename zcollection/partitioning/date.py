# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning by date
====================
"""
from __future__ import annotations

from typing import Any, ClassVar, Iterator, Sequence
import datetime

import dask.array.core
import numpy

from . import abc
from ..type_hints import ArrayLike, NDArray

#: Numpy time units
RESOLUTION = ('Y', 'M', 'D', 'h', 'm', 's')

#: Numpy time unit meanings
UNITS = ('year', 'month', 'day', 'hour', 'minute', 'second')

#: Data type for time units
DATA_TYPES = ('uint16', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8')

#: Time separation units
SEPARATORS: dict[str, str] = {
    'year': '-',
    'month': '-',
    'day': 'T',
    'hour': ':',
    'minute': ':',
    'second': '.'
}


def _unique(arr: ArrayLike, is_delayed: bool) -> tuple[NDArray, NDArray]:
    """Return unique elements and their indices.

    Args:
        arr: Array of elements.
        is_delayed: If True, the array is delayed.
    Returns:
        Tuple of unique elements and their indices.
    Raises:
        ValueError: If the array is not monotonic.
    """
    index: NDArray
    indices: NDArray

    if is_delayed:
        index, indices = abc.unique(arr)  # type: ignore[arg-type]
        # We don't use here the function `numpy.diff` but `abc.difference` for
        # optimization purposes.
        if not numpy.all(
                abc.difference(index.view(numpy.int64)) >= 0):  # type: ignore
            raise ValueError('index is not monotonic')
        return index, indices
    return abc.unique_and_check_monotony(arr)


class Date(abc.Partitioning):
    """Initialize a partitioning scheme based on dates.

    Args:
        variables: A list of strings representing the variables to be used for
            partitioning.
        resolution: Time resolution of the partitioning. Must be in
            :data:`RESOLUTION`.

    Raises:
        ValueError: If the resolution is not in the list of supported
            resolutions or if the partitioning is not performed on a one
            dimensional variable.

    Example:
        >>> partitioning = Date(variables=("time", ), resolution="Y")
    """
    __slots__ = ('_attrs', '_index', 'resolution')

    #: The ID of the partitioning scheme
    ID: ClassVar[str] = 'Date'

    def __init__(self, variables: Sequence[str], resolution: str) -> None:
        if len(variables) != 1:
            raise ValueError(
                'Partitioning on dates is performed on a single variable.')
        if resolution not in RESOLUTION:
            raise ValueError('resolution must be in: ' + ', '.join(RESOLUTION))
        index: int = RESOLUTION.index(resolution) + 1

        #: The time resolution of the partitioning
        self.resolution: str = resolution
        #: The time parts used for the partitioning
        self._attrs: tuple[str, ...] = UNITS[:index + 1]
        #: The indices of the time parts used for the partitioning
        self._index = tuple(range(index))
        super().__init__(variables,
                         tuple(DATA_TYPES[ix] for ix in self._index))

    def _keys(self) -> Sequence[str]:
        """Return the keys of the partitioning scheme."""
        return tuple(UNITS[ix] for ix in self._index)

    # pylint: disable=arguments-differ
    # False positive: the base method is static.
    def _partition(  # type: ignore[override]
        self,
        selection: tuple[tuple[str, Any], ...],
    ) -> tuple[str, ...]:
        """Return the partitioning scheme for the given selection."""
        datetime64: NDArray = selection[0][1]
        py_datetime: datetime.datetime = datetime64.astype('M8[s]').item()
        return tuple(UNITS[ix] + '=' +
                     f'{getattr(py_datetime, self._attrs[ix]):02d}'
                     for ix in self._index)
        # pylint: enable=arguments-differ

    def _split(
        self,
        variables: dict[str, ArrayLike],
    ) -> Iterator[abc.Partition]:
        """Return the partitioning scheme for the given variables."""
        index: NDArray
        indices: NDArray
        name: str
        values: ArrayLike

        # Determine if the variables are handled by Dask.
        is_delayed: bool = any(
            isinstance(value, dask.array.core.Array)
            for value in variables.values())
        name, values = tuple(variables.items())[0]

        if not numpy.issubdtype(values.dtype, numpy.dtype('datetime64')):
            raise TypeError('values must be a datetime64 array')

        index, indices = _unique(
            values.astype(f'datetime64[{self.resolution}]'), is_delayed)
        indices = abc.concatenate_item(indices, values.size)

        return ((((name, date), ), slice(start, indices[ix + 1], None))
                for date, (ix, start) in zip(index, enumerate(indices[:-1])))

    @staticmethod
    def _stringify(partition: tuple[tuple[str, int], ...]) -> str:
        """Return a string representation of the partitioning scheme."""
        string = ''.join(f'{value:02d}' + SEPARATORS[item]
                         for item, value in partition)
        if string[-1] in SEPARATORS.values():
            string = string[:-1]
        return string

    @staticmethod
    def join(partition_scheme: tuple[tuple[str, int], ...], sep: str) -> str:
        """Join a partitioning scheme.

        Args:
            partition_scheme: The partitioning scheme to be joined.
            sep: The separator to be used.

        Returns:
            The joined partitioning scheme.

        Example:
            >>> partitioning = Date(variables=("time", ), resolution="D")
            >>> partitioning.join((("year", 2020), ("month", 1), ("day", 1)),
            ...                   "/")
            'year=2020/month=01/day=01'
        """
        return sep.join(f'{k}={v:02d}' for k, v in partition_scheme)

    def encode(
        self,
        partition: tuple[tuple[str, int], ...],
    ) -> tuple[Any, ...]:
        """Encode a partitioning scheme.

        Args:
            partition: The partitioning scheme to be encoded.

        Returns:
            The encoded partitioning scheme.

        Example:
            >>> partitioning = Date(variables=("time", ), resolution="D")
            >>> fields = partitioning.parse("year=2020/month=01/day=01")
            >>> fields
            (("year", 2020), ("month", 1), ("day", 1))
            >>> partitioning.encode(fields)
            (numpy.datetime64('2020-01-01'),)
        """
        return tuple((numpy.datetime64(self._stringify(partition)), ))

    def decode(
        self,
        values: tuple[Any, ...],
    ) -> tuple[tuple[str, int], ...]:
        """Decode a partitioning scheme.

        Args:
            values: The partitioning scheme to be decoded.

        Returns:
            The decoded partitioning scheme.

        Example:
            >>> partitioning = Date(variables=("time", ), resolution="D")
            >>> partitioning.decode((numpy.datetime64('2020-01-01'), ))
            (("year", 2020), ("month", 1), ("day", 1))
        """
        datetime64: NDArray = values[0]
        py_datetime: datetime.datetime = datetime64.astype('M8[s]').item()
        return tuple((UNITS[ix], getattr(py_datetime, self._attrs[ix]))
                     for ix in self._index)
