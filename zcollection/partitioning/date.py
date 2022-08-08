# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partitioning by date
====================
"""
from typing import Any, ClassVar, Dict, Iterator, Sequence, Tuple

import dask.array.core
import numpy

from . import abc

#: Numpy time units
RESOLUTION = ('Y', 'M', 'D', 'h', 'm', 's')

#: Numpy time unit meanings
UNITS = ('year', 'month', 'day', 'hour', 'minute', 'second')

#: Data type for time units
DATA_TYPES = ('uint16', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8')

#: Time separation units
SEPARATORS = dict(year='-',
                  month='-',
                  day='T',
                  hour=':',
                  minute=':',
                  second='.')


class Date(abc.Partitioning):
    """Date partitioning.

    Args:
        variables: Variable names used for the partitioning.
        resolution: Time resolution of the partitioning.

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
        index = RESOLUTION.index(resolution) + 1

        #: The time resolution of the partitioning
        self.resolution = resolution
        #: The time parts used for the partitioning
        self._attrs = UNITS[:index + 1]
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
        selection: Tuple[Tuple[str, Any], ...],
    ) -> Tuple[str, ...]:
        """Return the partitioning scheme for the given selection."""
        _, datetime64 = selection[0]
        datetime = datetime64.astype('M8[s]').item()
        return tuple(UNITS[ix] + '=' +
                     f'{getattr(datetime, self._attrs[ix]):02d}'
                     for ix in self._index)
        # pylint: enable=arguments-differ

    def _split(
        self,
        variables: Dict[str, dask.array.core.Array],
    ) -> Iterator[abc.Partition]:
        """Return the partitioning scheme for the given variables."""
        name, values = tuple(variables.items())[0]

        if not numpy.issubdtype(values.dtype, numpy.dtype('datetime64')):
            raise TypeError('values must be a datetime64 array')

        index, indices = abc.unique(
            values.astype(f'datetime64[{self.resolution}]'))

        # We don't use here the function `numpy.diff` but `abc.difference` for
        # optimization purposes.
        if not numpy.all(
                abc.difference(index.view(numpy.int64)) >= 0):  # type: ignore
            raise ValueError('index is not monotonic')

        indices = abc.concatenate_item(indices, values.size)

        return ((((name, date), ), slice(start, indices[ix + 1], None))
                for date, (ix, start) in zip(index, enumerate(indices[:-1])))

    @staticmethod
    def _stringify(partition: Tuple[Tuple[str, int], ...]) -> str:
        """Return a string representation of the partitioning scheme."""
        string = ''.join(f'{value:02d}' + SEPARATORS[item]
                         for item, value in partition)
        if string[-1] in SEPARATORS.values():
            string = string[:-1]
        return string

    @staticmethod
    def join(partition_scheme: Tuple[Tuple[str, int], ...], sep: str) -> str:
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
        partition: Tuple[Tuple[str, int], ...],
    ) -> Tuple[Any, ...]:
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
        values: Tuple[Any, ...],
    ) -> Tuple[Tuple[str, int], ...]:
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
        datetime64, = values
        datetime = datetime64.astype('M8[s]').item()
        return tuple((UNITS[ix], getattr(datetime, self._attrs[ix]))
                     for ix in self._index)
