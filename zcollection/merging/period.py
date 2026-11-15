# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Time period
===========
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal
import enum
import re
from re import Match

import numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..type_hints import DType

# Parse the unit of NumPy datetime64 and timedelta64 types.
PATTERN: Callable[[str], Match[str] | None] = re.compile(
    r'(?:datetime|timedelta)64\[(\w+)\]').search

#: NumPy time units.
RESOLUTION: Final = ('as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D',
                     'W', 'M', 'Y')


def _time64_unit(dtype: DType[Any]) -> str:
    """Return the unit of a NumPy datetime64 or timedelta64."""
    match: Match[str] | None = PATTERN(dtype.name)
    if match is None:
        raise ValueError(f'dtype is not a time duration: {dtype}')
    return match.group(1)


def _min_time64_unit(
    *args: DType[Any]
) -> Literal['as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M',
             'Y']:
    """Return the smaller of two time units."""
    index = min(RESOLUTION.index(_time64_unit(item)) for item in args)
    return RESOLUTION[index]


class PeriodRelation(enum.IntEnum):
    """Relations between two periods."""
    __slots__ = ()

    #: The first period starts at or after the second period's end.
    #:
    #: .. code-block:: text
    #:
    #:             [-----p1-----]
    #:      [-p2-]
    AFTER = 0

    #: The first period starts at the last element of the second period.
    #:
    #: .. code-block:: text
    #:
    #:           [-----p1-----]
    #:      [-p2-]
    START_TOUCHING = 1

    #: The first period starts within the second.
    #:
    #: .. code-block:: text
    #:
    #:        [-----p1-----]
    #:      [-p2-]
    START_INSIDE = 2

    #: Both periods start together, and the first ends within the second.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:      [-------p2-------]
    INSIDE_START_TOUCHING = 3

    #: Both periods start together, and the first contains the second.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:      [--p2--]
    ENCLOSING_START_TOUCHING = 4

    #: The first period contains the second.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:         [--p2--]
    ENCLOSING = 5

    #: The first period contains the second, and both end together.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:            [--p2--]
    ENCLOSING_END_TOUCHING = 6

    #: The two periods are exactly the same.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:      [-----p2-----]
    EXACT_MATCH = 7

    #: The first period is contained within the second.
    #:
    #: .. code-block:: text
    #:
    #:        [---p1---]
    #:      [-----p2-----]
    INSIDE = 8

    #: The first period is contained within the second, and both end
    #: together.
    #:
    #: .. code-block:: text
    #:
    #:          [---p1---]
    #:      [-----p2-----]
    INSIDE_END_TOUCHING = 9

    #: The first period ends within the second.
    #:
    #: .. code-block:: text
    #:
    #:      [---p1---]
    #:            [-----p2-----]
    END_INSIDE = 10

    #: The second period starts at the last element of the first period.
    #:
    #: .. code-block:: text
    #:
    #:      [---p1---]
    #:               [-----p2-----]
    END_TOUCHING = 11

    #: The first period is before the second period.
    #:
    #: .. code-block:: text
    #:
    #:      [---p1---]
    #:                  [-----p2-----]
    BEFORE = 12

    def is_after(self) -> bool:
        """Return whether the first period is after the second."""

        return self.value == PeriodRelation.AFTER

    def is_before(self) -> bool:
        """Return whether the first period is before the second."""

        return self.value == PeriodRelation.BEFORE

    def contains(self) -> bool:
        """Return whether the first period contains or equals the second."""
        return self.value in (PeriodRelation.ENCLOSING_END_TOUCHING,
                              PeriodRelation.ENCLOSING_START_TOUCHING,
                              PeriodRelation.ENCLOSING,
                              PeriodRelation.EXACT_MATCH)

    def is_before_overlapping(self) -> bool:
        """Return True for ``END_INSIDE``, ``END_TOUCHING`` or
        ``INSIDE_START_TOUCHING``."""
        return self.value in (PeriodRelation.END_INSIDE,
                              PeriodRelation.END_TOUCHING,
                              PeriodRelation.INSIDE_START_TOUCHING)

    def is_after_overlapping(self) -> bool:
        """Return True for ``START_INSIDE``, ``START_TOUCHING`` or
        ``INSIDE_END_TOUCHING``."""
        return self.value in (PeriodRelation.START_INSIDE,
                              PeriodRelation.START_TOUCHING,
                              PeriodRelation.INSIDE_END_TOUCHING)

    def is_inside(self) -> bool:
        """Return whether the first period lies strictly inside the second."""

        return self.value == PeriodRelation.INSIDE


class Period:
    """Create a period from ``begin`` to ``end``.

    By default, the period includes ``begin`` but excludes ``end``.
    Set ``within=True`` to include ``end``.

    Args:
        begin: The beginning of the period.
        end: The end of the period.
        within: Whether to include ``end``. Defaults to ``False``.
    """
    __slots__: tuple[str, ...] = ('_begin', '_duration_unit', '_last')

    def __init__(self,
                 begin: numpy.datetime64,
                 end: numpy.datetime64,
                 *,
                 within: bool = False) -> None:
        duration_unit = _min_time64_unit(begin.dtype, end.dtype)

        #: The beginning of the period.
        self._begin: numpy.datetime64 = begin
        #: The duration unit of the period.
        self._duration_unit: numpy.timedelta64 = numpy.timedelta64(
            1, duration_unit)
        #: The last date of the period.
        self._last: numpy.datetime64 = (end if within else end -
                                        self._duration_unit)

    def __repr__(self) -> str:
        return f'[{self._begin}, {self.end()}['

    def __getstate__(self) -> tuple[Any, ...]:
        return self._begin, self._duration_unit, self._last

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self._begin, self._duration_unit, self._last = state

    @classmethod
    def from_duration(cls, begin: numpy.datetime64,
                      duration: numpy.timedelta64) -> Period:
        """Create a period from ``begin`` to ``begin + duration``, exclusive.

        Args:
            begin: The beginning of the period.
            duration: The duration of the period.

        Returns:
            The created period.
        """
        return cls(begin, begin + duration)

    @property
    def begin(self) -> numpy.datetime64:
        """Return the first element in the period."""
        return self._begin

    @property
    def last(self) -> numpy.datetime64:
        """Return the last item in the period."""
        return self._last

    def end(self) -> numpy.datetime64:
        """Return one past the last element."""
        return self._last + self._duration_unit

    def is_null(self) -> bool:
        """Return whether the period has zero or negative length."""
        return bool(self.end() <= self._begin)  # numpy.bool_ -> bool

    def length(self) -> numpy.timedelta64:
        """Return the length of the period."""
        if self._last < self._begin:
            # invalid period
            return self._last + self._duration_unit - self._begin
        return self.end() - self._begin

    def __eq__(self, rhs: object) -> bool:
        if not isinstance(rhs, Period):
            return NotImplemented
        return (self._begin == rhs._begin) and (self._last == rhs._last)

    def __ne__(self, rhs: object) -> bool:
        if not isinstance(rhs, Period):
            return NotImplemented
        return (self._begin != rhs._begin) or (self._last != rhs._last)

    def __lt__(self, rhs: Any) -> bool:
        if not isinstance(rhs, Period):
            return NotImplemented
        return bool(self._last < rhs._begin)  # numpy.bool_ -> bool

    def shift(self, duration: numpy.timedelta64) -> None:
        """Shift the start and end by the specified amount.

        Args:
            duration: The amount to shift the period.
        """
        self._begin = self._begin + duration
        self._last = self._last + duration

    def expand(self, duration: numpy.timedelta64) -> None:
        """Expand the period by ``duration`` at both ends.

        So before expand:

        .. code-block:: text

                    [-------]
            ^   ^   ^   ^   ^   ^  ^
            1   2   3   4   5   6  7

        After expand(2):

        .. code-block:: text

            [----------------------]
            ^   ^   ^   ^   ^   ^  ^
            1   2   3   4   5   6  7

        Args:
            duration: The amount to expand the period.
        """
        self._begin = self._begin - duration
        self._last = self._last + duration

    def contains(
        self,
        other: numpy.datetime64 | Period,
    ) -> bool:
        """Return whether this period contains a date or another period.

        Args:
            other: The date or period to check.

        Returns:
            ``True`` if the date lies within this period, or if this period
            contains or equals the other period. A zero-length period
            contains no dates.
        """
        if isinstance(other, numpy.datetime64):
            return bool(self._begin <= other <= self._last)
        return bool((self._begin <= other.begin)
                    and (self._last >= other.last))

    def is_adjacent(self, other: Period) -> bool:
        """Return whether two periods are adjacent without a gap or overlap.

        In the example, p1 and p2 are adjacent, but p3 is not adjacent
        to either of them.

        .. code-block:: text

            [-p1-[
                 [-p2-]
                   [-p3-]

        Args:
            other: The other period to check.

        Returns:
            ``True`` if ``other`` is a period adjacent to this one.
        """
        return other.begin == self.end() or self.begin == other.end()

    def is_after(self, point: numpy.datetime64) -> bool:
        """Return whether the period starts after the given point.

        In the example below only point 1 would evaluate to true.

        .. code-block:: text

                [----------[
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The point to check.

        Returns:
            ``True`` if the point precedes the beginning of the period.
        """
        if self.is_null():
            # null period isn't after
            return False
        return bool(point < self._begin)

    def is_before(self, point: numpy.datetime64) -> bool:
        """Return whether the last element of the period precedes the point.

        In the example below points 4 and 5 return true.

        .. code-block:: text

                [----------[
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The point to check.

        Returns:
            ``True`` if the point follows the last element of the period.
        """
        if self.is_null():
            # null period isn't before anything
            return False
        return bool(self._last < point)

    def intersects(self, other: Period) -> bool:
        """Return whether the periods overlap.

        In the example below p1 intersects with p2, p4, and p6.

        .. code-block:: text

                  [---p1---[
                        [---p2---[
                           [---p3---[
             [---p4---[
            [-p5-[
                    [-p6-[

        Args:
            other: The other period to check.

        Returns:
            Whether the periods intersect.
        """
        return bool(
            self.contains(other.begin) or other.contains(self._begin)
            or ((other.begin < self._begin) and (other.last >= self._begin)))

    def intersection(self, other: Period) -> Period:
        """Return the intersection, or a null period if there is none.

        Args:
            other: The other period to check.

        Returns:
            The intersecting period, or a null period.
        """
        if self._begin > other.begin:
            if self._last <= other.last:
                return self
            return Period(self._begin, other.end())

        if self._last <= other.last:
            return Period(other.begin, self.end())
        return other

    def merge(self, other: Period) -> Period:
        """Return the union of intersecting periods, or a null period.

        Args:
            other: The other period to merge.

        Returns:
            The combined period, or a null period if the periods do not
            intersect.
        """
        if self.intersects(other):
            if self._begin < other.begin:
                return Period(
                    self._begin,
                    self.end() if self._last > other.last else other.end())
            return Period(
                other.begin,
                self.end() if self._last > other.last else other.end())
        # Return a null period if there is no intersection.
        return Period(self._begin, self._begin)

    def span(self, other: Period) -> Period:
        """Return the shortest period spanning both periods, including any gap.

        The resulting period starts at the earlier beginning and ends at the
        later end:

          * start = min(p1.start, p2.start)
          * end   = max(p1.end  , p2.end)

        .. code-block:: text

            [---p1---]
                           [---p2---]

        result:

        .. code-block:: text

            [-----------p3----------]

        Args:
            other: The other period to combine.

        Returns:
            The combined period.
        """
        start: numpy.datetime64 = (self._begin if self._begin < other.begin
                                   else other.begin)
        end: numpy.datetime64 = (other.end()
                                 if self._last < other.last else self.end())
        return Period(start, end)

    # Each return handles a distinct relation between the two periods.
    def _get_direct_relation(  # noqa: PLR0911
            self,
            other: Period,
    ) -> PeriodRelation | None:
        """Get the direct relation between two periods."""
        if other.last < self._begin:
            return PeriodRelation.AFTER

        if other.begin > self._last:
            return PeriodRelation.BEFORE

        if other.begin == self._begin and other.last == self._last:
            return PeriodRelation.EXACT_MATCH

        if other.last == self._begin:
            return PeriodRelation.START_TOUCHING

        if other.begin == self._last:
            return PeriodRelation.END_TOUCHING

        if self.contains(other):
            if other.begin == self.begin:
                return PeriodRelation.ENCLOSING_START_TOUCHING
            return (PeriodRelation.ENCLOSING_END_TOUCHING
                    if other.last == self.last else PeriodRelation.ENCLOSING)

        return None

    def get_relation(self, other: Period) -> PeriodRelation:
        """Return the relation between this period and another.

        Args:
            other: The other period.

        Returns:
            The relation.
        """
        relation: PeriodRelation | None = self._get_direct_relation(other)
        if relation is not None:
            return relation

        period_contains_start: bool = other.contains(self._begin)
        period_contains_end: bool = other.contains(self._last)

        if period_contains_start and period_contains_end:
            if other.begin == self.begin:
                return PeriodRelation.INSIDE_START_TOUCHING
            return (PeriodRelation.INSIDE_END_TOUCHING
                    if other.last == self.last else PeriodRelation.INSIDE)

        if period_contains_start:
            return PeriodRelation.START_INSIDE

        assert period_contains_end, 'Period must contain end'
        return PeriodRelation.END_INSIDE
