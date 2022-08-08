# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Time period
===========
"""
from typing import Any, Optional, Tuple, Union
import enum
import re

import numpy

from ..typing import DType

# Parse the unit of numpy.timedelta64.
PATTERN = re.compile(r'(?:datetime|timedelta)64\[(\w+)\]').search

#: Numpy time units
RESOLUTION = [
    'as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y'
]


def _time64_unit(dtype: DType[Any]) -> str:
    """Get the unit of time."""
    match = PATTERN(dtype.name)
    if match is None:
        raise ValueError(f'dtype is not a time duration: {dtype}')
    return match.group(1)


def _min_time64_unit(*args: DType[Any]) -> str:
    """Get the minimum unit of time."""
    index = min(RESOLUTION.index(_time64_unit(item)) for item in args)
    return RESOLUTION[index]


class PeriodRelation(enum.IntEnum):
    """Enumeration of the relations which can exist between two periods."""
    __slots__ = ()

    #: The first period is after.
    #:
    #: .. code-block:: text
    #:
    #:             [-----p1-----]
    #:      [-p2-]
    AFTER = 0

    #: The first period is after, but the start date of the first period is
    #: the same date as the end date of the second period.
    #:
    #: .. code-block:: text
    #:
    #:           [-----p1-----]
    #:      [-p2-]
    START_TOUCHING = 1

    #: The start date of the first period is inside the second period.
    #:
    #: .. code-block:: text
    #:
    #:        [-----p1-----]
    #:      [-p2-]
    START_INSIDE = 2

    #: The start date of the first period is same as the first date of the
    #: second period and the end date is inside the first period.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:      [-------p2-------]
    INSIDE_START_TOUCHING = 3

    #: The first period is inside the second period and the start date of the
    #: two periods are the same.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:      [--p2--]
    ENCLOSING_START_TOUCHING = 4

    #: The second period is inside the first period.
    #:
    #: .. code-block:: text
    #:
    #:      [-----p1-----]
    #:         [--p2--]
    ENCLOSING = 5

    #: The second period is inside the first period and the end date of the
    #: two periods are the same.
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

    #: The first period is inside the second period.
    #:
    #: .. code-block:: text
    #:
    #:        [---p1---]
    #:      [-----p2-----]
    INSIDE = 8

    #: The first period is inside the second period and the end date of the
    #: two periods are the same.
    #:
    #: .. code-block:: text
    #:
    #:          [---p1---]
    #:      [-----p2-----]
    INSIDE_END_TOUCHING = 9

    #: The end date of the first period is inside the second period.
    #:
    #: .. code-block:: text
    #:
    #:      [---p1---]
    #:            [-----p2-----]
    END_INSIDE = 10

    #: The end date of the first period is same as the start date of the
    #: second period.
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
        """Return true if the relation is after."""
        # pylint: disable=comparison-with-callable
        return self.value == PeriodRelation.AFTER
        # pylint: enable=comparison-with-callable

    def is_before(self) -> bool:
        """Return true if the relation is before."""
        # pylint: disable=comparison-with-callable
        return self.value == PeriodRelation.BEFORE
        # pylint: enable=comparison-with-callable

    def contains(self) -> bool:
        """Return true if one period enclosing the other period."""
        return self.value in (PeriodRelation.ENCLOSING_END_TOUCHING,
                              PeriodRelation.ENCLOSING_START_TOUCHING,
                              PeriodRelation.ENCLOSING,
                              PeriodRelation.EXACT_MATCH)

    def is_before_overlapping(self) -> bool:
        """Return true if one period is before but there is an overlap between
        the two periods."""
        return self.value in (PeriodRelation.END_INSIDE,
                              PeriodRelation.END_TOUCHING,
                              PeriodRelation.INSIDE_START_TOUCHING)

    def is_after_overlapping(self) -> bool:
        """Return true if one period is after but there is an overlap between
        the two periods."""
        return self.value in (PeriodRelation.START_INSIDE,
                              PeriodRelation.START_TOUCHING,
                              PeriodRelation.INSIDE_END_TOUCHING)

    def is_inside(self) -> bool:
        """Return true if one period is inside the other period."""
        # pylint: disable=comparison-with-callable
        return self.value == PeriodRelation.INSIDE
        # pylint: enable=comparison-with-callable


class Period:
    """Create a Period from begin to last eg: [begin, last[

    Args:
        begin: The beginning of the period.
        end: The ending of the period.
        within: If true, the given period defines a closed interval
            (i.e. the end date is within the period), otherwise the
            interval is open.
    """
    __slots__ = ('_begin', '_duration_unit', '_last')

    def __init__(self,
                 begin: numpy.datetime64,
                 end: numpy.datetime64,
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

    def __getstate__(self) -> Tuple[Any, ...]:
        return self._begin, self._duration_unit, self._last

    def __setstate__(self, state: Tuple[Any, ...]) -> None:
        self._begin, self._duration_unit, self._last = state

    @classmethod
    def from_duration(cls, begin: numpy.datetime64,
                      duration: numpy.timedelta64) -> 'Period':
        """Create a Period as [begin, begin + duration[

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
        """True if period is ill formed (length is zero or less)"""
        return bool(self.end() <= self._begin)  # numpy.bool_ -> bool

    def length(self) -> numpy.timedelta64:
        """Return the length of the period."""
        if self._last < self._begin:
            # invalid period
            return self._last + self._duration_unit - self._begin
        return self.end() - self._begin

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, Period):
            return NotImplemented
        return (self._begin == rhs._begin) and (self._last == rhs._last)

    def __ne__(self, rhs: Any) -> bool:
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
        """Expand the size of the period by the duration on both ends.

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
        other: Union[numpy.datetime64, 'Period'],
    ) -> numpy.bool_:
        """Check if the given period is contains this period.

        Args:
            other: The other period to check.

        Returns:
            * True if other is a date and is inside the period, zero length
              periods contain no points
            * True if other is a period and  fully contains (or equals) the
              other period
        """
        if isinstance(other, numpy.datetime64):
            return self._begin <= other <= self._last
        return (self._begin <= other.begin) and (self._last >= other.last)

    def is_adjacent(self, other: 'Period') -> bool:
        """True if periods are next to each other without a gap.

        In the example below, p1 and p2 are adjacent, but p3 is not adjacent
        with either of p1 or p2.

        .. code-block:: text

            [-p1-[
                 [-p2-]
                   [-p3-]

        Args:
            other: The other period to check.

        Returns:
            * True if other is a date and is adjacent to this period
            * True if other is a period and is adjacent to this period
        """
        return other.begin == self.end() or self.begin == other.end()

    def is_after(self, point: numpy.datetime64) -> bool:
        """True if all of the period is prior or point < start.

        In the example below only point 1 would evaluate to true.

        .. code-block:: text

                [----------[
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The point to check.

        Returns:
            True if point is after the period
        """
        if self.is_null():
            # null period isn't after
            return False
        return bool(point < self._begin)  # numpy.bool_ -> bool

    def is_before(self, point: numpy.datetime64) -> bool:
        """True if all of the period is prior to the passed point or end <=
        point.

        In the example below points 4 and 5 return true.

        .. code-block:: text

                [----------[
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The point to check.

        Returns:
            True if point is before the period
        """
        if self.is_null():
            # null period isn't before anything
            return False
        return bool(self._last < point)  # numpy.bool_ -> bool

    def intersects(self, other: 'Period') -> bool:
        """True if the periods overlap in any way.

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
            True if the periods intersect
        """
        return (self.contains(other.begin) or other.contains(self._begin)
                or ((other.begin < self._begin) and
                    (other.last >= self._begin)))  # type:ignore

    def intersection(self, other: 'Period') -> 'Period':
        """Return the period of intersection or null period if no intersection.

        Args:
            other: The other period to check.

        Returns:
            The intersection period or null period if no intersection.
        """
        if self._begin > other.begin:
            if self._last <= other.last:
                return self
            return Period(self._begin, other.end())

        if self._last <= other.last:
            return Period(other.begin, self.end())
        return other

    def merge(self, other: 'Period') -> 'Period':
        """Return the union of intersecting periods -- or null period.

        Args:
            other: The other period to merge.

        Returns:
            The union period of intersection or null if no intersection.
        """
        if self.intersects(other):
            if self._begin < other.begin:
                return Period(
                    self._begin,
                    self.end() if self._last > other.last else other.end())
            return Period(
                other.begin,
                self.end() if self._last > other.last else other.end())
        # no intersect return null
        return Period(self._begin, self._begin)

    def span(self, other: 'Period') -> 'Period':
        """Combine two periods with earliest start and latest end.

        Combines two periods and any gap between them such that
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
        start = self._begin if self._begin < other.begin else other.begin
        end = other.end() if self._last < other.last else self.end()
        return Period(start, end)

    # pylint: disable=too-many-return-statements
    # This code has a lot of "return" statements because we have to determine
    # the relation between the two periods among 8 different cases.
    def _get_direct_relation(self,
                             other: 'Period') -> Optional[PeriodRelation]:
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
        # pylint: enable=too-many-return-statements

    def get_relation(self, other: 'Period') -> PeriodRelation:
        """Get the relationship between the two time periods.

        Args:
            other: Period to consider

        Returns:
            The relation.
        """
        relation = self._get_direct_relation(other)
        if relation is not None:
            return relation

        period_contains_start = other.contains(self._begin)
        period_contains_end = other.contains(self._last)

        if period_contains_start and period_contains_end:
            if other.begin == self.begin:
                return PeriodRelation.INSIDE_START_TOUCHING
            return (PeriodRelation.INSIDE_END_TOUCHING
                    if other.last == self.last else PeriodRelation.INSIDE)

        if period_contains_start:
            return PeriodRelation.START_INSIDE

        assert period_contains_end, 'Period must contain end'
        return PeriodRelation.END_INSIDE
