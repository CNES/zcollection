# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of the time periods.
=========================
"""
import datetime
import pickle

import numpy
import pytest

from ..period import Period, PeriodRelation


def _datetime64(year: int, month: int, day: int) -> numpy.datetime64:
    return numpy.datetime64(datetime.datetime(year, month, day))


def _datetime(value: int) -> numpy.datetime64:
    return numpy.datetime64(value, 'D')


def _timedelta(value: int) -> numpy.timedelta64:
    return numpy.timedelta64(value, 'D')


def _period(start: int, end: int, within: bool = False) -> Period:
    return Period(_datetime(start), _datetime(end), within=within)


def _period1(within=False):
    return _period(1, 10, within=within)


def _period2():
    return _period(5, 30)


def _period3():
    return _period(35, 81)


def test_interface():
    """Test the interface of the Period class."""
    period1 = _period1()
    assert period1.begin == _datetime(1)
    assert period1.last == _datetime(9)
    assert period1.end() == _datetime(10)
    assert period1.length() == _timedelta(9)
    assert not period1.is_null()
    assert isinstance(str(period1), str)

    period1 = _period1(within=True)
    assert period1.begin == _datetime(1)
    assert period1.last == _datetime(10)
    assert period1.end() == _datetime(11)
    assert period1.length() == _timedelta(10)
    assert not period1.is_null()

    period2 = _period2()
    assert period2.begin == _datetime(5)
    assert period2.last == _datetime(29)
    assert period2.end() == _datetime(30)
    assert period2.length() == _timedelta(25)
    assert not period2.is_null()


def test_cmp():
    """Test the comparison operators."""
    period1 = _period1()
    period2 = _period2()
    period3 = _period3()

    assert period1 == _period1()
    assert period1 != period2
    assert period1 < period3

    with pytest.raises(TypeError):
        assert period1 < 1
    assert not period1 == 1  # pylint: disable=unneeded-not
    assert period1 != 1


def test_shift():
    """Test the shift method."""
    period1 = _period1()
    period1.shift(_timedelta(5))
    assert period1 == _period(6, 15)
    period1.shift(_timedelta(-15))
    assert period1 == _period(-9, 0)


def test_relation():
    """Test the relations between periods."""
    period1 = _period1()
    period2 = _period2()
    period3 = _period3()

    assert period2.contains(_datetime(20))
    assert not period2.contains(_datetime(2))

    assert period1.contains(_period(2, 8))
    assert not period1.contains(period3)

    assert period1.intersects(period2)
    assert period2.intersects(period1)

    assert period1.is_adjacent(_period(-5, 1))
    assert period1.is_adjacent(_period(10, 20))
    assert not period1.is_adjacent(period3)

    assert period1.is_before(_datetime(15))
    assert period3.is_after(_datetime(15))

    assert period1.intersection(period2) == _period(5, 10)
    assert period1.intersection(period3).is_null()

    assert period1.merge(period2) == _period(1, 30)
    assert period1.merge(period3).is_null()

    assert period3.span(period1) == _period(1, 81)

    period2 = Period(period1.begin - numpy.timedelta64(10, 's'),
                     period1.end() - numpy.timedelta64(1, 'D'))
    assert period1.intersection(period2) == Period(period1.begin,
                                                   period2.end())
    assert period1.merge(period2) == Period(period2.begin, period1.end())

    period2 = Period(period1.begin + numpy.timedelta64(10, 's'),
                     period1.end() + numpy.timedelta64(1, 'D'))
    assert period1.intersection(period2) == Period(period2.begin,
                                                   period1.end())

    period1 = _period(10, 20)
    period2 = _period(15, 18)
    assert period1.intersection(period2) == period2


def test_zero_length_period():
    """Test the zero length period."""
    zero_len = Period.from_duration(_datetime(3), _timedelta(0))
    assert _period(1, 1) == Period.from_duration(_datetime(1), _timedelta(0))
    assert _period(3, 3) == zero_len

    # zero_length period always returns false for is_before & is_after
    assert not zero_len.is_before(_datetime(5))
    assert not zero_len.is_after(_datetime(5))
    assert not zero_len.is_before(_datetime(-5))
    assert not zero_len.is_after(_datetime(-5))

    assert zero_len.is_null()
    assert not zero_len.contains(_datetime(20))
    # a null_period cannot contain any points
    assert not zero_len.contains(_datetime(3))
    assert not zero_len.contains(_period(5, 8))

    period1 = _period1()
    assert period1.contains(zero_len)
    assert zero_len.intersects(period1)
    assert period1.intersects(zero_len)
    assert zero_len.is_adjacent(_period(-10, 3))
    assert _period(-10, 3).is_adjacent(zero_len)
    assert zero_len.intersection(period1) == zero_len
    period2 = _period2()
    assert zero_len.span(period2) == _period(3, 30)


def test_invalid_period():
    """Test the invalid period."""
    null_per = _period(5, 1)

    assert not null_per.is_before(_datetime(7))
    assert not null_per.is_after(_datetime(7))
    assert not null_per.is_before(_datetime(-5))
    assert not null_per.is_after(_datetime(-5))

    assert null_per.is_null()
    assert not null_per.contains(_datetime(20))
    assert not null_per.contains(_datetime(3))
    assert not null_per.contains(_period(7, 9))
    period1 = _period1()
    assert period1.contains(null_per)
    assert null_per.intersects(period1)
    assert period1.intersects(null_per)
    assert null_per.is_adjacent(_period(-10, 5))
    assert null_per.is_adjacent(_period(1, 10))

    assert null_per.span(_period3()) == _period(5, 81)


def test_invalid():
    """Test the invalid periods."""
    period1 = _period(0, -2)
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(-3)
    assert period1.end() == _datetime(-2)
    assert period1.length() == _timedelta(-2)
    assert period1.is_null()

    period1 = _period(0, -1)
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(-2)
    assert period1.end() == _datetime(-1)
    assert period1.length() == _timedelta(-1)
    assert period1.is_null()

    period1 = _period(0, 0)
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(-1)
    assert period1.end() == _datetime(0)
    assert period1.length() == _timedelta(0)
    assert period1.is_null()

    period1 = _period(0, 1)
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(0)
    assert period1.end() == _datetime(1)
    assert period1.length() == _timedelta(1)
    assert not period1.is_null()

    period1 = _period(0, 2)
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(1)
    assert period1.end() == _datetime(2)
    assert period1.length() == _timedelta(2)
    assert not period1.is_null()

    period1 = Period.from_duration(_datetime(0), _timedelta(-1))
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(-2)
    assert period1.end() == _datetime(-1)
    assert period1.length() == _timedelta(-1)
    assert period1.is_null()

    period1 = Period.from_duration(_datetime(0), _timedelta(-2))
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(-3)
    assert period1.end() == _datetime(-2)
    assert period1.length() == _timedelta(-2)
    assert period1.is_null()

    period1 = Period.from_duration(_datetime(0), _timedelta(0))
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(-1)
    assert period1.end() == _datetime(0)
    assert period1.length() == _timedelta(0)
    assert period1.is_null()

    period1 = Period.from_duration(_datetime(0), _timedelta(1))
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(0)
    assert period1.end() == _datetime(1)
    assert period1.length() == _timedelta(1)
    assert not period1.is_null()

    period1 = Period.from_duration(_datetime(0), _timedelta(2))
    assert period1.begin == _datetime(0)
    assert period1.last == _datetime(1)
    assert period1.end() == _datetime(2)
    assert period1.length() == _timedelta(2)
    assert not period1.is_null()

    period1 = _period(1, 1)
    period2 = _period(1, 2)
    period3 = _period(1, 3)
    assert period1.length() == _timedelta(0)
    assert period2.length() == _timedelta(1)
    assert period3.length() == _timedelta(2)
    assert period1.is_null()
    assert not period2.is_null()

    period1 = _period(1, 2)
    period1.shift(_timedelta(1))
    period2 = _period(2, 3)
    assert period1 == period2

    period1 = Period.from_duration(_datetime(5), _timedelta(3))
    period2 = _period(3, 10)
    period1.expand(_timedelta(2))
    assert period1 == period2


def test_period_get_relation():
    """Test the get_relation method."""

    #          ##################
    #  #####
    start = _datetime64(2000, 6, 1)
    end = _datetime64(2000, 6, 30)
    period1 = Period(start, end, within=True)

    start = _datetime64(2000, 1, 5)
    end = _datetime64(2000, 1, 20)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.AFTER
    assert period1.get_relation(period2).is_after()

    #          ##################
    #     #####
    start = _datetime64(2000, 1, 5)
    end = _datetime64(2000, 6, 1)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.START_TOUCHING
    assert period1.get_relation(period2).is_after_overlapping()

    #          ##################
    #      #####
    start = _datetime64(2000, 1, 5)
    end = _datetime64(2000, 6, 2)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.START_INSIDE
    assert period1.get_relation(period2).is_after_overlapping()

    #          ##################
    #          ######################
    start = _datetime64(2000, 6, 1)
    end = _datetime64(2000, 7, 15)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(
        period2) == PeriodRelation.INSIDE_START_TOUCHING
    assert period1.get_relation(period2).is_before_overlapping()

    #          ##################
    #          #####
    start = _datetime64(2000, 6, 1)
    end = _datetime64(2000, 6, 15)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(
        period2) == PeriodRelation.ENCLOSING_START_TOUCHING
    assert period1.get_relation(period2).contains()

    #          ##################
    #                #####
    start = _datetime64(2000, 6, 10)
    end = _datetime64(2000, 6, 15)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.ENCLOSING
    assert period1.get_relation(period2).contains()

    #          ##################
    #                       #####
    start = _datetime64(2000, 6, 10)
    end = _datetime64(2000, 6, 30)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(
        period2) == PeriodRelation.ENCLOSING_END_TOUCHING
    assert period1.get_relation(period2).contains()

    #          ##################
    #          ##################
    start = _datetime64(2000, 6, 1)
    end = _datetime64(2000, 6, 30)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.EXACT_MATCH
    assert period1.get_relation(period2).contains()

    #          ##################
    #      ##########################
    start = _datetime64(2000, 5, 1)
    end = _datetime64(2000, 7, 15)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.INSIDE
    assert period1.get_relation(period2).is_inside()

    #          ##################
    #      ######################
    start = _datetime64(2000, 5, 1)
    end = _datetime64(2000, 6, 30)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.INSIDE_END_TOUCHING
    assert period1.get_relation(period2).is_after_overlapping()

    #          ##################
    #                         #####
    start = _datetime64(2000, 6, 15)
    end = _datetime64(2000, 7, 15)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.END_INSIDE
    assert period1.get_relation(period2).is_before_overlapping()

    #          ##################
    #                           #####
    start = _datetime64(2000, 6, 30)
    end = _datetime64(2000, 7, 15)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.END_TOUCHING
    assert period1.get_relation(period2).is_before_overlapping()

    #          ##################
    #                              #####
    start = _datetime64(2000, 7, 10)
    end = _datetime64(2000, 7, 15)
    period2 = Period(start, end, within=True)

    assert period1.get_relation(period2) == PeriodRelation.BEFORE
    assert period1.get_relation(period2).is_before()


def test_pickle():
    """Test pickling."""
    period = _period1()
    assert pickle.loads(pickle.dumps(period)) == period


def test_invalid_type():
    """Test invalid type."""
    with pytest.raises(ValueError):
        Period(numpy.int64(1), numpy.int64(2))  # type: ignore
