# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Tests of the expression evaluation
==================================
"""
from __future__ import annotations

import pytest

from ..expression import Expression


def test_expression() -> None:
    """Test of the creation of expressions."""
    expr = Expression('a == b')
    assert expr({'a': 1, 'b': 1})
    assert not expr({'a': 1, 'b': 2})

    with pytest.raises(SyntaxError):
        Expression('a==')

    with pytest.raises(NameError):
        assert expr({'a': 1, 'c': 1})


@pytest.mark.parametrize(('expression', 'non_matching'), [
    ('year==2000', {
        'year': 2001
    }),
    ('year==2000 and month==1', {
        'year': 2000,
        'month': 2
    }),
    ('year==2000 and month==1 and day in range(1, 12)', {
        'year': 2000,
        'month': 1,
        'day': 12
    }),
])
def test_date_expression(expression, non_matching) -> None:
    """Evaluate expressions using partition keys."""
    expr = Expression(expression)
    assert expr({'year': 2000, 'month': 1, 'day': 11})
    assert not expr(non_matching)
