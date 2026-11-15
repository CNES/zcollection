# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Partition selection expressions
===============================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import ast
import dataclasses

if TYPE_CHECKING:
    import types


@dataclasses.dataclass
class Expression:
    """Evaluate an expression against partition keys.

    Args:
        expression: The expression to evaluate.

    Raises:
        NameError: If a variable is missing when the expression is evaluated.

    Example:
        >>> expr = Expression("year==2000 and month==1 and day in range(1, 12)")
    """
    #: Compiled expression to be evaluated
    code: types.CodeType

    #: Names of the instance attributes defined by ``__slots__``.
    __slots__: tuple[str, ...] = ('code', )

    #: The builtins that are allowed in the expression.
    BUILTINS: ClassVar[dict[str, Any]] = {'range': range}

    def __init__(self, expression: str) -> None:
        self.code = compile(ast.parse(expression, mode='eval'), ' ', 'eval')

    def __call__(self, variables: dict[str, Any]) -> Any:
        try:
            __locals: dict[str, Any] = {
                name: variables[name]
                for name in self.code.co_names if name not in self.BUILTINS
            }

            # Evaluate a simple expression with ``eval``, allowing only
            # ``range`` as a built-in function.
            return eval(self.code, {'__builtins__': self.BUILTINS}, __locals)

        except KeyError as err:
            raise NameError(f'name {err!s} is not defined') from err
