# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Handles the partition selection expressions
===========================================
"""
from typing import Any, ClassVar, Dict
import ast
import dataclasses


@dataclasses.dataclass
class Expression:
    """Partitioning expressions.

    Args:
        expression: The expression to be evaluated

    Raises:
        NameError: If a variable is not defined.

    Example:
        >>> expr = Expression("year==2000 and month==1 and day in range(1, 12)")
    """
    __slots__ = ('code', )

    #: Compiled expression to be evaluated
    code: Any

    #: The builtins that are allowed in the expression.
    BUILTINS: ClassVar[Dict[str, Any]] = {'range': range}

    def __init__(self, expression: str) -> None:
        self.code = compile(ast.parse(expression, mode='eval'), ' ', 'eval')

    def __call__(self, variables: Dict[str, Any]) -> Any:
        try:
            __locals = {
                name: variables[name]
                for name in self.code.co_names if name not in self.BUILTINS
            }
            # pylint: disable=eval-used
            # The eval function is used here to evaluate a simple expression.
            # The only builtin functions allowed is the range function.
            return eval(self.code, {'__builtins__': self.BUILTINS}, __locals)
            # pylint: enable=eval-used
        except KeyError as err:
            raise NameError(f'name {err!s} is not defined') from err
