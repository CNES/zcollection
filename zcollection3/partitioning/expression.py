"""Typed AST walker for partition filter expressions.

Replaces the v2 ``eval()``-based filter. Only a small, total subset of Python
syntax is permitted: comparisons, ``and``/``or``/``not``, ``in``, names that
match partition key components, and integer/string literals.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from ..errors import ExpressionError

PartitionKey = tuple[tuple[str, Any], ...]
Predicate = Callable[[dict[str, Any]], bool]


_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load,
    ast.Constant, ast.Tuple, ast.List, ast.Set,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)


def compile_filter(expr: str | None) -> Predicate:
    """Compile a filter expression to a predicate over partition-key dicts.

    ``expr=None`` or empty string returns a tautology.
    """
    if not expr:
        return lambda _ctx: True

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid filter expression: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"disallowed syntax in filter: {type(node).__name__}"
            )

    return _make_predicate(tree.body)


def _make_predicate(node: ast.AST) -> Predicate:
    fn = _compile(node)
    return lambda ctx: bool(fn(ctx))


def _compile(node: ast.AST) -> Callable[[dict[str, Any]], Any]:
    handler = _NODE_HANDLERS.get(type(node))
    if handler is None:
        raise ExpressionError(f"unsupported node: {type(node).__name__}")
    return handler(node)


def _compile_constant(node: ast.Constant) -> Callable[[dict[str, Any]], Any]:
    v = node.value
    return lambda _c: v


def _compile_name(node: ast.Name) -> Callable[[dict[str, Any]], Any]:
    n = node.id

    def _name(ctx: dict[str, Any]) -> Any:
        if n not in ctx:
            raise ExpressionError(f"unknown partition key {n!r}")
        return ctx[n]
    return _name


def _compile_sequence(
    node: ast.Tuple | ast.List | ast.Set,
) -> Callable[[dict[str, Any]], Any]:
    children = [_compile(e) for e in node.elts]
    return lambda c: tuple(f(c) for f in children)


def _compile_boolop(node: ast.BoolOp) -> Callable[[dict[str, Any]], Any]:
    children = [_compile(v) for v in node.values]
    if isinstance(node.op, ast.And):
        return lambda c: all(f(c) for f in children)
    return lambda c: any(f(c) for f in children)


def _compile_unaryop(node: ast.UnaryOp) -> Callable[[dict[str, Any]], Any]:
    if not isinstance(node.op, ast.Not):
        raise ExpressionError(f"unsupported unary op: {type(node.op).__name__}")
    inner = _compile(node.operand)
    return lambda c: not inner(c)


def _compile_compare(node: ast.Compare) -> Callable[[dict[str, Any]], Any]:
    left = _compile(node.left)
    ops = node.ops
    rights = [_compile(c) for c in node.comparators]

    def _cmp(ctx: dict[str, Any]) -> bool:
        cur = left(ctx)
        for op, r in zip(ops, rights, strict=True):
            rv = r(ctx)
            if not _apply_op(op, cur, rv):
                return False
            cur = rv
        return True
    return _cmp


_NODE_HANDLERS: dict[type[ast.AST], Callable[[Any], Callable[[dict[str, Any]], Any]]] = {
    ast.Constant: _compile_constant,
    ast.Name: _compile_name,
    ast.Tuple: _compile_sequence,
    ast.List: _compile_sequence,
    ast.Set: _compile_sequence,
    ast.BoolOp: _compile_boolop,
    ast.UnaryOp: _compile_unaryop,
    ast.Compare: _compile_compare,
}


_CMP_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _apply_op(op: ast.cmpop, a: Any, b: Any) -> bool:
    fn = _CMP_OPS.get(type(op))
    if fn is None:
        raise ExpressionError(
            f"unsupported comparison operator: {type(op).__name__}"
        )
    return fn(a, b)


def key_to_dict(key: PartitionKey) -> dict[str, Any]:
    return dict(key)
