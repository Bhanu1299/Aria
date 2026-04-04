"""
skills/calculate — Aria Phase 3D: Safe math expression evaluation.

Handles: "calculate 5 times 8", "how much is 120 divided by 4", "compute 2 to the power 10"

Uses AST parsing — never calls eval(). Only supports arithmetic operators.
"""

from __future__ import annotations

import ast
import operator
import re

TRIGGERS = [
    "calculate",
    "compute",
    "how much is",
    "what is the result of",
]

_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
}

_SPOKEN_MAP = [
    (r'\btimes\b',          '*'),
    (r'\bmultiplied\s+by\b', '*'),
    (r'\bdivided\s+by\b',   '/'),
    (r'\bover\b',           '/'),
    (r'\bplus\b',           '+'),
    (r'\bminus\b',          '-'),
    (r'\bsquared\b',        '**2'),
    (r'\bcubed\b',          '**3'),
    (r'\bto\s+the\s+power\s+of\b', '**'),
    (r'\bpercent\s+of\b',   '* 0.01 *'),
]


def _safe_eval(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def handle(command: str) -> str:
    """Extract a math expression from the command and evaluate it safely."""
    expr = command.lower()

    # Strip trigger phrases
    for trigger in TRIGGERS:
        expr = expr.replace(trigger, "")

    # Translate spoken operators to symbols
    for pattern, replacement in _SPOKEN_MAP:
        expr = re.sub(pattern, replacement, expr)

    # Keep only math-safe characters
    expr = re.sub(r'[^0-9\s\+\-\*\/\.\(\)\*]', '', expr).strip()

    if not expr:
        return "I couldn't find a math expression to calculate. Try saying 'calculate 5 times 8'."

    try:
        tree = ast.parse(expr, mode='eval')
        result = _safe_eval(tree.body)

        # Format: drop .0 from whole numbers
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        else:
            formatted = f"{result:.6g}"  # up to 6 significant figures

        return f"The result is {formatted}."
    except (SyntaxError, ZeroDivisionError, ValueError, OverflowError) as exc:
        return "I couldn't calculate that. Try saying something like 'calculate 5 times 8'."
