"""Static checks on the generated, hand-assembled polynomial code."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "PyToF"


@pytest.mark.parametrize(
    "module", ["AlgoToF.py", "FunctionsToF.py", "ClassToF.py"]
)
def test_no_discarded_expressions(module):
    """
    A statement that is a bare expression computes a value and drops it. In
    the generated ToF code this is the signature of a forgotten assignment,
    e.g. a missing `A0 +=` in front of a Mathematica export (which silently
    ignored the order-10 differential-rotation term of A0 for a year).
    """
    tree = ast.parse((PACKAGE / module).read_text())
    harmless = (ast.Call, ast.Constant, ast.Await, ast.Yield, ast.YieldFrom)
    bare = [
        (
            node.lineno,
            ast.get_source_segment((PACKAGE / module).read_text(), node)[:60],
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and not isinstance(node.value, harmless)
    ]
    assert not bare, f"discarded expressions in {module}: {bare}"
