import pytest

from satisfy.expression import (
    Expression,
    Variable,
    Parameter,
    Const,
)


def test_variable():
    var = Variable("alpha")
    assert var.name == "alpha"


def test_coerce():
    e = Expression.coerce(8)
    assert isinstance(e, Const)
    assert e.value == 8
    assert Expression.coerce(e) is e


@pytest.fixture(params=['False', 'True'])
def compiled(request):
    return request.param


@pytest.mark.parametrize("expr, subst, value", [
    (Const(10), {}, 10),
    (Const(10), {'x': 20}, 10),
    (Variable('x'), {'x': 10}, 10),
    (Variable('x'), {'x': 10, 'y': 20}, 10),
    (3 * Variable('x'), {'x': 10}, 30),
    (3 * Variable('x') * 2 + 3 - Variable('y'), {'x': 10, 'y': 13}, 50),
    (3 * Variable('x') * 2 + 3 - Parameter('y', 13), {'x': 10}, 50),
    (3 * Variable('x') * 2 + 3 - Parameter('y', 13), {'x': 10, 'y': 1000}, 50),
])
def test_evaluate(expr, subst, value, compiled):
    if compiled:
        fn = expr.compile_py_function()
        assert callable(fn)
        assert fn(**subst) == value
    else:
        assert expr.evaluate(subst) == value
