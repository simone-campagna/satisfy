import pytest

from satisfy.expression import (
    set_expression_globals,
    Expression,
    Variable,
    GlobalVariable,
    FunctionCall,
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


def _foo(*args, **kwargs):
    s = 0
    for arg in args:
        s += arg
    # print(s, args, kwargs, s * kwargs['c'])
    return s * kwargs['c']


_EXPRESSIONS = [
    (Const(10),
     {}, {}, 10),
    (Const(10),
     {}, {'x': 20}, 10),
    (Variable('x'),
     {}, {'x': 10}, 10),
    (Variable('x'),
     {}, {'x': 10, 'y': 20}, 10),
    (3 * Variable('x'),
     {}, {'x': 10}, 30),
    (3 * Variable('x') * 2 + 3 - Variable('y'),
     {}, {'x': 10, 'y': 13}, 50),
    (3 * Variable('x') * 2 + 3 - Parameter('y', 13),
     {}, {'x': 10}, 50),
    (3 * Variable('x') * 2 + 3 - Parameter('y', 13),
     {}, {'x': 10, 'y': 1000}, 50),
    (7 * GlobalVariable('M') + Variable('x'),
     {'M': 10}, {'x': 10, 'y': 1000}, 80),
    (7 * GlobalVariable('M') + Variable('x'),
     {'M': 10}, {'x': 10, 'M': 1000}, 7010),
    (7 * FunctionCall('foo', [1, 2, 3, 4], {'c': 3}) + Variable('x'),
     {'foo': _foo}, {'x': 10, 'M': 1000}, 7 * 30 + 10),
]


@pytest.mark.parametrize("expr, g_dict, subst, result", _EXPRESSIONS)
def test_evaluate(expr, g_dict, subst, result):
    with set_expression_globals(g_dict, merge=True):
        assert expr.evaluate(subst) == result

@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("expr, g_dict, subst, result", _EXPRESSIONS)
def test_call(compiled, expr, g_dict, subst, result):
    with set_expression_globals(g_dict, merge=True):
        expr.compile(compiled)
        assert expr.is_compiled() == compiled
        assert expr(subst) == result
