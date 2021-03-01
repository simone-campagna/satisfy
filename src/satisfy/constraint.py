import abc

from .expression import (
    expression_globals,
    ExpressionBase,
    BoundExpression,
)

__all__ = [
    'Constraint',
    'ConstConstraint',
    'ExpressionConstraint',
    'AllDifferentConstraint',
]


class FakeCompileMixin:
    def compile(self):
        pass


class Constraint(ExpressionBase):
    def unsatisfied(self, substitution):
        for var_name in self._var_names:
            if var_name not in substitution:
                return False
        return not self.evaluate(substitution)


class ConstConstraint(Constraint, FakeCompileMixin):
    def __init__(self, value):
        self._value = bool(value)

    def free_vars(self, substitution):
        yield from ()

    def evaluate(self, substitution):
        return self._value

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._value)

    def __str__(self):
        return str(self._value)


class AllDifferentConstraint(Constraint, FakeCompileMixin):
    def __init__(self, var_names):
        self._var_names = frozenset(var_names)

    def free_vars(self, substitution):
        yield from self._var_names.difference(substitution.keys())

    def evaluate(self, substitution):
        return len(set(substitution[var_name] for var_name in self._var_names)) == len(self._var_names)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, sorted(self._var_names))


class ExpressionConstraint(Constraint, BoundExpression):
    pass
