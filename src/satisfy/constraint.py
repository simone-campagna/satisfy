import abc

from .expression import (
    expression_globals,
    Expression,
    Const,
    BoundExpression,
    build_expression,
)

__all__ = [
    'Constraint',
    'ConstConstraint',
    'ExpressionConstraint',
    'AllDifferentConstraint',
]


class Constraint(abc.ABC):
    def __init__(self, function, vars=()):
        self.function = None
        self.vars = frozenset(vars)

    @abc.abstractmethod
    def evaluate(substitution):
        raise NotImplementedError()

    @abc.abstractmethod
    def is_externally_updated(self):
        raise NotImplementedError()

    def compile(self, enabled=True):
        pass

    def unsatisfied(self, substitution):
        for var_name in self.vars:
            if var_name not in substitution:
                return False
        return not self.evaluate(substitution)


class ConstConstraint(Constraint):
    def __init__(self, satisfied):
        self._satisfied = satisfied
        super().__init__(function=self.evaluate, vars=())

    def evaluate(self):
        return self._satisfied

    def is_externally_updated(self):
        return False

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._satisfied)


class AllDifferentConstraint(Constraint):
    def __init__(self, vars):
        super().__init__(function=self.evaluate, vars=vars)

    def is_externally_updated(self):
        return False

    def evaluate(self, substitution):
        return len(set(substitution[var_name] for var_name in self.vars)) == len(self.vars)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, sorted(self.vars))


class ExpressionConstraint(Constraint):
    def __init__(self, expression):
        self.expression = build_expression(expression)
        super().__init__(function=self._get_function(), vars=self.expression.vars)

    def _get_function(self):
        if self.expression.is_compiled:
            function = self.expression.compiled_function
        else:
            function = self.expression.evaluate
        return function

    def is_externally_updated(self):
        return self.expression.is_externally_updated()

    def evaluate(self, substitution):
        return self.expression(substitution)

    def compile(self, enabled=True):
        self.expression.compile(enabled)
        self.function = self._get_function()

    def __repr__(self):
        return repr(self.expression)
