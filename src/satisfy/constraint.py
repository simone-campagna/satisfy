import abc

from .expression import (
    Expression,
)

__all__ = [
    'Constraint',
    'ConstConstraint',
    'ExpressionConstraint',
    'AllDifferentConstraint',
]


class Constraint(abc.ABC):
    def __init__(self, var_names):
        self._var_names = frozenset(var_names)

    @abc.abstractmethod
    def evaluate(self, substitution):
        pass

    def unsatisfied(self, substitution):
        for var_name in self._var_names:
            if var_name not in substitution:
                return False
        return not self.evaluate(substitution)

    def free_vars(self, substitution):
        return self._var_names.difference(substitution)

    def vars(self):
        return self._var_names

    def compile(self):
        pass

    def is_compiled(self):
        return False


class ConstConstraint(Constraint):
    def __init__(self, value):
        super().__init__([])
        self._value = bool(value)

    def evaluate(self, substitution):
        return self._value

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._value)

    def __str__(self):
        return str(self._value)


class ExpressionConstraint(Constraint):
    def __init__(self, expression):
        if not isinstance(expression, Expression):
            raise TypeError("{} is not an Expression".format(expression))
        self._expression = expression
        super().__init__(self._expression.vars())
        self._evaluate_function = None

    def is_compiled(self):
        return self._evaluate_function is not None

    def compile(self):
        self._evaluate_function = self._expression.as_function()

    def evaluate_function(self):
        if self._evaluate_function is None:
            self._evaluate_function = self._expression.as_function()
        return self._evaluate_function

    def evaluate(self, substitution):
        if self._evaluate_function is None:
            return self._expression.evaluate(substitution)
        else:
            return self._evaluate_function(**substitution)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._expression)

    def __str__(self):
        return str(self._expression)


class AllDifferentConstraint(Constraint):
    def evaluate(self, substitution):
        return len(set(substitution[var_name] for var_name in self._var_names)) == len(self._var_names)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, sorted(self._var_names))
