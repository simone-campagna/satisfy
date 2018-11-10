import abc
import operator

from .constraint import ExpressionConstraint
from .expression import Expression

__all__ = [
    'Objective',
    'Maximize',
    'Minimize',
]


class Objective(abc.ABC):
    @abc.abstractmethod
    def make_constraint(self, model):
        raise NotImplementedError()

    @abc.abstractmethod
    def add_solution(self, substitution):
        raise NotImplementedError()

    def evaluate(self, substitution):
        return self._expression.evaluate(substitution)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._expression)


class MinMaxConstraint(ExpressionConstraint):
    def __init__(self, model, expression, op):
        self._model = model
        self._not_set = model.add_parameter(1)
        self._bound = model.add_parameter(0)
        super().__init__(self._not_set | op(expression, self._bound))

    def set_bound(self, value):
        self._not_set.value = 0
        self._bound.value = value
        if self.is_compiled():
            self.compile()


class MinConstraint(MinMaxConstraint):
    def __init__(self, model, expression):
        super().__init__(model, expression, op=operator.lt)


class MaxConstraint(MinMaxConstraint):
    def __init__(self, model, expression):
        super().__init__(model, expression, op=operator.gt)


class MinMax(Objective):
    def __init__(self, expression):
        if not isinstance(expression, Expression):
            raise TypeError("{} is not an Expression".format(expression))
        self._expression = expression
        super().__init__()

    def add_solution(self, constraint, substitution):
        value =  self._expression.evaluate(substitution)
        constraint.set_bound(value)


class Minimize(MinMax):
    def make_constraint(self, model):
        return MinConstraint(model, self._expression)


class Maximize(MinMax):
    def make_constraint(self, model):
        return MaxConstraint(model, self._expression)
