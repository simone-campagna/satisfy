import abc
import operator

from .constraint import ExpressionConstraint
from .expression import Expression, expression_globals

__all__ = [
    'ObjectiveConstraint',
    'ObjectiveFunction',
    'Objective',
    'Maximize',
    'Minimize',
]


class ObjectiveConstraint(ExpressionConstraint):
    pass


class ObjectiveExpression(ExpressionConstraint):
    pass


class ObjectiveFunction:
    def __init__(self, model, expression, *constraints):
        self._model = model
        self._expression = ObjectiveExpression(expression)
        self._constraints = constraints

    def compile(self, globals_d):
        self._expression.globals = globals_d
        self._expression.compile()
        for constraint in self._constraints:
            constraint.globals = globals_d
            constraint.compile()

    def add_solution(self, solution):
        pass

    @property
    def constraints(self):
        return self._constraints

    def __call__(self, substitution):
        return self._expression.evaluate(substitution)

    def __repr__(self):
        lst = [repr(self._model), str(self._expression)]
        lst.extend(str(constraint) for constraint in self._constraints)
        return "{}({})".format(type(self).__name__, ', '.join(lst))


class Objective(abc.ABC):
    def __init__(self, expression):
        if not isinstance(expression, Expression):
            raise TypeError("{} is not an Expression".format(expression))
        self._expression = expression

    @property
    def expression(self):
        return self._expression

    @abc.abstractmethod
    def build(self, model):
        raise NotImplementedError()

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self._expression)


class MinMaxConstraint(ObjectiveConstraint):
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


class MinMaxObjectiveFunction(ObjectiveFunction):
    def add_solution(self, solution):
        value = self(solution)
        for constraint in self._constraints:
            constraint.set_bound(value)


class Minimize(Objective):
    def build(self, model):
        return MinMaxObjectiveFunction(
            model, self._expression,
            MinConstraint(model, self._expression))


class Maximize(Objective):
    def build(self, model):
        return MinMaxObjectiveFunction(
            model, self._expression,
            MaxConstraint(model, self._expression))
