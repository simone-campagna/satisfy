import collections

from .model import Model
from .objective import Maximize
from .solver import Solver, SelectVar, SelectValue

__all__ = [
    'Knapsack',
]


KnapsackSolution = collections.namedtuple(
    "KnapsackSolution",
    "solution value weights")


class Knapsack(Model):
    def __init__(self, values, capacities, weights):
        super().__init__()
        values = tuple(values)
        if not values:
            return
        for value in values:
            if not isinstance(value, int):
                raise TypeError("{} is not a valid value".format(value))
        capacities = tuple(capacities)
        for capacity in capacities:
            if not isinstance(capacity, int):
                raise TypeError("{} is not a valid capacity".format(capacity))
        ws = []
        for weight in weights:
            if len(ws) >= len(values):
                raise ValueError("too many weights")
            w = tuple(weight)
            if len(w) != len(capacities):
                raise ValueError("value {}: wrong number of weights".format(values[len(ws)]))
            for v in w:
                if not isinstance(v, int):
                    raise TypeError("{} is not a valid weight".format(weight))
            ws.append(w)
        if len(ws) < len(values):
            raise ValueError("too few weights")
        variables = [self.add_bool_variable(name="i_{}".format(i)) for i in range(len(values))]
        constraints = []
        for idx, capacity in enumerate(capacities):
            constraint = self.add_constraint(sum(var * weight[idx] for var, weight in zip(variables, weights)) <= capacity)
        self.add_objective(Maximize(sum(var * value for var, value in zip(variables, values))))
        self._values = values
        self._capacities = capacities
        self._weights = weights
        self._k_variables = variables

    def solver(self, **kwargs):
        return Solver(
            select_var=kwargs.pop('select_var', SelectVar.group_prio),
            select_value=kwargs.pop('select_value', SelectValue.min_value),
            **kwargs
        )

    def make_knapsack_solution(self, solution):
        values = self._values
        capacities = self._capacities
        weights = self._weights
        variables = self._k_variables
        k_solution = tuple(solution[var.name] for var in variables)
        value = sum(selected * val for selected, val in zip(k_solution, values))
        weights = tuple(sum(selected * weight[idx] for selected, weight in zip(k_solution, weights)) for idx in range(len(capacities)))
        return KnapsackSolution(
            solution=k_solution,
            value=value,
            weights=weights)
