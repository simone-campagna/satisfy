import collections

from .model import Model
from .objective import Maximize
from .solver import ModelSolver, SelectVar, SelectValue

__all__ = [
    'KnapsackSolver',
]


KnapsackSolution = collections.namedtuple(
    "KnapsackSolution",
    "is_optimal solution value weights")


class KnapsackSolver(ModelSolver):
    def __init__(self, values, capacities, weights, **args):
        if args.get('select_var', None) is None:
            args['select_var'] = SelectVar.group_prio
        if args.get('select_value', None) is None:
            args['select_value'] = SelectValue.min_value
        super().__init__(**args)
        model = self._model
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
        variables = [model.add_bool_variable(name="i_{}".format(i)) for i in range(len(values))]
        constraints = []
        for idx, capacity in enumerate(capacities):
            constraint = model.add_constraint(sum(var * weight[idx] for var, weight in zip(variables, weights)) <= capacity)
        model.add_objective(Maximize(sum(var * value for var, value in zip(variables, values))))
        self._values = values
        self._capacities = capacities
        self._weights = weights
        self._variables = variables
        self._model = model

    def make_knapsack_solution(self, opt_result):
        values = self._values
        capacities = self._capacities
        weights = self._weights
        variables = self._variables
        model = self._model
        solver = self._solver
        opt_solution = opt_result.solution
        if opt_solution is not None:
            solution = tuple(opt_solution[var.name] for var in variables)
            value = sum(selected * val for selected, val in zip(solution, values))
            weights = tuple(sum(selected * weight[idx] for selected, weight in zip(solution, weights)) for idx in range(len(capacities)))
        else:
            solution = None
            value = None
            weights = None
        return KnapsackSolution(
            is_optimal=opt_result.is_optimal,
            solution=solution,
            value=value,
            weights=weights)
