import abc
import collections
import itertools

from .constraint import AllDifferentConstraint
from .model import Model
from .objective import Objective
from .utils import INFINITY, Timer, SolveStats


__all__ = [
    'ModelInfo',
    'OptimalSolution',
    'Solver',
    'ModelSolver',
    'ModelOptimizer',
]


ModelInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'ModelInfo',
    [
        'initial_domains',
        'reduced_domains',
        'domains',
        'var_names',
        'var_constraints',
        'var_ad',
        'bound_constraints',
    ]
)


OptimalSolution = collections.namedtuple(  # pylint: disable=invalid-name
    "OptimalSolution",
    "is_optimal solution")


def max_bound(bound_var_names, unbound_var_names, model_info):
    if len(bound_var_names) == 0:
        domains = model_info.domains
        varset = frozenset(bound_var_names)
        bound_constraints = model_info.bound_constraints
        def skey(var_name):
            return -len(bound_constraints[varset.union([var_name])])
        unbound_var_names.sort(key=lambda v: (-len(bound_constraints[varset.union([v])]), len(domains[v])))
        # unbound_var_names.sort(key=lambda v: -len(bound_constraints[varset.union([v])]))
    var_name = unbound_var_names.pop(0)
    return var_name, unbound_var_names


def min_value(var_name, substitution, reduced_domain):
    value = min(reduced_domain)
    reduced_domain.discard(value)
    return value, reduced_domain


class Solver(object):
    def __init__(self,
                 select_var=max_bound,
                 select_value=min_value,
                 timeout=None,
                 limit=None,
                 compile_constraints=True):
        self._select_var = None
        self.select_var = select_var
        self._select_value = None
        self.select_value = select_value
        self._timeout = None
        self.timeout = timeout
        self._limit = None
        self.limit = limit
        self._compile_constraints = None
        self.compile_constraints = compile_constraints
        self._timer = None
        self._interrupt = None
        self._reset()

    @property
    def select_var(self):
        return self._select_var

    @select_var.setter
    def select_var(self, value):
        if False:
            raise TypeError("{!r} is not a VarSelectionPolicy".format(value))
        self._select_var = value

    @property
    def select_value(self):
        return self._select_value

    @select_value.setter
    def select_value(self, value):
        if False:
            raise TypeError("{!r} is not a VarSelectionPolicy".format(value))
        self._select_value = value

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if value is INFINITY:
            value = None
        self._timeout = value

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, value):
        if value is INFINITY:
            value = None
        self._limit = value

    @property
    def compile_constraints(self):
        return self._compile_constraints

    @compile_constraints.setter
    def compile_constraints(self, value):
        self._compile_constraints = bool(value)

    def solve(self, model, *, additional_constraints=(), **args):
        select_var = args.get('select_var', self._select_var)
        select_value = args.get('select_value', self._select_value)
        timeout = args.get('timeout', self._timeout)
        limit = args.get('limit', self._limit)
        compile_constraints = args.get('compile_constraints', self._compile_constraints)

        # 1. internal data structures:
        variables = model.variables()
        initial_domains = {
            var_name: var_info.domain for var_name, var_info in variables.items() if var_info.domain is not None
        }
        var_names = list(initial_domains)
        var_names_set = set(var_names)
        var_constraints = {var_name: [] for var_name in var_names}
        var_ad = {var_name: set() for var_name in var_names}
        bound_constraints = collections.defaultdict(list)
        constraint_vars = collections.defaultdict(set)
        for constraint in itertools.chain(model.constraints(), additional_constraints):
            if compile_constraints:
                constraint.compile()
            if isinstance(constraint, AllDifferentConstraint):
                ad_var_names = set()
                for var_name in constraint.vars():
                    if var_name in var_names_set:
                        ad_var_names.add(var_name)
                for var_name in ad_var_names:
                    var_ad[var_name].update(ad_var_names.difference([var_name]))
            else:
                varset = set()
                c_vars = constraint_vars[constraint]
                for var_name in constraint.vars():
                    if var_name in var_names_set:
                        c_vars.add(var_name)
                        var_constraints[var_name].append(constraint)
                        varset.add(var_name)
                bound_constraints[frozenset(varset)].append(constraint)

        reduced_domains = {}
        domains = collections.ChainMap(reduced_domains, initial_domains)
        model_info = ModelInfo(
            initial_domains=initial_domains,
            reduced_domains=reduced_domains,
            domains=domains,
            var_names=var_names,
            var_constraints=var_constraints,
            var_ad=var_ad,
            bound_constraints=bound_constraints,
        )

        # 2. solve:
        stack = []
        if var_names:
            var_name, unbound_var_names = select_var([], var_names, model_info)
            stack.append(([var_name], unbound_var_names, {}))

        timer = self._timer
        timer.start()
        num_solutions = 0
        while stack:
            if timeout is not None:
                cur_elapsed = timer.elapsed()
                if cur_elapsed > timeout:
                    self._interrupt = "timeout"
                    return

            bound_var_names, unbound_var_names, substitution = stack[-1]
            var_name = bound_var_names[-1]
            #print(var_name, unbound_var_names, substitution)
            substitution = substitution.copy()
            reduced_domain = reduced_domains.get(var_name, None)
            if reduced_domain is None:
                bound_vars = set(substitution)
                bound_vars.add(var_name)
                var_bound_constraints = bound_constraints[frozenset(bound_vars)]
                forbidden_values = {substitution[vname] for vname in var_ad[var_name].intersection(substitution)}
                reduced_domain = set()
                for value in initial_domains[var_name]:
                    if value in forbidden_values:
                        continue
                    substitution[var_name] = value
                    for constraint in var_bound_constraints:
                        if not constraint.evaluate(substitution):
                            break
                    else:
                        reduced_domain.add(value)
                if reduced_domain:
                    reduced_domains[var_name] = reduced_domain
                else:
                    stack.pop(-1)
                    #print("A")
                    continue
            elif not reduced_domain:
                reduced_domains.pop(var_name)
                stack.pop(-1)
                #print("B")
                continue
            # REM print("   ", var_name, unbound_var_names, reduced_domain)

            #print("{} -> {}".format(var_name, reduced_domain))
            #input("...")
            # select value:
            value, reduced_domain = select_value(var_name, substitution, reduced_domain)
            substitution[var_name] = value
            if unbound_var_names:
                # REM print("...", var_name, unbound_var_names, substitution)
                unbound_var_names = list(unbound_var_names)
                next_var_name, next_unbound_var_names = select_var(bound_var_names, unbound_var_names, model_info)
                stack.append((bound_var_names + [next_var_name], next_unbound_var_names, substitution))
            else:
                timer.stop()
                # for constraint in model.constraints():
                #     print(constraint.evaluate(substitution), constraint)
                num_solutions += 1
                # REM print(":::", var_name, unbound_var_names, substitution)
                yield substitution
                if limit is not None and num_solutions >= limit:
                    timer.abort()
                    self._interrupt = "limit"
                    return
                timer.start()
                continue
        timer.abort()

    def optimize(self, model, objective, **args):
        if isinstance(objective, Objective):
            objectives = (objective,)
        else:
            objectives = tuple(objective)
        objective_constraints = []
        for objective in objectives:
            if not isinstance(objective, Objective):
                raise ValueError("{} is not an Objective".format(objective))
            objective_constraints.append(objective.make_constraint(model))
        args['additional_constraints'] = tuple(args.get('additional_constraints', ())) + tuple(objective_constraints)
        for solution in self.solve(model, **args):
            # print("sol found:", solution)
            for objective, constraint in zip(objectives, objective_constraints):
                objective.add_solution(constraint, solution)
            # print("values:", values)
            # input("---")
            yield OptimalSolution(
                is_optimal=None,
                solution=solution)

    def optimal_solution(self, model, objective, **args):
        solution = None
        for opt_solution in self.optimize(model, objective, **args):
            solution = opt_solution.solution
        if solution is not None and self._interrupt is None:
            is_optimal = True
        else:
            is_optimal = False
        return OptimalSolution(
            is_optimal=is_optimal,
            solution=solution)

    def get_stats(self):
        stats = self._timer.stats()
        return SolveStats(
            count=stats.count,
            elapsed=stats.elapsed,
            interrupt=self._interrupt)

    def _reset(self):
        self._timer = Timer()
        self._interrupt = None


class ModelSolverBase(abc.ABC):
    def __init__(self, *, model=None, solver=None, **args):
        if model is None:
            model = Model()
        self._model = model
        if solver is None:
            solver = Solver(**args)
        self._solver = solver

    @property
    def model(self):
        return self._model

    @property
    def solver(self):
        return self._solver

    def get_stats(self):
        return self._solver.get_stats()


class ModelSolver(ModelSolverBase):
    def __iter__(self):
        yield from self._solver.solve(self._model)


class ModelOptimizer(ModelSolverBase):
    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()

