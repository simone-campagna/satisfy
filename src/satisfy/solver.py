import abc
import collections
import enum
import itertools
import random

from .constraint import AllDifferentConstraint
from .model import Model
from .objective import Objective
from .utils import INFINITY, Timer, SolveStats

__all__ = [
    'Solver',
    'VarSelectionPolicy',
    'ModelSolver',
    'ModelOptimizer',
]

ModelInfo = collections.namedtuple(
    "ModelInfo",
    "variables var_constraints var_ad_constraints var_bounds var_sizes var_groups groups group_bounds group_sizes")


OptSolution = collections.namedtuple(
    "OptSolution",
    "is_optimal solution")


class VarSelectionPolicy(enum.Enum):
    ORDERED = 0
    REVERSED = 1
    RANDOM = 2
    GROUPS = 3
    MIN_BOUND = 4
    MAX_BOUND = 5
    MIN_CONSTRAINT = 6
    MAX_CONSTRAINT = 7


class Solver(object):
    def __init__(self,
                 var_selection_policy=VarSelectionPolicy.GROUPS,
                 timeout=None, limit=None,
                 compile_constraints=True):
        self._var_selection_policy = None
        self.var_selection_policy = var_selection_policy
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
    def var_selection_policy(self):
        return self._var_selection_policy

    @var_selection_policy.setter
    def var_selection_policy(self, value):
        if not isinstance(value, VarSelectionPolicy):
            raise TypeError("{!r} is not a VarSelectionPolicy".format(value))
        self._var_selection_policy = value

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

    def _setup(self, model, *, additional_constraints=()):
        var_constraints = collections.defaultdict(list)
        pure_variables = collections.OrderedDict((var_name, var_info) for var_name, var_info in model.variables().items() if var_info.domain is not None)
        var_ad_constraints = collections.defaultdict(set)
        var_bounds = {var_name: 0 for var_name in pure_variables}
        var_sizes = {var_name: len(var_info.domain) for var_name, var_info in pure_variables.items()}
        groups = {}
        var_groups = collections.defaultdict(list)
        compile_constraints = self._compile_constraints
        for constraint in itertools.chain(model.constraints(), additional_constraints):
            if compile_constraints:
                constraint.compile()
            c_vars = constraint.vars()
            for var_name in c_vars:
                if var_name in pure_variables:
                    var_bounds[var_name] += len(c_vars) - 1
            if isinstance(constraint, AllDifferentConstraint):
                groupid = len(groups)
                groups[groupid] = c_vars
                for var_name in c_vars:
                    var_groups[var_name].append(groupid)
                if len(c_vars) > 1:
                    for var_name in c_vars:
                        var_ad_constraints[var_name].update(v for v in c_vars if v != var_name)
            else:
                if len(c_vars) == 1:
                    # single value constraint
                    var_name = next(iter(c_vars))
                    if var_name in pure_variables:
                        domain = []
                        for value in pure_variables[var_name].domain:
                            if constraint.evaluate({var_name: value}):
                                domain.append(value)
                        pure_variables[var_name] = pure_variables[var_name]._replace(domain=tuple(domain))
                    # ignore constraint
                    continue
                for var_name in c_vars:
                    var_constraints[var_name].append(constraint)
        group_bounds = {}
        group_sizes = {}
        for groupid, c_vars in groups.items():
            group_bounds[groupid] = sum(var_bounds[var_name] for var_name in c_vars)
            group_sizes[groupid] = max(var_sizes[var_name] for var_name in c_vars)
        return ModelInfo(
            variables=pure_variables,
            var_constraints=var_constraints,
            var_ad_constraints=var_ad_constraints,
            var_bounds=var_bounds,
            var_sizes=var_sizes,
            var_groups=var_groups,
            groups=groups,
            group_bounds=group_bounds,
            group_sizes=group_sizes,
        )

    def _reset(self):
        self._timer = Timer()
        self._interrupt = None

    def solve(self, model, **args):
        self._reset()
        timeout = args.get('timeout', self._timeout)
        limit = args.get('limit', self._limit)
        var_selection_policy = args.get('var_selection_policy', self._var_selection_policy)
        additional_constraints = args.get('additional_constraints', ())
        # print("::: timeout=", timeout)
        # print("::: limit=", limit)
        # print("::: var_selection_policy=", var_selection_policy)
        # print("::: additional_constraints=", additional_constraints)
        if not model.solvable():
            return

        model_info = self._setup(model, additional_constraints=additional_constraints)
        variables = model_info.variables
        var_constraints = model_info.var_constraints
        var_ad_constraints = model_info.var_ad_constraints
        var_names = tuple(variables)
        if var_selection_policy is VarSelectionPolicy.GROUPS:
            var_bounds = model_info.var_bounds
            var_sizes = model_info.var_sizes
            var_groups = model_info.var_groups
            groups = model_info.groups
            group_bounds = model_info.group_bounds
            group_sizes = model_info.group_sizes


            sort_group = lambda x: (-group_bounds[x], -group_sizes[x])
            group_prio = {groupid: groupid for groupid in groups}  #idx for idx, groupid in enumerate(sorted(groups, key=lambda x: (group_bounds[x], group_sizes[x]), reverse=True))}
            var_group_prio = {}
            for var_name in var_names:
                if var_groups[var_name]:
                    var_group_prio[var_name] = min(group_prio[groupid] for groupid in var_groups[var_name])
                else:
                    var_group_prio[var_name] = -1
            sort_key = lambda x: (var_group_prio[x], var_bounds[x], var_sizes[x])
            s_var_names = tuple(sorted(variables, key=sort_key, reverse=True))
            var_names = tuple(reversed(s_var_names))
        elif var_selection_policy is VarSelectionPolicy.RANDOM:
            var_names = list(var_names)
            random.shuffle(var_names)
            var_names = tuple(var_names)
        elif var_selection_policy is VarSelectionPolicy.REVERSED:
            var_names = tuple(reversed(var_names))
        elif var_selection_policy is VarSelectionPolicy.MIN_BOUND:
            # var_names = tuple(sorted(var_names, key=lambda x: len(var_constraints[x]), reverse=False))
            var_names = tuple(sorted(var_names, key=lambda x: len(model.get_var_domain(x))))
        elif var_selection_policy is VarSelectionPolicy.MAX_BOUND:
            # var_names = tuple(sorted(var_names, key=lambda x: len(var_constraints[x]), reverse=False))
            var_names = tuple(sorted(var_names, key=lambda x: len(model.get_var_domain(x)), reverse=True))
        elif var_selection_policy is VarSelectionPolicy.MIN_CONSTRAINT:
            var_names = tuple(sorted(var_names, key=lambda x: len(var_constraints[x])))
        elif var_selection_policy is VarSelectionPolicy.MAX_CONSTRAINT:
            var_names = tuple(sorted(var_names, key=lambda x: len(var_constraints[x]), reverse=True))

        if not var_names:
            return

        # print("::: var_names=", var_names)
        # for var_name in var_names:
        #     print("::: var_name={}".format(var_name))
        #     print(":::     domain={}".format(variables[var_name].domain))
        #     print(":::     var_constraints: [#{}]".format(len(model_info.var_constraints.get(var_name, []))))
        #     for c in model_info.var_constraints.get(var_name, []):
        #         print(":::         {}".format(c))

        stack = []
        domains = {}
        stack.append((var_names, {}))

        timer = self._timer
        timer.start()
        num_solutions = 0
        while stack:
            if timeout is not None:
                cur_elapsed = timer.elapsed()
                if cur_elapsed > timeout:
                    self._interrupt = "timeout"
                    return

            unbound_vars, substitution = stack[-1]

            var_name, unbound_vars = unbound_vars[0], unbound_vars[1:]

            substitution = substitution.copy()
            domain = domains.get(var_name, None)
            if domain is None:
                domain = []
                v_ad_constraints = var_ad_constraints.get(var_name, ())
                for value in variables[var_name].domain:
                    substitution[var_name] = value
                    for c_var in v_ad_constraints:
                        if substitution.get(c_var, None) == value:
                            # var_name == value is *not* acceptable
                            # because it breaks some AllDifferentConstraint
                            break
                    else:
                        for constraint in var_constraints[var_name]:
                            if constraint.unsatisfied(substitution):
                                # var_name == value is *not* acceptable
                                # because it breaks the Constraint
                                break
                        else:
                            # var_name == value is acceptable
                            domain.append(value)
                if domain:
                    domains[var_name] = domain
                else:
                    stack.pop(-1)
                    continue
            elif not domain:
                domains.pop(var_name)
                stack.pop(-1)
                continue
            value = domain.pop(0)
            substitution[var_name] = value
            if unbound_vars:
                stack.append((unbound_vars, substitution))
            else:
                timer.stop()
                # for constraint in model.constraints():
                #     print(constraint.evaluate(substitution), constraint)
                num_solutions += 1
                yield substitution
                if limit is not None and num_solutions >= limit:
                    timer.abort()
                    self._interrupt = "limit"
                    return
                timer.start()
                continue

    def _solve_forbidden(self, model, **args):
        timeout = args.get('timeout', self._timeout)
        var_selection_policy = args.get('var_selection_policy', self._var_selection_policy)
        additional_constraints = args.get('additional_constraints', ())
        # print("::: timeout=", timeout)
        # print("::: var_selection_policy=", var_selection_policy)
        # print("::: additional_constraints=", additional_constraints)

        model_info = self._setup(model, additional_constraints=additional_constraints)
        variables = model_info.variables
        var_constraints = model_info.var_constraints
        var_ad_constraints = model_info.var_ad_constraints
        var_names = tuple(variables)
        if var_selection_policy is VarSelectionPolicy.GROUPS:
            var_bounds = model_info.var_bounds
            var_sizes = model_info.var_sizes
            var_groups = model_info.var_groups
            groups = model_info.groups
            group_bounds = model_info.group_bounds
            group_sizes = model_info.group_sizes

            sort_group = lambda x: (-group_bounds[x], -group_sizes[x])
            group_prio = {groupid: groupid for groupid in groups}  #idx for idx, groupid in enumerate(sorted(groups, key=lambda x: (group_bounds[x], group_sizes[x]), reverse=True))}
            var_group_prio = {}
            for var_name in var_names:
                if var_groups[var_name]:
                    var_group_prio[var_name] = min(group_prio[groupid] for groupid in var_groups[var_name])
                else:
                    var_group_prio[var_name] = -1
            sort_key = lambda x: (var_group_prio[x], var_bounds[x], var_sizes[x])
            s_var_names = tuple(sorted(variables, key=sort_key, reverse=True))
            var_names = tuple(reversed(s_var_names))
        elif var_selection_policy is VarSelectionPolicy.RANDOM:
            var_names = list(var_names)
            random.shuffle(var_names)
            var_names = tuple(var_names)
        elif var_selection_policy is VarSelectionPolicy.REVERSED:
            var_names = tuple(reversed(var_names))

        # print("::: var_names=", var_names)
        # for var_name in var_names:
        #     print("::: var_name={}".format(var_name))
        #     print(":::     domain={}".format(variables[var_name].domain))
        #     print(":::     var_constraints: [#{}]".format(len(model_info.var_constraints.get(var_name, []))))
        #     for c in model_info.var_constraints.get(var_name, []):
        #         print(":::         {}".format(c))

        stack = []
        domains = {}
        forbidden = {var_name: {} for var_name in var_names}
        stack.append((var_names, {}))

        timer = Timer()
        timer.start()
        while stack:
            if timeout is not None:
                cur_elapsed = timer.elapsed()
                if cur_elapsed > timeout:
                    timer.abort()
                    self._interrupt = "timeout"
                    return

            unbound_vars, substitution = stack[-1]

            var_name, unbound_vars = unbound_vars[0], unbound_vars[1:]

            substitution = substitution.copy()

            for c_var in var_ad_constraints[var_name]:
                forbidden[c_var].pop(var_name, None)

            domain = domains.get(var_name, None)
            if domain is None:
                fset = set(forbidden[var_name].values())
                domain = []
                for value in variables[var_name].domain:
                    if value not in fset:
                        substitution[var_name] = value
                        for constraint in var_constraints[var_name]:
                            if constraint.unsatisfied(substitution):
                                # var_name == value is *not* acceptable
                                # because it breaks the Constraint
                                break
                        else:
                            # var_name == value is acceptable
                            domain.append(value)
                if domain:
                    domains[var_name] = domain
                else:
                    stack.pop(-1)
                    continue
            elif not domain:
                domains.pop(var_name)
                # for c_var in var_ad_constraints[var_name]:
                #     forbidden[c_var].pop(var_name, None)
                stack.pop(-1)
                continue

            value = domain.pop(0)
            substitution[var_name] = value

            if unbound_vars:
                for c_var in var_ad_constraints[var_name].intersection(unbound_vars):
                    forbidden[c_var][var_name] = value
                stack.append((unbound_vars, substitution))
            else:
                timer.stop()
                # for constraint in model.constraints():
                #     print(constraint.evaluate(substitution), constraint)
                yield substitution
                if limit is not None and num_solutions >= limit:
                    timer.abort()
                    self._interrupt = "limit"
                    return
                timer.start()
                continue

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
            yield OptSolution(
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
        return OptSolution(
            is_optimal=is_optimal,
            solution=solution)

    def get_stats(self):
        stats = self._timer.stats()
        return SolveStats(
            count=stats.count,
            elapsed=stats.elapsed,
            interrupt=self._interrupt)


class ModelSolverBase(object):
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

