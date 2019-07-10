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
    'in_order',
    'min_bound',
    'max_bound',
    'min_domain',
    'max_domain',
    'group_prio',
    'min_value',
    'max_value',
]


ModelInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'ModelInfo',
    [
        'initial_domains',
        'reduced_domains',
        'domains',
        'var_names',
        'var_bounds',
        'var_constraints',
        'var_ad',
        'var_map',
        'var_group_prio',
        'groups',
        'var_groups',
    ]
)


OptimalSolution = collections.namedtuple(  # pylint: disable=invalid-name
    "OptimalSolution",
    "is_optimal solution")


def in_order(bound_var_names, unbound_var_names, model_info):
    var_name = unbound_var_names.pop(0)
    return var_name, unbound_var_names


def _sort_bound(reverse, bound_var_names, unbound_var_names, model_info):
    var_map = model_info.var_map
    dct = {}
    for var_name in unbound_var_names:
        if bound_var_names:
            other_var_names = bound_var_names
        else:
            other_var_names = filter(lambda v: v != var_name, model_info.var_names)
        count = 0
        for other_var_name in other_var_names:
            count += var_map[var_name][other_var_name]
        dct[var_name] = count
    unbound_var_names.sort(key=lambda v: dct[v], reverse=reverse)


def min_bound(bound_var_names, unbound_var_names, model_info):
    if len(bound_var_names) < 2:
        _sort_bound(False, bound_var_names, unbound_var_names, model_info)
    var_name = unbound_var_names.pop(0)
    return var_name, unbound_var_names


def max_bound(bound_var_names, unbound_var_names, model_info):
    if len(bound_var_names) < 2:
        _sort_bound(True, bound_var_names, unbound_var_names, model_info)
    var_name = unbound_var_names.pop(0)
    return var_name, unbound_var_names


def _sort_domain(reverse, bound_var_names, unbound_var_names, model_info):
    if bound_var_names:
        var_domains = model_info.domains
    else:
        var_domains = model_info.initial_domains
    unbound_var_names.sort(key=lambda v: len(var_domains[v]), reverse=reverse)


def min_domain(bound_var_names, unbound_var_names, model_info):
    if len(bound_var_names) < 2:
        _sort_domain(False, bound_var_names, unbound_var_names, model_info)
    var_name = unbound_var_names.pop(0)
    return var_name, unbound_var_names


def max_domain(bound_var_names, unbound_var_names, model_info):
    if len(bound_var_names) < 2:
        _sort_domain(True, bound_var_names, unbound_var_names, model_info)
    var_name = unbound_var_names.pop(0)
    return var_name, unbound_var_names


def group_prio(bound_var_names, unbound_var_names, model_info):
    if not bound_var_names:
        var_group_prio = model_info.var_group_prio
        var_bounds = model_info.var_bounds
        var_domains = model_info.initial_domains
        unbound_var_names.sort(key=lambda v: (var_group_prio[v], var_bounds[v], len(var_domains[v])))
    var_name = unbound_var_names.pop(0)
    return var_name, unbound_var_names


def min_value(var_name, substitution, reduced_domain):
    value = min(reduced_domain)
    reduced_domain.discard(value)
    return value, reduced_domain


def max_value(var_name, substitution, reduced_domain):
    value = max(reduced_domain)
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
        var_map = {var_name: collections.Counter() for var_name in var_names}
        var_bounds = collections.Counter()
        groups = {}
        var_groups = collections.defaultdict(list)
        for constraint in itertools.chain(model.constraints(), additional_constraints):
            if compile_constraints:
                constraint.compile()
            c_vars = set(filter(lambda v: v in var_names_set, constraint.vars()))
            for c_var in c_vars:
                var_bounds[c_var] += len(c_vars) - 1
            if isinstance(constraint, AllDifferentConstraint):
                groupid = len(groups)
                groups[groupid] = c_vars
                for var_name in c_vars:
                    var_groups[var_name].append(groupid)
                ad_var_names = set()
                for var_name in c_vars:
                    other_var_names = c_vars.difference({var_name})
                    var_ad[var_name].update(other_var_names)
                    for other_var_name in other_var_names:
                        var_map[var_name][other_var_name] += 1
            else:
                for var_name in c_vars:
                    var_constraints[var_name].append(constraint)
                    other_var_names = c_vars.difference({var_name})
                    for other_var_name in other_var_names:
                        var_map[var_name][other_var_name] += 1

        group_prio = {groupid: groupid for groupid in groups}  #idx for idx, groupid in enumerate(sorted(groups, key=lambda x: (group_bounds[x], group_sizes[x]), reverse=True))}
        var_group_prio = {}
        for var_name in var_names:
            if var_groups[var_name]:
                var_group_prio[var_name] = min(group_prio[groupid] for groupid in var_groups[var_name])
            else:
                var_group_prio[var_name] = -1

        reduced_domains = {}
        domains = collections.ChainMap(reduced_domains, initial_domains)
        model_info = ModelInfo(
            initial_domains=initial_domains,
            reduced_domains=reduced_domains,
            domains=domains,
            var_names=var_names,
            var_bounds=var_bounds,
            var_constraints=var_constraints,
            var_ad=var_ad,
            var_map=var_map,
            var_group_prio=var_group_prio,
            groups=groups,
            var_groups=var_groups,
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
            # print("@-0", var_name)
            #print(var_name, unbound_var_names, substitution)
            substitution = substitution.copy()
            reduced_domain = reduced_domains.get(var_name, None)
            if reduced_domain is None:
                bound_var_set = set(bound_var_names)
                forbidden_values = {substitution[vname] for vname in var_ad[var_name].intersection(substitution)}
                reduced_domain = set()
                
                use_constraints = set()
                for constraint in var_constraints[var_name]:
                    if constraint.vars() <= bound_var_set:
                        use_constraints.add(constraint)

                for value in initial_domains[var_name]:
                    if value in forbidden_values:
                        continue
                    substitution[var_name] = value
                    for constraint in use_constraints:
                        if not constraint.evaluate(substitution):
                            break
                    else:
                        reduced_domain.add(value)
                if reduced_domain:
                    reduced_domains[var_name] = reduced_domain
                else:
                    stack.pop(-1)
                    # print("@-1")
                    #print("A")
                    continue
            elif not reduced_domain:
                reduced_domains.pop(var_name)
                stack.pop(-1)
                # print("@-2", var_name)
                #print("B")
                continue
            # REM print("   ", var_name, unbound_var_names, reduced_domain)

            #print("{} -> {}".format(var_name, reduced_domain))
            #input("...")
            # select value:
            value, reduced_domain = select_value(var_name, substitution, reduced_domain)
            # print("@-3", var_name, value)
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
                # print("@-4", substitution)
                for constraint in model.constraints():
                    if constraint.unsatisfied(substitution):
                        print("!!!", constraint)
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

