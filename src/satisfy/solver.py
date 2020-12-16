import abc
import collections
import itertools
import random

from .constraint import AllDifferentConstraint
from .model import Model
from .objective import Objective
from .utils import INFINITY, Timer, SolveStats


__all__ = [
    'ModelInfo',
    'OptimalSolution',
    'SelectVar',
    'SelectValue',
    'Solver',
    'ModelSolver',
    'ModelOptimizer',
]


ModelInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'ModelInfo',
    [
        'solvable',
        'original_domains',
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
        'extra',
    ]
)


OptimalSolution = collections.namedtuple(  # pylint: disable=invalid-name
    "OptimalSolution",
    "is_optimal solution")


class SelectVar:
    @classmethod
    def random(cls, bound_var_names, unbound_var_names, model_info):
        idx = random.randrange(0, len(unbound_var_names))
        var_name = unbound_var_names.pop(idx)
        return var_name, unbound_var_names

    @classmethod
    def in_order(cls, bound_var_names, unbound_var_names, model_info):
        var_name = unbound_var_names.pop(0)
        return var_name, unbound_var_names

    @classmethod
    def _sort_bound(cls, bound_var_names, unbound_var_names, model_info, minmax):
        var_map = model_info.var_map
        for var_name in unbound_var_names:
            if bound_var_names:
                other_var_names = bound_var_names
            else:
                other_var_names = filter(lambda v: v != var_name, model_info.var_names)
        unbound_var_names.sort(key=lambda v: minmax(var_map[v].values()))
    
    @classmethod
    def min_bound(cls, bound_var_names, unbound_var_names, model_info):
        if len(bound_var_names) == 0:
            cls._sort_bound(bound_var_names, unbound_var_names, model_info, min)
        var_name = unbound_var_names.pop(0)
        return var_name, unbound_var_names
    
    @classmethod
    def max_bound(cls, bound_var_names, unbound_var_names, model_info):
        if len(bound_var_names) == 0:
            cls._sort_bound(bound_var_names, unbound_var_names, model_info, max)
        var_name = unbound_var_names.pop(0)
        return var_name, unbound_var_names
    
    @classmethod
    def _sort_domain(cls, bound_var_names, unbound_var_names, model_info):
        if bound_var_names:
            var_domains = model_info.domains
        else:
            var_domains = model_info.original_domains
        unbound_var_names.sort(key=lambda v: len(var_domains[v]))
    
    @classmethod
    def min_domain(cls, bound_var_names, unbound_var_names, model_info):
        if len(bound_var_names) < 1:
            cls._sort_domain(bound_var_names, unbound_var_names, model_info)
        var_name = unbound_var_names.pop(0)
        return var_name, unbound_var_names
    
    @classmethod
    def max_domain(cls, bound_var_names, unbound_var_names, model_info):
        if len(bound_var_names) < 1:
            cls._sort_domain(bound_var_names, unbound_var_names, model_info)
        var_name = unbound_var_names.pop(-1)
        return var_name, unbound_var_names
    
    @classmethod
    def group_prio(cls, bound_var_names, unbound_var_names, model_info):
        if not bound_var_names:
            var_group_prio = model_info.var_group_prio
            var_bounds = model_info.var_bounds
            var_domains = model_info.initial_domains
            unbound_var_names.sort(key=lambda v: (var_group_prio[v], var_bounds[v], len(var_domains[v])))
        var_name = unbound_var_names.pop(0)
        return var_name, unbound_var_names


class SelectValue:
    @classmethod
    def random(cls, var_name, substitution, reduced_domain):
        value = random.sample(reduced_domain, 1)[0]
        reduced_domain.discard(value)
        return value, reduced_domain

    @classmethod
    def min_value(cls, var_name, substitution, reduced_domain):
        value = min(reduced_domain)
        reduced_domain.discard(value)
        return value, reduced_domain
    
    @classmethod
    def max_value(cls, var_name, substitution, reduced_domain):
        value = max(reduced_domain)
        reduced_domain.discard(value)
        return value, reduced_domain


class Solver(object):
    def __init__(self,
                 select_var=SelectVar.max_bound,
                 select_value=SelectValue.min_value,
                 timeout=None,
                 limit=None,
                 reduce_max_depth=1,
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
        self._reduce_max_depth = None
        self.reduce_max_depth = reduce_max_depth
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

    @property
    def reduce_max_depth(self):
        return self._reduce_max_depth

    @reduce_max_depth.setter
    def reduce_max_depth(self, value):
        self._reduce_max_depth = int(value)

    def make_model_info(self, model, *, _additional_constraints=(), **args):
        compile_constraints = args.get('compile_constraints', self._compile_constraints)
        reduce_max_depth = args.get('reduce_max_depth', self._reduce_max_depth)

        # 1. internal data structures:
        variables = model.variables()
        original_domains = {
            var_name: list(var_info.domain) for var_name, var_info in variables.items() if var_info.domain is not None
        }
        initial_domains = {
            var_name: list(var_info.domain) for var_name, var_info in variables.items() if var_info.domain is not None
        }
        var_names = list(initial_domains)
        var_names_set = set(var_names)
        var_constraints = {var_name: [] for var_name in var_names}
        var_ad = {var_name: set() for var_name in var_names}
        var_map = {var_name: collections.Counter() for var_name in var_names}
        var_bounds = collections.Counter()
        groups = {}
        var_groups = collections.defaultdict(list)
        solvable = True
        for constraint in itertools.chain(model.constraints(), _additional_constraints):
            if compile_constraints:
                constraint.compile()
            c_vars = set(filter(lambda v: v in var_names_set, constraint.vars()))
            accepted = True
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
                if reduce_max_depth >= 0 and len(c_vars) == 0:
                    b_value = constraint.evaluate({})
                    # print("discard {}: always -> {}".format(constraint, b_value))
                    if not b_value:
                        solvable = False
                    accepted = False
                elif reduce_max_depth >= 1 and len(c_vars) == 1:
                    c_var = list(c_vars)[0]
                    var_domain = []
                    for value in initial_domains[c_var]:
                        if constraint.evaluate({c_var: value}):
                            var_domain.append(value)
                    if initial_domains[c_var] != var_domain:
                        # print("discard {}: domain reduced: {} -> {}".format(constraint, initial_domains[c_var], var_domain))
                        initial_domains[c_var] = var_domain
                    # else:
                    #     print("discard {}: always True on domain {}".format(constraint, initial_domains[c_var]))
                    accepted = False
                for var_name in c_vars:
                    var_constraints[var_name].append(constraint)
                    other_var_names = c_vars.difference({var_name})
                    for other_var_name in other_var_names:
                        var_map[var_name][other_var_name] += 1
            if accepted:
                for c_var in c_vars:
                    var_bounds[c_var] += len(c_vars) - 1

        group_prio = {groupid: groupid for groupid in groups}  #idx for idx, groupid in enumerate(sorted(groups, key=lambda x: (group_bounds[x], group_sizes[x]), reverse=True))}
        var_group_prio = {}
        for var_name in var_names:
            if var_groups[var_name]:
                var_group_prio[var_name] = min(group_prio[groupid] for groupid in var_groups[var_name])
            else:
                var_group_prio[var_name] = -1

        reduced_domains = {}
        domains = collections.ChainMap(reduced_domains, initial_domains)
        return ModelInfo(
            solvable=solvable,
            original_domains=original_domains,
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
            extra={}
        )

    def solve(self, model, *, _additional_constraints=(), **args):
        select_var = args.get('select_var', self._select_var)
        select_value = args.get('select_value', self._select_value)
        timeout = args.get('timeout', self._timeout)
        limit = args.get('limit', self._limit)

        # 1. make model_info:
        model_info = self.make_model_info(model, _additional_constraints=_additional_constraints, **args)

        if not (model_info.solvable and model_info.var_names):
            return

        var_names = model_info.var_names
        reduced_domains = model_info.reduced_domains
        initial_domains = model_info.initial_domains
        var_ad = model_info.var_ad
        var_constraints = model_info.var_constraints

        # 2. solve:
        stack = []
        if var_names:
            var_name, unbound_var_names = select_var([], var_names, model_info)
            # o = model_info.original_domains
            # i = model_info.initial_domains
            # m = model_info.var_map
            # for v in [var_name] + unbound_var_names:
            #     print("{:16s} {:4d} {:4d} {:4d}".format(v, len(o[v]), len(i[v]), sum(m[v].values())))
            # input("===!!!===")
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
            substitution = substitution.copy()
            reduced_domain = reduced_domains.get(var_name, None)
            if reduced_domain is None:
                bound_var_set = set(bound_var_names)
                forbidden_values = {value for vname, value in substitution.items() if vname in var_ad[var_name]}

                reduced_domain = set(initial_domains[var_name]).difference(forbidden_values)
                # if 0 and True:
                #     c_functions = []
                #     for constraint in var_constraints[var_name]:
                #         if constraint.vars() <= bound_var_set:
                #             c_functions.append(constraint.evaluate)
                #     #print("***", var_name, len(var_constraints[var_name]), len(c_functions))
                #     if c_functions:
                #         nr_dom = set()
                #         for value in reduced_domain:
                #             substitution[var_name] = value
                #             # print(var_name, value, substitution)
                #             # for ccc in c_functions:
                #             #     print("   +", ccc, ccc.evaluate(substitution))
                #             for c_fun in c_functions:
                #                 if not c_fun(substitution):
                #                     break
                #             else:
                #                 nr_dom.add(value)
                #         # input("///")
                #         reduced_domain = nr_dom
                # else:
                for constraint in var_constraints[var_name]:
                    if constraint.vars() <= bound_var_set:
                        c_fun = constraint.evaluate
                        nr_dom = set()
                        for value in reduced_domain:
                            substitution[var_name] = value
                            if c_fun(substitution):
                                nr_dom.add(value)
                        reduced_domain = nr_dom
                        if not reduced_domain:
                            break
                if reduced_domain:
                    reduced_domains[var_name] = reduced_domain
                else:
                    stack.pop(-1)
                    continue
            elif not reduced_domain:
                reduced_domains.pop(var_name)
                stack.pop(-1)
                continue
            # select value:
            value, reduced_domain = select_value(var_name, substitution, reduced_domain)
            substitution[var_name] = value
            if unbound_var_names:
                unbound_var_names = list(unbound_var_names)
                next_var_name, next_unbound_var_names = select_var(bound_var_names, unbound_var_names, model_info)
                stack.append((bound_var_names + [next_var_name], next_unbound_var_names, substitution))
            else:
                timer.stop()
                num_solutions += 1
                for constraint in model.constraints():
                    if constraint.unsatisfied(substitution):
                        raise RuntimeError("constraint {} is not satisfied".format(constraint))
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
        args['_additional_constraints'] = tuple(args.get('_additional_constraints', ())) + tuple(objective_constraints)
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

