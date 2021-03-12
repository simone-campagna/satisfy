import abc
import collections
import contextlib
import enum
import itertools
import logging
import random

from .constraint import AllDifferentConstraint, ExpressionConstraint
from .expression import set_expression_globals, EXPRESSION_GLOBALS
from .objective import Objective, ObjectiveConstraint
from .utils import INFINITY, Timer


__all__ = [
    'ModelInfo',
    'OptimizationResult',
    'SelectVar',
    'SelectValue',
    'Solver',
    'State',
    'SolverState',
    'ModelSolver',
]

LOG = logging.getLogger(__name__)


ModelInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'ModelInfo',
    [
        'model',
        'solver',
        'algorithm',
        'solvable',
        'objective_functions',
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
        'initial_substitution',
        'extra',
    ]
)


OptimizationResult = collections.namedtuple(  # pylint: disable=invalid-name
    "OptimizationResult",
    "is_optimal state trials_count solutions_count solution")


class SelectNamespace:
    def __init__(self):
        self.__names = []

    def __entries__(self):
        yield from self.__names

    def __register__(self, name, *aliases):
        def register_decorator(cls):
            for key in itertools.chain([name], aliases):
                if key in self.__names:
                    raise ValueError("{}: redefined {}".format(type(self).__name__, key))
                self.__dict__[key] = cls(key)
                self.__names.append(key)
            return cls
        return register_decorator


class SelectVarNamespace(SelectNamespace):
    pass


class SelectValueNamespace(SelectNamespace):
    pass


SelectVar = SelectVarNamespace()
SelectValue = SelectValueNamespace()


class Algorithm(enum.Enum):
    auto = 0
    reduce = 1
    propagate = 2


class Selector(abc.ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class ValueSelector(Selector):
    def init(self, domain):
        pass

    @abc.abstractmethod
    def __call__(self, var_name, substitution, domain):
        raise NotImplementedError()

    def __repr__(self):
        return 'SelectValue.{}'.format(self.name)


@SelectValue.__register__('in_order')
class InOrderValueSelector(ValueSelector):
    def init(self, domain):
        domain.reverse()

    def __call__(self, var_name, substitution, domain):
        value = domain.pop(-1)
        return value, domain


@SelectValue.__register__('random')
class RandomValueSelector(ValueSelector):
    def init(self, domain):
        random.shuffle(domain)

    def __call__(self, var_name, substitution, domain):
        value = domain.pop(-1)
        return value, domain


@SelectValue.__register__('min_value')
class MinValueSelector(ValueSelector):
    def init(self, domain):
        domain.sort(reverse=True)

    def __call__(self, var_name, substitution, domain):
        value = domain.pop(-1)
        return value, domain


@SelectValue.__register__('max_value')
class MaxValueSelector(ValueSelector):
    def init(self, domain):
        domain.sort()

    def __call__(self, var_name, substitution, domain):
        value = domain.pop(-1)
        return value, domain


class VarSelector(Selector):
    @abc.abstractmethod
    def init(self, unbound_var_names, model_info):
        raise NotImplementedError()

    def preferred_algorithm(self):
        return Algorithm.auto

    def __call__(self, substitution, bound_var_names, unbound_var_names, model_info):
        var_name = self.select_var(substitution, bound_var_names, unbound_var_names, model_info)
        bound_var_names.append(var_name)
        enabled_constraints = self.select_enabled_constraints(substitution, bound_var_names, unbound_var_names, model_info)
        return var_name, bound_var_names, unbound_var_names, enabled_constraints

    @abc.abstractmethod
    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        raise NotImplementedError()

    @abc.abstractmethod
    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        raise NotImplementedError()

    @abc.abstractmethod
    def select_enabled_constraints(self, substitution, bound_var_names, unbound_var_names, model_info):
        raise NotImplementedError()

    def __repr__(self):
        return 'SelectVar.{}'.format(self.name)


class StaticVarSelector(VarSelector):
    def init(self, unbound_var_names, model_info):
        join_constraints = False  # no difference in performances
        self.sort_var_names({}, [], unbound_var_names, model_info)
        self.constraints = [[]]
        var_constraints = model_info.var_constraints
        s_bound_var_names = set()
        compile_constraints = model_info.solver.compile_constraints
        for var_name in reversed(unbound_var_names):
            s_bound_var_names.add(var_name)
            var_clist = []
            for constraint in var_constraints[var_name]:
                if constraint.vars <= s_bound_var_names:
                    var_clist.append(constraint)
            if join_constraints and len(var_clist) > 1:
                new_var_clist = []
                var_elist = []
                for var_c in var_clist:
                    if isinstance(var_c, ExpressionConstraint) and not isinstance(var_c, ObjectiveConstraint):
                        var_elist.append(var_c.expression)
                    else:
                        new_var_clist.append(var_c)
                if var_elist:
                    var_e = var_elist[0]
                    for var_e1 in var_elist[1:]:
                        var_e &= var_e1
                    new_var_c = ExpressionConstraint(var_e)
                    if compile_constraints:
                        new_var_c.compile()
                    new_var_clist.append(new_var_c)
                var_clist = new_var_clist
            self.constraints.append(var_clist)

    def select_enabled_constraints(self, substitution, bound_var_names, unbound_var_names, model_info):
        return self.constraints[len(bound_var_names)]

    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


class DynamicVarSelector(VarSelector):
    def init(self, unbound_var_names, model_info):
        pass

    def select_enabled_constraints(self, substitution, bound_var_names, unbound_var_names, model_info):
        var_name = bound_var_names[-1]
        var_constraints = model_info.var_constraints[var_name]
        enabled_constraints = []
        var_set = set(bound_var_names)
        for constraint in var_constraints:
            if constraint.vars <= var_set:
                enabled_constraints.append(constraint)
        return enabled_constraints


@SelectVar.__register__('in_order')
class InOrderVarSelector(StaticVarSelector):
    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        unbound_var_names.reverse()


@SelectVar.__register__('random')
class RandomVarSelector(StaticVarSelector):
    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        random.shuffle(unbound_var_names)


def iterator_len(it):
    count = 0
    for dummy in it:
        count += 1
    return count


def count_min(values):
    if values:
        return min(values)
    else:
        return 0


def count_max(values):
    if values:
        return max(values)
    else:
        return 0


class BoundVarSelector(StaticVarSelector):
    __key_function__ = None
    __reverse__ = None

    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        var_map = model_info.var_map
        key_function = self.__class__.__key_function__
        if self.__reverse__:
            unbound_var_names.reverse()  # stable sort!
        unbound_var_names.sort(key=lambda v: key_function(var_map[v].values()), reverse=self.__reverse__)


@SelectVar.__register__('min_boundmin')
class Min_BoundMinSelector(BoundVarSelector):
    __key_function__ = count_min
    __reverse__ = True


@SelectVar.__register__('min_boundmax', 'min_bound')
class Min_BoundMaxSelector(BoundVarSelector):
    __key_function__ = count_max
    __reverse__ = True


@SelectVar.__register__('min_boundsum')
class Min_BoundSumSelector(BoundVarSelector):
    __key_function__ = sum
    __reverse__ = True


@SelectVar.__register__('min_boundlen')
class Min_BoundLenSelector(BoundVarSelector):
    __key_function__ = iterator_len
    __reverse__ = True


@SelectVar.__register__('max_boundmin', 'max_bound')
class Max_BoundMinSelector(BoundVarSelector):
    __key_function__ = count_min
    __reverse__ = False


@SelectVar.__register__('max_boundmax')
class Max_BoundMaxSelector(BoundVarSelector):
    __key_function__ = count_max
    __reverse__ = False


@SelectVar.__register__('max_boundsum')
class Max_BoundSumSelector(BoundVarSelector):
    __key_function__ = sum
    __reverse__ = False


@SelectVar.__register__('max_boundlen')
class Max_BoundLenSelector(BoundVarSelector):
    __key_function__ = iterator_len
    __reverse__ = False


class DomainVarSelector(DynamicVarSelector):
    def init(self, unbound_var_names, model_info):
        pass

    def preferred_algorithm(self):
        return Algorithm.propagate

    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info, reverse):
        if reverse:
            unbound_var_names.reverse()  # stable sort
        domains = model_info.domains
        unbound_var_names.sort(key=lambda v: len(domains[v]), reverse=reverse)


@SelectVar.__register__('min_domain')
class MinDomainVarSelector(DomainVarSelector):
    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        self.sort_var_names(substitution, bound_var_names, unbound_var_names, model_info, reverse=True)
        return unbound_var_names.pop(-1)


@SelectVar.__register__('max_domain')
class MaxDomainVarSelector(DomainVarSelector):
    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        self.sort_var_names(substitution, bound_var_names, unbound_var_names, model_info, reverse=False)
        return unbound_var_names.pop(-1)


@SelectVar.__register__('min_alphanumeric')
class MinAlphanumericVarSelector(StaticVarSelector):
    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        unbound_var_names.sort(reverse=True)

    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


@SelectVar.__register__('max_alphanumeric')
class MaxAlphanumericVarSelector(StaticVarSelector):
    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        unbound_var_names.sort(reverse=False)

    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


class ActivationVarSelector(StaticVarSelector):
    __reverse__ = False

    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        c_free_vars = {}
        for constraint in model_info.model.constraints():
            c_free_vars[constraint] = set(constraint.vars)

        v_constraints = {}
        for var_name, constraints in model_info.var_constraints.items():
            v_constraints[var_name] = set(constraints)

        unbound_var_set = set(unbound_var_names)
        unbound_var_names.clear()
        while unbound_var_set:
            # 1 select free constraints
            if not c_free_vars:
                break
            key_fn = lambda c: len(c_free_vars[c])
            c_data = list(c_free_vars)
            c_data.sort(key=key_fn)
            _, c_group = next(iter(itertools.groupby(c_data, key=key_fn)))

            # 2 select vars
            v_set = set()
            for constraint in c_group:
                v_set.update(c_free_vars[constraint])
            key_fn = lambda v: len(v_constraints[v])
            var_name = sorted(v_set, key=key_fn)[0]
            unbound_var_names.append(var_name)

            # 3 update data
            unbound_var_set.discard(var_name)
            del v_constraints[var_name]
            del_constraints = []
            for constraint, c_vars in c_free_vars.items():
                c_vars.discard(var_name)
                if not c_vars:
                    del_constraints.append(constraint)
            for constraint in del_constraints:
                del c_free_vars[constraint]
        if self.__reverse__:
            unbound_var_names.reverse()


@SelectVar.__register__('min_activation')
class MinActivationVarSelector(ActivationVarSelector):
    __reverse__ = False


@SelectVar.__register__('max_activation')
class MaxActivationVarSelector(ActivationVarSelector):
    __reverse__ = True


@SelectVar.__register__('group_prio')
class GroupPrioVarSelector(StaticVarSelector):
    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        var_group_prio = model_info.var_group_prio
        var_bounds = model_info.var_bounds
        var_domains = model_info.initial_domains
        unbound_var_names.sort(key=lambda v: (var_group_prio[v], var_bounds[v], len(var_domains[v])), reverse=True)

    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


@SelectVar.__register__('dyn_group_prio')
class DynGroupPrioVarSelector(DynamicVarSelector):
    def preferred_algorithm(self):
        return Algorithm.propagate

    def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
        var_group_prio = model_info.var_group_prio
        var_bounds = model_info.var_bounds
        var_domains = model_info.domains
        # unbound_var_names.sort(key=lambda v: (len(var_domains[v]), var_group_prio[v], var_bounds[v]), reverse=True)
        unbound_var_names.sort(key=lambda v: (var_group_prio[v], var_bounds[v], len(var_domains[v])), reverse=True)

    def select_var(self, substitution, bound_var_names, unbound_var_names, model_info):
        self.sort_var_names(substitution, bound_var_names, unbound_var_names, model_info)
        return unbound_var_names.pop(-1)


# class DistanceVarSelector(StaticVarSelector):
#     __reverse__ = None
#
#     def sort_var_names(self, substitution, bound_var_names, unbound_var_names, model_info):
#         bvars = []
#         var_constraints = model_info.var_constraints
#         value = {v: 0.0 for v in unbound_var_names}
#         while unbound_var_names:
#             for constraint in model_info.constraints:
#                 c_vars = constraint.vars
#                 if len(c_vars) == 1:
#                     value[list(c_vars)[0]] += 1
#                 elif c_vars:
#                     for v0, v1 in itertools.combinations(c_vars, 2):
#                         vd = 1.0 / len(c_vars)
#                         value[v0] += vd
#                         value[v1] += vd
#             unbound_var_names.sort(key=value.__getitem__, reverse=self.__reverse__)
#             var_name = unbound_var_names.pop(-1)
#             bvars.append(var_name)
#         bvars.reverse()
#         unbound_var_names.clear()
#         unbound_var_names.extend(bvars)
#
#
# @SelectVar.__register__('min_distance')
# class MinDistanceVarSelector(DistanceVarSelector):
#     __reverse__ = True
#
#
# @SelectVar.__register__('max_distance')
# class MaxDistanceVarSelector(DistanceVarSelector):
#     __reverse__ = False


class Solver:
    def __init__(self,
                 select_var=SelectVar.max_boundmin,
                 select_value=SelectValue.min_value,
                 algorithm=Algorithm.auto,
                 timeout=None,
                 limit=None,
                 reduce_max_depth=1,
                 discard_const_vars=False,
                 compile_constraints=True):
        self._select_var = None
        self.select_var = select_var
        self._select_value = None
        self.select_value = select_value
        self._algorithm = None
        self.algorithm = algorithm
        self._timeout = None
        self.timeout = timeout
        self._limit = None
        self.limit = limit
        self._discard_const_vars = None
        self.discard_const_vars = discard_const_vars
        self._compile_constraints = None
        self.compile_constraints = compile_constraints
        self._reduce_max_depth = None
        self.reduce_max_depth = reduce_max_depth

    @property
    def select_var(self):
        return self._select_var

    @select_var.setter
    def select_var(self, function):
        self._select_var = function

    @property
    def select_value(self):
        return self._select_value

    @select_value.setter
    def select_value(self, function):
        self._select_value = function

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        if isinstance(value, str):
            value = Algorithm[value]
        if not isinstance(value, Algorithm):
            raise TypeError(value)
        self._algorithm = value

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
    def discard_const_vars(self):
        return self._discard_const_vars

    @discard_const_vars.setter
    def discard_const_vars(self, value):
        self._discard_const_vars = bool(value)

    @property
    def reduce_max_depth(self):
        return self._reduce_max_depth

    @reduce_max_depth.setter
    def reduce_max_depth(self, value):
        self._reduce_max_depth = int(value)

    @contextlib.contextmanager
    def __call__(self, model):
        with set_expression_globals(model.globals, merge=True):
            yield ModelSolver(model, self)


class State(enum.Enum):
    RUNNING = 0
    DONE = 1
    INTERRUPT_TIMEOUT = 2
    INTERRUPT_LIMIT = 3


class SolverState:
    def __init__(self):
        self._timer = Timer()
        self._trials_count = 0
        self._solutions_count = 0
        self._current_solution = None
        self._state = State.RUNNING
        self.stats = self._timer.stats

    @property
    def state(self):
        return self._state

    @property
    def timer(self):
        return self._timer

    @property
    def solution(self):
        return self._current_solution

    @property
    def trials_count(self):
        return self._trials_count

    @property
    def solutions_count(self):
        return self._solutions_count

    @state.setter
    def state(self, value):
        if not isinstance(value, State):
            raise TypeError(value)
        self._state = value

    def add_try(self):
        self._trials_count += 1

    def add_solution(self, solution):
        self._current_solution = solution
        self._solutions_count += 1


class ModelSolver:
    def __init__(self, model, solver):
        self._model = model
        self._solver = solver
        compile_constraints = solver.compile_constraints
        reduce_max_depth = solver.reduce_max_depth
        self._state = SolverState()
        select_var = solver.select_var
        select_value = solver.select_value
        discard_const_vars = solver.discard_const_vars

        # 0. objectives:
        objective_functions = []
        additional_constraints = []
        for objective_function in model.objective_functions():
            objective_constraints = list(objective_function.constraints)
            additional_constraints.extend(objective_constraints)
            #print("+++", objective_constraints)
            objective_functions.append(objective_function)

        # 1. internal data structures:
        variables = model.variables()
        original_domains = {}
        initial_domains = {}
        for var_name, var_info in variables.items():
            if var_info.domain is not None:
                original_domains[var_name] = var_info.domain
                initial_domain = list(var_info.domain)
                initial_domains[var_name] = initial_domain

        var_names = list(initial_domains)
        var_names_set = set(var_names)
        var_constraints = {var_name: [] for var_name in var_names}
        var_ad = {var_name: set() for var_name in var_names}
        var_map = {var_name: collections.Counter() for var_name in var_names}
        var_bounds = collections.Counter()
        groups = {}
        var_groups = collections.defaultdict(list)
        solvable = True
        for objective_function in objective_functions:
            objective_function.compile()
        for constraint in itertools.chain(model.constraints(), additional_constraints):
            if compile_constraints:
                constraint.compile()
            c_vars = set(filter(lambda v: v in var_names_set, constraint.vars))
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
                if not constraint.is_externally_updated():
                    if reduce_max_depth >= 0 and len(c_vars) == 0:
                        b_value = constraint.function({})
                        if not b_value:
                            solvable = False
                        accepted = False
                    elif reduce_max_depth >= 1 and len(c_vars) == 1:
                        c_var = list(c_vars)[0]
                        var_domain = []
                        for value in initial_domains[c_var]:
                            if constraint.function({c_var: value}):
                                var_domain.append(value)
                        if initial_domains[c_var] != var_domain:
                            initial_domains[c_var] = var_domain
                        accepted = False
                for var_name in c_vars:
                    var_constraints[var_name].append(constraint)
                    other_var_names = c_vars.difference({var_name})
                    for other_var_name in other_var_names:
                        var_map[var_name][other_var_name] += 1
            if accepted:
                for c_var in c_vars:
                    var_bounds[c_var] += len(c_vars) - 1

        ### domain reduction:
        #   if a variable's domain has a single value, apply
        #   all_different constraint to reduce other variables' domain
        vars_set = set(initial_domains)
        discarded_var_names = set()
        while vars_set:
            v_forbidden = collections.defaultdict(set)
            reduced_vars = set()
            for var1 in vars_set:
                domain = initial_domains[var1]
                if len(domain) == 1:
                    reduced_vars.add(var1)
                    value = list(domain)[0]
                    for var2 in var_ad[var1]:
                        v_forbidden[var2].add(value)
            if not reduced_vars:
                break
            discarded_var_names.update(reduced_vars)
            vars_set -= reduced_vars
            for var, forbidden_values in v_forbidden.items():
                if forbidden_values:
                    initial_domains[var] = [value for value in initial_domains[var] if value not in forbidden_values]

        initial_substitution = {}
        if discard_const_vars and discarded_var_names:
            for var_name in discarded_var_names:
                var_names.remove(var_name)
                var_constraints.pop(var_name, None)
                var_ad.pop(var_name, None)
                var_map.pop(var_name, None)
                var_bounds.pop(var_name, None)
                var_groups.pop(var_name, None)
                domain = initial_domains.pop(var_name)
                initial_substitution[var_name] = list(domain)[0]

            for var_name in var_names_set.difference(discarded_var_names):
                var_ad[var_name].difference_update(discarded_var_names)
                for discarded_var_name in discarded_var_names:
                    discarded_bounds = var_map[var_name].pop(discarded_var_name, 0)
                    var_bounds[var_name] -= discarded_bounds

        group_prio = {groupid: groupid for groupid in groups}  #idx for idx, groupid in enumerate(sorted(groups, key=lambda x: (group_bounds[x], group_sizes[x]), reverse=True))}
        var_group_prio = {}
        for var_name in var_names:
            if var_groups[var_name]:
                var_group_prio[var_name] = min(group_prio[groupid] for groupid in var_groups[var_name])
            else:
                var_group_prio[var_name] = -1

        algorithm = solver.algorithm
        if algorithm is Algorithm.auto:
            algorithm = select_var.preferred_algorithm()
        if algorithm is Algorithm.auto:
            algorithm = Algorithm.reduce  ### TODO FIXME

        reduced_domains = {}
        domains = collections.ChainMap(reduced_domains, initial_domains)
        self._model_info = ModelInfo(
            model=model,
            solver=solver,
            algorithm=algorithm,
            solvable=solvable,
            objective_functions=objective_functions,
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
            initial_substitution=initial_substitution,
            extra={}
        )
        if False:
            import json
            def _sorted_dset(dct):
                return {key: sorted(value) for key, value in dct.items()}

            dct = {
                'solvable': solvable,
                # 'original_domains': _sorted_dset(original_domains),
                'initial_domains': _sorted_dset(initial_domains),
                'reduced_domains': _sorted_dset(reduced_domains),
                'var_names': var_names,
                'var_bounds': var_bounds,
                'var_constraints': var_constraints,
                'var_ad': _sorted_dset(var_ad),
                'var_map': var_map,
                # 'var_group_prio': var_group_prio,
                # 'groups': _sorted_dset(groups),
                # 'var_groups': var_groups,
                'initial_substitution': initial_substitution,
            }
            print(json.dumps(dct, indent=4, sort_keys=1))

    @property
    def model(self):
        return self._model

    @property
    def model_info(self):
        return self._model_info

    @property
    def solver(self):
        return self._solver

    @property
    def state(self):
        return self._state

    @property
    def stats(self):
        return self._state.stats

    def get_optimization_result(self):
        if self._model.has_objectives():
            solution = self._state.solution
            state = self._state.state
            if solution is None:
                is_optimal = False
            else:
                is_optimal = state is State.DONE
            return OptimizationResult(
                is_optimal=is_optimal,
                state=state,
                trials_count=self._state.trials_count,
                solutions_count=self._state.solutions_count,
                solution=self._state.solution)

    def __iter__(self):
        # 1. set vars:
        model = self._model
        solver = self._solver
        model_info = self._model_info
        select_var = solver.select_var
        select_value = solver.select_value
        timeout = solver.timeout
        limit = solver.limit
        state = self._state
        timer = state.timer
        solution_transformer = model.transform_solution
        algorithm = model_info.algorithm
        propagate = algorithm is Algorithm.propagate
        # print("propagate:", propagate)

        if not (model_info.solvable and model_info.var_names):
            state.state = State.DONE
            return

        var_names = model_info.var_names
        reduced_domains = model_info.reduced_domains
        initial_domains = model_info.initial_domains
        var_ad = model_info.var_ad
        var_constraints = model_info.var_constraints
        objective_functions = model_info.objective_functions
        initial_substitution = model_info.initial_substitution

        # 2. solve:
        timer.start()
        stack = []
        if var_names:
            unbound_var_names = list(var_names)
            select_var.init(unbound_var_names, model_info)
            var_name, bound_var_names, unbound_var_names, enabled_constraints = select_var(initial_substitution, [], unbound_var_names, model_info)
            stack.append((var_name, bound_var_names, unbound_var_names, enabled_constraints, initial_substitution.copy()))
            for var_name in var_names:
                initial_domain = model_info.initial_domains[var_name]
                select_value.init(initial_domain)

        while stack:
            if timeout is not None:
                cur_elapsed = timer.elapsed()
                if cur_elapsed > timeout:
                    state.state = State.INTERRUPT_TIMEOUT
                    return

            var_name, bound_var_names, unbound_var_names, enabled_constraints, substitution = stack[-1]
            # input("::: var={!r} substitution={}".format(var_name, substitution))
            substitution = substitution.copy()
            reduced_domain = reduced_domains.get(var_name, None)
            if reduced_domain is None:
                _propagate([var_name], None, substitution, model_info)
                reduced_domain = reduced_domains[var_name]
            value_found = False
            # print("  " * len(substitution), "===", var_name, substitution, reduced_domain)
            while reduced_domain:
                # select value:
                value, reduced_domain = select_value(var_name, substitution, reduced_domain)
                substitution[var_name] = value
                state.add_try()
                if propagate:
                    if not _propagate(unbound_var_names, var_name, substitution, model_info):
                        # reduced_domains.pop(var_name, None)
                        for unbound_var_name in unbound_var_names:
                            reduced_domains.pop(unbound_var_name, None)
                        continue
                value_found = True
                break
            if not value_found:
                substitution.pop(var_name, None)
                reduced_domains.pop(var_name, None)
                for unbound_var_name in unbound_var_names:
                    reduced_domains.pop(unbound_var_name, None)
                stack.pop(-1)
                continue
            # print(var_name, substitution, EXPRESSION_GLOBALS['__parameters__'])
            if unbound_var_names:
                next_var_name, next_bound_var_names, next_unbound_var_names, next_enabled_constraints = select_var(substitution, list(bound_var_names), list(unbound_var_names), model_info)
                stack.append((next_var_name, next_bound_var_names, next_unbound_var_names, next_enabled_constraints, substitution))
            else:
                timer.stop()
                state.add_solution(substitution)
                for constraint in model.constraints():
                    if constraint.unsatisfied(substitution):
                        raise RuntimeError("constraint {} is not satisfied by {}".format(constraint, substitution))
                yield solution_transformer(substitution)
                self._solution = substitution
                for objective_function in objective_functions:
                    objective_function.add_solution(substitution)
                if limit is not None and state.solutions_count >= limit:
                    timer.abort()
                    state.state = State.INTERRUPT_LIMIT
                    return
                timer.start()
                continue
        state.state = State.DONE
        timer.abort()


def _propagate(unbound_var_names, var_name, substitution, model_info):
    var_ad = model_info.var_ad
    var_constraints = model_info.var_constraints
    domains = model_info.domains
    reduced_domains = model_info.reduced_domains
    bound_var_set = set(substitution)
    indent = '  ' * len(substitution)
    if var_name is None:
        v_names = unbound_var_names
    else:
        v_map = model_info.var_map[var_name]
        v_names = []
        for v_name in unbound_var_names:
            if v_map[v_name]:
                v_names.append(v_name)
    for unbound_var_name in v_names:
        subst = substitution.copy()
        b_set = bound_var_set | {unbound_var_name}
        enabled_c_functions = []
        # apply all_different constraints:
        forbidden_values = {substitution[v_name] for v_name in var_ad[unbound_var_name].intersection(bound_var_set)}
        reduced_domain = [v for v in domains[unbound_var_name] if v not in forbidden_values]
        for constraint in var_constraints[unbound_var_name]:
            c_function = constraint.function
            if constraint.vars <= b_set:
                new_reduced_domain = []
                for unbound_var_value in reduced_domain:
                    subst[unbound_var_name] = unbound_var_value
                    if c_function(subst):
                        new_reduced_domain.append(unbound_var_value)
                reduced_domain = new_reduced_domain
                if not reduced_domain:
                    break

        reduced_domains[unbound_var_name] = reduced_domain
        if not reduced_domain:
            return False
    return True
