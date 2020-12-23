import abc
import collections
import contextlib
import enum
import itertools
import logging
import random

from .constraint import AllDifferentConstraint
from .expression import expression_globals
from .objective import Objective
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
        'extra',
    ]
)


OptimizationResult = collections.namedtuple(  # pylint: disable=invalid-name
    "OptimizationResult",
    "is_optimal state count solution")


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

    def __call__(self, bound_var_names, unbound_var_names, model_info):
        var_name = self.select_var(bound_var_names, unbound_var_names, model_info)
        bound_var_names.append(var_name)
        enabled_constraints = self.select_enabled_constraints(bound_var_names, unbound_var_names, model_info)
        return var_name, bound_var_names, unbound_var_names, enabled_constraints

    @abc.abstractmethod
    def sort_var_names(self, unbound_var_names, model_info):
        raise NotImplementedError()

    @abc.abstractmethod
    def select_var(self, bound_var_names, unbound_var_names, model_info):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def select_enabled_constraints(self, bound_var_names, unbound_var_names, model_info):
        raise NotImplementedError()
    
    def __repr__(self):
        return 'SelectVar.{}'.format(self.name)

    
class StaticVarSelector(VarSelector):
    def init(self, unbound_var_names, model_info):
        self.sort_var_names(unbound_var_names, model_info)
        self.constraints = [[]]
        var_constraints = model_info.var_constraints
        s_bound_var_names = set()
        for var_name in reversed(unbound_var_names):
            s_bound_var_names.add(var_name)
            var_clist = []
            for constraint in var_constraints[var_name]:
                if constraint.vars() <= s_bound_var_names:
                    var_clist.append(constraint)
            self.constraints.append(var_clist)

    def select_enabled_constraints(self, bound_var_names, unbound_var_names, model_info):
        return self.constraints[len(bound_var_names)]

    def select_var(self, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


class DynamicVarSelector(VarSelector):
    def init(self, unbound_var_names, model_info):
        pass

    def select_enabled_constraints(self, bound_var_names, unbound_var_names, model_info):
        var_name = bound_var_names[-1]
        var_constraints = model_info.var_constraints[var_name]
        enabled_constraints = []
        var_set = set(bound_var_names)
        for constraint in var_constraints:
            if constraint.vars() <= var_set:
                enabled_constraints.append(constraint)
        return enabled_constraints


@SelectVar.__register__('in_order')
class InOrderVarSelector(StaticVarSelector):
    def sort_var_names(self, unbound_var_names, model_info):
        unbound_var_names.reverse()


@SelectVar.__register__('random')
class RandomVarSelector(StaticVarSelector):
    def sort_var_names(self, unbound_var_names, model_info):
        random.shuffle(unbound_var_names)


def iterator_len(it):
    count = 0
    for dummy in it:
        count += 1
    return count


class BoundVarSelector(StaticVarSelector):
    def sort_var_names(self, unbound_var_names, model_info):
        var_map = model_info.var_map
        key_function = self.__class__.__key_function__
        if self.__reverse__:
            unbound_var_names.reverse()  # stable sort!
        unbound_var_names.sort(key=lambda v: key_function(var_map[v].values()), reverse=self.__reverse__)
        # print(":::", self)
        # for var_name in unbound_var_names:
        #     v = var_map[var_name].values()
        #     print("  :", var_name, key_function, key_function(v), v)
        # input("...")


@SelectVar.__register__('min_boundmin')
class Min_BoundMinSelector(BoundVarSelector):
    __key_function__ = min
    __reverse__ = True


@SelectVar.__register__('min_boundmax', 'min_bound')
class Min_BoundMaxSelector(BoundVarSelector):
    __key_function__ = max
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
    __key_function__ = min
    __reverse__ = False


@SelectVar.__register__('max_boundmax')
class Max_BoundMaxSelector(BoundVarSelector):
    __key_function__ = max
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
    def sort_var_names(self, unbound_var_names, model_info, reverse):
        var_domains = model_info.original_domains
        if reverse:
            unbound_var_names.reverse()  # stable sort
        unbound_var_names.sort(key=lambda v: len(var_domains[v]), reverse=reverse)


@SelectVar.__register__('min_domain')
class MinDomainVarSelector(DomainVarSelector):
    def select_var(self, bound_var_names, unbound_var_names, model_info):
        self._sort(unbound_var_names, model_info, reverse=True)
        return unbound_var_names.pop(-1)


@SelectVar.__register__('max_domain')
class MaxDomainVarSelector(DomainVarSelector):
    def select_var(self, bound_var_names, unbound_var_names, model_info):
        self._sort(unbound_var_names, model_info, reverse=False)
        return unbound_var_names.pop(-1)


@SelectVar.__register__('group_prio')
class GroupPrioVarSelector(StaticVarSelector):
    def sort_var_names(self, unbound_var_names, model_info):
        var_group_prio = model_info.var_group_prio
        var_bounds = model_info.var_bounds
        var_domains = model_info.initial_domains
        unbound_var_names.sort(key=lambda v: (var_group_prio[v], var_bounds[v], len(var_domains[v])), reverse=True)

    def select_var(self, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


@SelectVar.__register__('min_alphanumeric')
class MinAlphanumericVarSelector(StaticVarSelector):
    def sort_var_names(self, unbound_var_names, model_info):
        unbound_var_names.sort(reverse=True)

    def select_var(self, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


@SelectVar.__register__('max_alphanumeric')
class MaxAlphanumericVarSelector(StaticVarSelector):
    def sort_var_names(self, unbound_var_names, model_info):
        unbound_var_names.sort(reverse=False)

    def select_var(self, bound_var_names, unbound_var_names, model_info):
        return unbound_var_names.pop(-1)


class Solver:
    def __init__(self,
                 select_var=SelectVar.max_boundmin,
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

    @contextlib.contextmanager
    def __call__(self, model):
        yield ModelSolver(model, self)


class State(enum.Enum):
    RUNNING = 0
    DONE = 1
    INTERRUPT_TIMEOUT = 2
    INTERRUPT_LIMIT = 3


class SolverState:
    def __init__(self):
        self._timer = Timer()
        self._count = 0
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
    def count(self):
        return self._count

    @state.setter
    def state(self, value):
        if not isinstance(value, State):
            raise TypeError(value)
        self._state = value

    def add_solution(self, solution):
        self._current_solution = solution
        self._count += 1


class ModelSolver:
    def __init__(self, model, solver):
        self._model = model
        self._solver = solver
        compile_constraints = solver.compile_constraints
        reduce_max_depth = solver.reduce_max_depth
        self._state = SolverState()
        select_var = solver.select_var
        select_value = solver.select_value

        # 0. objectives:
        objective_functions = []
        additional_constraints = []
        for objective_function in model.objective_functions():
            objective_constraints = list(objective_function.constraints)
            additional_constraints.extend(objective_constraints)
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
        m_globals = model.globals
        for constraint in itertools.chain(model.constraints(), additional_constraints):
            constraint.globals = m_globals
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

        group_prio = {groupid: groupid for groupid in groups}  #idx for idx, groupid in enumerate(sorted(groups, key=lambda x: (group_bounds[x], group_sizes[x]), reverse=True))}
        var_group_prio = {}
        for var_name in var_names:
            if var_groups[var_name]:
                var_group_prio[var_name] = min(group_prio[groupid] for groupid in var_groups[var_name])
            else:
                var_group_prio[var_name] = -1

        reduced_domains = {}
        domains = collections.ChainMap(reduced_domains, initial_domains)
        self._model_info = ModelInfo(
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
            extra={}
        )

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
                count=self._state.count,
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

        if not (model_info.solvable and model_info.var_names):
            state.state = State.DONE
            return

        var_names = model_info.var_names
        reduced_domains = model_info.reduced_domains
        initial_domains = model_info.initial_domains
        var_ad = model_info.var_ad
        var_constraints = model_info.var_constraints
        objective_functions = model_info.objective_functions

        # 2. solve:
        timer.start()
        stack = []
        if var_names:
            unbound_var_names = list(var_names)
            select_var.init(unbound_var_names, model_info)
            var_name, bound_var_names, unbound_var_names, enabled_constraints = select_var([], unbound_var_names, model_info)
            stack.append((var_name, bound_var_names, unbound_var_names, enabled_constraints, {}))
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
            # var_name = bound_var_names[-1]
            substitution = substitution.copy()
            reduced_domain = reduced_domains.get(var_name, None)
            if reduced_domain is None:
                # bound_var_set = set(bound_var_names)
                forbidden_values = {value for vname, value in substitution.items() if vname in var_ad[var_name]}

                reduced_domain = [value for value in initial_domains[var_name] if value not in forbidden_values]
                for constraint in enabled_constraints:
                    c_fun = constraint.evaluate
                    nr_dom = []
                    for value in reduced_domain:
                        substitution[var_name] = value
                        try:
                            if c_fun(substitution):
                                nr_dom.append(value)
                        except:
                            LOG.error("constraint: %s", constraint)
                            LOG.error("substitution: %s", substitution)
                            LOG.exception("constraint evaluation failed:")
                            raise
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
                next_var_name, next_bound_var_names, next_unbound_var_names, next_enabled_constraints = select_var(list(bound_var_names), list(unbound_var_names), model_info)
                stack.append((next_var_name, next_bound_var_names, next_unbound_var_names, next_enabled_constraints, substitution))
            else:
                timer.stop()
                state.add_solution(substitution)
                for constraint in model.constraints():
                    if constraint.unsatisfied(substitution):
                        raise RuntimeError("constraint {} is not satisfied".format(constraint))
                yield solution_transformer(substitution)
                self._solution = substitution
                for objective_function in objective_functions:
                    objective_function.add_solution(substitution)
                if limit is not None and state.count >= limit:
                    timer.abort()
                    state.state = State.INTERRUPT_LIMIT
                    return
                timer.start()
                continue
        state.state = State.DONE
        timer.abort()
