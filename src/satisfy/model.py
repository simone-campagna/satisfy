import copy
import collections
import keyword
import logging
import re
import types

from .constraint import (
    Constraint,
    ExpressionConstraint,
    ConstConstraint,
    AllDifferentConstraint,
)
from .expression import (
    expression_globals,
    Expression,
    Variable,
    Parameter,
)
from .objective import Objective
from .solver import SelectVar, SelectValue, Solver


__all__ = [
    'Model',
]


LOG = logging.getLogger(__name__)

class VariableInfo:
    def __init__(self, variable, domain):
        self.variable = variable
        self.domain = domain

    def __repr__(self):
        return "{}(variable={!r}, domain={!r})".format(
            type(self).__name__,
            self.variable,
            self.domain)


class Model(object):
    __reserved__ = set(keyword.kwlist)
    __re_name__ = re.compile(r'^[a-zA-Z]\w*$')

    def __init__(self):
        self.__variables = collections.OrderedDict()
        self.__constraints = []
        self.__variables_proxy = types.MappingProxyType(self.__variables)
        self.__solvable = True
        self.__objectives = []
        self.__globals = {}
        self.__globals.update(expression_globals())
        self.add_function(min)
        self.add_function(max)

    def add_global_symbol(self, name, obj):
        self.__globals[name] = obj

    def add_function(self, function, name=None):
        if name is None:
            name = function.__name__
        self.add_global_symbol(name, function)

    @property
    def globals(self):
        return self.__globals

    def has_objectives(self):
        return bool(self.__objectives)

    def objectives(self):
        yield from self.__objectives

    def objective_functions(self):
        for objective in self.__objectives:
            yield objective.build(self)

    def solvable(self):
        return self.__solvable

    def variables(self):
        return self.__variables_proxy

    def constraints(self):
        yield from self.__constraints

    def _check_name(self, name):
        if not self.__re_name__.match(name):
            raise ValueError("bad name {!r}".format(name))
        if name in self.__reserved__:
            raise ValueError("bad name {!r}: this is a reserved keyword".format(name))
        if name in expression_globals():
            raise ValueError("bad name {!r}: this is a reserved symbol name".format(name))

    def _get_variable(self, name):
        if name is None:
            name = "_v{}".format(len(self.__variables))
        else:
            self._check_name(name)
            assert ":" not in name
            if name in self.__variables:
                raise ValueError("variable {} already defined".format(name))
        return Variable(name)

    def _get_parameter(self, name, value):
        if name is None:
            name = "_p{}".format(len(self.__variables))
        else:
            self._check_name(name)
            if name in self.__variables:
                raise ValueError("variable {} already defined".format(name))
        return Parameter(name, value)

    def add_parameter(self, value, name=None):
        parameter = self._get_parameter(name, value)
        self.__variables[parameter.name] = VariableInfo(
            variable=parameter,
            domain=None)
        return parameter

    def _check_domain(self, domain):
        values = set()
        for value in domain:
            if value in values:
                raise ValueError("duplicated value {}".format(value))
            values.add(value)

    def add_int_variable(self, domain, *, name=None):
        self._check_domain(domain)
        variable = self._get_variable(name)
        self.__variables[variable.name] = VariableInfo(
            variable=variable,
            domain=domain)
        return variable

    def set_variable_domain(self, variable, domain):
        self._check_domain(domain)
        if isinstance(variable, Variable):
            var_name = variable.name
        else:
            var_name = variable
        self.__variables[var_name].domain = domain

    def add_bool_variable(self, *, name=None):
        return self.add_int_variable(domain=(0, 1), name=name)

    def get_variable(self, name):
        return self.__variables[name].variable

    def set_solvable(self, solvable):
        self.__unsolvable = True

    def add_constraint(self, constraint):
        if isinstance(constraint, bool):
            constraint = ConstConstraint(constraint)
        elif isinstance(constraint, Expression):
            constraint = ExpressionConstraint(constraint)
        elif not isinstance(constraint, Constraint):
            raise TypeError("{} is not a valid constraint".format(constraint))
        var_names = tuple(constraint.vars())
        if not var_names:
            value = constraint.evaluate({})
            if value:
                LOG.warning("constraint {} is always satisfied".format(constraint))
                return
            else:
                LOG.warning("constraint {} is never satisfied - model is marked as not solvable".format(constraint))
                self.__solvable = False
        for var_name in var_names:
            if var_name not in self.__variables:
                raise ValueError("constraint {} depends on undefined variable {}".format(constraint, var_name))
        self.__constraints.append(constraint)

    def add_all_different_constraint(self, variables):
        self.add_constraint(AllDifferentConstraint([var.name for var in variables]))

    def add_objective(self, objective):
        if not isinstance(objective, Objective):
            raise TypeError(objective)
        self.__objectives.append(objective)

    def get_var_domain(self, var):
        if isinstance(var, Variable):
            var_name = var.name
        else:
            var_name = var
        return self.__variables[var_name].domain

    def solver(self, **kwargs):
        return Solver(**kwargs)

    def solve(self, **kwargs):
        return self.solver(**kwargs)(self)

    def transform_solution(self, solution):
        return solution
