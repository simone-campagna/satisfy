import copy
import collections
import re
import types

from .constraint import (
    Constraint,
    ExpressionConstraint,
    AllDifferentConstraint,
)
from .expression import (
    Expression,
    Variable,
    Parameter,
)

__all__ = [
    'Model',
]


VariableInfo = collections.namedtuple("VariableInfo", "variable domain")


class Model(object):
    __re_name__ = re.compile(r'^[a-zA-Z]\w*$')

    def __init__(self):
        self._variables = collections.OrderedDict()
        self._constraints = []
        self._variables_proxy = types.MappingProxyType(self._variables)

    def variables(self):
        return self._variables_proxy

    def constraints(self):
        yield from self._constraints

    def _check_name(self, name):
        if not self.__re_name__.match(name):
            raise ValueError("bad name {!r}".format(name))

    def _get_variable(self, name):
        if name is None:
            name = "_v{}".format(len(self._variables))
        else:
            self._check_name(name)
            assert ":" not in name
            if name in self._variables:
                raise ValueError("variable {} already defined".format(name))
        return Variable(name)

    def _get_parameter(self, name, value):
        if name is None:
            name = "_p{}".format(len(self._variables))
        else:
            self._check_name(name)
            if name in self._variables:
                raise ValueError("variable {} already defined".format(name))
        return Parameter(name, value)

    def add_parameter(self, value, name=None):
        parameter = self._get_parameter(name, value)
        self._variables[parameter.name] = VariableInfo(
            variable=parameter,
            domain=None)
        return parameter

    def add_int_variable(self, domain, *, name=None):
        values = set()
        for value in domain:
            if value in values:
                raise ValueError("duplicated value {}".format(value))
            values.add(value)
        variable = self._get_variable(name)
        self._variables[variable.name] = VariableInfo(
            variable=variable,
            domain=domain)
        return variable

    def add_bool_variable(self, *, name=None):
        return self.add_int_variable(domain=(0, 1), name=name)

    def get_variable(self, name):
        return self._variables[name].variable

    def add_constraint(self, constraint):
        if isinstance(constraint, Expression):
            constraint = ExpressionConstraint(constraint)
        elif not isinstance(constraint, Constraint):
            raise TypeError("{} is not a valid constraint".format(constraint))
        var_names = tuple(constraint.vars())
        if not var_names:
            raise ValueError("constraint {} does not depends on model variables".format(constraint))
        for var_name in var_names:
            if var_name not in self._variables:
                raise ValueError("constraint {} depends on undefined variable {}".format(constraint, var_name))
        self._constraints.append(constraint)

    def add_all_different_constraint(self, variables):
        self.add_constraint(AllDifferentConstraint([var.name for var in variables]))
