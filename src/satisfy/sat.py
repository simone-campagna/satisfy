#!/usr/bin/env python

import ast
import collections
import functools
import itertools
import logging
import operator
import re
import sys
import types

import ply.lex as lex
import ply.yacc as yacc
 
from .constraint import AllDifferentConstraint
from .expression import (
    Const, InputConst, InputReader, Variable,
    FunctionCall, GlobalVariable,
)
from .model import Model
from .solver import Solver, SelectVar, SelectValue
from .objective import Maximize, Minimize
from .utils import safe_call
from . import expression as _expr


__all__ = [
    'sat_compile',
    'Sat',
    'SatParser',
    'SatLexer',
]

LOG = logging.getLogger()


class SatSyntaxError(RuntimeError):
    pass


def flatten(*terms):
    lst = []
    for term in terms:
        if isinstance(term, list):
            lst.extend(term)
        else:
            lst.append(term)
    return lst


BINOP = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.floordiv,
    '**': operator.pow,
    '&': operator.and_,
    '|': operator.or_,
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne,
}


OBJECTIVE = {
    'maximize': Maximize,
    'minimize': Minimize,
}


def make_value(sat, v):
    if isinstance(v, str):
        v = sat.get_symbol(v)
    return v


def make_const_value(sat, v):
    if isinstance(v, str):
        v = sat.get_symbol(v)
        if not isinstance(v, Const):
            raise SatSyntaxError("invalid non-const expression {}".format(v))
    return v


def make_binop(sat, l, op, r):
    return BINOP[op](make_value(sat, l), make_value(sat, r))


class SatProxy:
    def __init__(self, sat):
        self.__sat = sat

    @property
    def variables(self):
        return {
            var_name: list(domain) for var_name, domain in self.__sat.variables().items()
        }

    @property
    def macros(self):
        return {
            macro: str(expression) for macro, expression in self.__sat.macros().items()
        }

    @property
    def constraints(self):
        return [
            str(constraint) for constraint in self.__sat.constraints()
        ]

    @property
    def objectives(self):
        return [
            str(objective) for objective in self.__sat.objectives()
        ]


class SatTerm:
    def values(self, sat):
        raise NotImplementedError()

    @classmethod
    def build(cls, value):
        if isinstance(value, cls):
            return value
        return SatValue(value)


class SatValue(SatTerm):
    def __init__(self, value):
        assert not isinstance(value, SatTerm)
        self.value =  value

    def values(self, sat):
        yield sat.get_value(self.value)


class SatRange(SatTerm):
    def __init__(self, start, stop, stride=1):
        self.start =  start
        self.stop =  stop
        self.stride =  stride

    def values(self, sat):
        start = sat.get_value(self.start)
        stop = sat.get_value(self.stop)
        stride = sat.get_value(self.stride)
        if stride < 0:
            increment = -1
        else:
            increment = +1
        yield from range(start, stop + increment, stride)


class SatIO:
    pass


class SatInput(SatIO):
    def __init__(self, input_const, prompt=None):
        self.input_const = input_const
        self.prompt = prompt

    def __call__(self, input_file, output_file, *args, **kwargs):
        reader = InputReader(input_file=input_file, output_file=output_file, prompt=self.prompt)
        self.input_const.reader = reader
        return self.input_const.value

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.input_const)


class SatOutput(SatIO):
    def __init__(self, text):
        self.text = text

    def __call__(self, input_file, output_file, *args, **kwargs):
        print(self.text.format(*args, **kwargs), file=output_file, flush=True)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.text)


class SatPySource:
    def __init__(self, source, variables=None):
        self.source = source
        self.variables = variables
        self.ast = ast.parse(self.source)

    def make_callback(self, registry, scope):
        def callback(function):
            registry[scope].append(function)
            return function
        return callback

    def compile(self, globals_dict, callback_registry, scope_mapping):
        gdict = {}
        for scope, name in scope_mapping.items():
            gdict[name] = self.make_callback(callback_registry, scope)
        result = exec(self.source, gdict)
        for name in scope_mapping.values():
            gdict.pop(name, None)
        if self.variables is None:
            globals_dict.update(gdict)
        else:
            globals_dict.update({key: gdict[key] for key in self.variables})
        return result


class SatModel(Model):
    def __init__(self, *, options=None, **kwargs):
        if options is None:
            options = {}
        self.options = options
        super().__init__(**kwargs)

    def solve(self, **kwargs):
        for key, value in self.options.items():
            if key not in kwargs:
                kwargs[key] = value
        return super().solve(**kwargs)


class Sat:
    SCOPE_BEGIN = '[begin]'
    SCOPE_SOLUTION = '[solution]'
    SCOPE_OPTIMAL_SOLUTION = '[optimal-solution]'
    SCOPE_END = '[end]'
    SCOPES = (
        SCOPE_BEGIN,
        SCOPE_SOLUTION,
        SCOPE_OPTIMAL_SOLUTION,
        SCOPE_END,
    )
    def __init__(self, **kwargs):
        self.model_kwargs = kwargs
        self.__globals_dict = {}
        self.__global_vars = {}
        self.__py_sources = []
        self.__domains = {}
        self.__constraints = []
        self.__objectives = []
        self.__vars = {}
        self.__var_domains = {}
        self.__macro_names = []
        self.__macros = {}
        self.__solver_options = {}
        self.__io = {
            self.SCOPE_BEGIN: [],
            self.SCOPE_SOLUTION: [],
            self.SCOPE_OPTIMAL_SOLUTION: [],
            self.SCOPE_END: [],
        }
        self.__io_callbacks = {
            self.SCOPE_BEGIN: [],
            self.SCOPE_SOLUTION: [],
            self.SCOPE_OPTIMAL_SOLUTION: [],
            self.SCOPE_END: [],
        }
        self.__ucount = collections.Counter()

    def _get_uid(self, namespace):
        idx = self.__ucount[namespace]
        self.__ucount[namespace] += 1
        return '__{}::{}'.format(namespace, idx)

    def define_sat_input(self, prompt=None, value_type=int):
        input_const = InputConst()
        in_obj = SatInput(input_const=input_const, prompt=prompt)
        self.__io[self.SCOPE_BEGIN].append(in_obj)
        return input_const

    def define_sat_output(self, p, scope, text):
        out_obj = SatOutput(text=text)
        self.__io[scope].append(out_obj)
        return out_obj

    def define_sat_domain(self, p, name, domain):
        self.__domains[name] = domain
        return domain

    def define_sat_vars(self, p, domain, *names):
        if isinstance(domain, str):
            domain_name = domain
        else:
            domain_name = self._get_uid('domain')
            self.__domains[domain_name] = domain
            domain = []
        variables = []
        for name in names:
            self.__var_domains[name] = domain_name
            var = Variable(name)
            self.__vars[name] = var
            variables.append(var)
        return variables

    def define_sat_macro(self, p, name, expression):
        if isinstance(expression, int):
            expression = Const(expression)
        self.__vars[name] = expression
        self.__macros[name] = expression
        self.__macro_names.append(name)

    def define_sat_function_call(self, p, function_name, parameters, named_parameters):
        return FunctionCall(function_name, parameters, named_parameters)

    def define_sat_globals(self, p, source, variables=None):
        py_source = SatPySource(source=source, variables=variables)
        self.__py_sources.append(py_source)
        return py_source

###############################################################################

    def build_model(self, input_file=None, output_file=None):
        if input_file is None:
            input_file = sys.stdin
        if output_file is None:
            output_file = sys.stdout
        self.io_init(input_file, output_file)

        globals_dict = {}
        scope_mapping = {
            self.SCOPE_BEGIN: 'begin',
            self.SCOPE_SOLUTION: 'solution',
            self.SCOPE_OPTIMAL_SOLUTION: 'optimal_solution',
            self.SCOPE_END: 'end',
        }
        for py_source in self.__py_sources:
            py_source.compile(globals_dict, self.__io_callbacks, scope_mapping)

        macros = self.expand_macros()
        domains = {}
        for domain_name, domain_terms in self.__domains.items():
            domain = []
            for term in domain_terms:
                domain.extend(term.values(self))
            domains[domain_name] = domain

        model = SatModel(**self.model_kwargs, options=self.__solver_options)
        for key, value in globals_dict.items():
            model.add_global_symbol(key, value)

        for var_name in self.__vars:
            if var_name in self.__var_domains:
                domain = self.__var_domains[var_name]
                if isinstance(domain, str):
                    domain = domains[domain]
                model.add_int_variable(domain, name=var_name)

        for constraint in self.__constraints:
            model.add_constraint(constraint)

        for objective in self.__objectives:
            model.add_objective(objective)

        return model

###############################################################################
    def get_value(self, value):
        if isinstance(value, str):
            macros = self.expand_macros()
            if value in macros:
                return macros[value]
            else:
                raise KeyError(value)
        elif isinstance(value, Const):
            value = value.value
        return value

    def domains(self):
        return types.MappingProxyType(self.__domains)

    def vars(self):
        return types.MappingProxyType(self.__vars)

    def macros(self):
        return types.MappingProxyType(self.__macros)

    def get_symbol(self, symbol):
        if symbol in self.__vars:
            return self.__vars[symbol]
        else:
            if symbol not in self.__global_vars:
                self.__global_vars[symbol] = GlobalVariable(symbol)
            return self.__global_vars[symbol]

    def expand_macros(self, substitution=None, force_input_const_eval=True):
        data = {}
        if substitution:
            data.update(substitution)
        macros = {}
        for macro_name in self.__macro_names:
            expression = self.__macros[macro_name]
            if (not force_input_const_eval) and isinstance(expression, InputConst) and not expression.has_value():
                continue
            if expression.is_free(data):
                value = expression.evaluate(data)
                macros[macro_name] = value
                data[macro_name] = value
        return macros

    def _get_data(self, *, model_solver=None, solution=None):
        data = {
            '_MODEL': SatProxy(self),
        }
        if model_solver is not None:
            data['_STATE'] = model_solver.state.state.name
            data['_COUNT'] =  model_solver.stats.count
            data['_ELAPSED'] = model_solver.stats.elapsed
        if solution is not None:
            data['_SOLUTION'] = solution
            if '_COUNT' in data:
                data['_INDEX'] = data['_COUNT'] - 1
            data.update(self.expand_macros(solution))
            for var_name, value in solution.items():
                data[var_name] = value
        return data

    def _do_io(self, scope, input_file, output_file, data):
        for io_obj in self.__io[scope]:
            io_obj(input_file, output_file, **data)

    def io_begin_callbacks(self, input_file, output_file, model_solver):
        for io_callback in self.__io_callbacks[self.SCOPE_BEGIN]:
            safe_call(io_callback,
                      input_file=input_file, output_file=output_file,
                      model_solver=model_solver,
                      state=model_solver.state.state,
                      stats=model_solver.stats,
            )

    def io_solution_callbacks(self, input_file, output_file, model_solver, solution):
        for io_callback in self.__io_callbacks[self.SCOPE_SOLUTION]:
            safe_call(io_callback,
                      input_file=input_file, output_file=output_file,
                      solution=solution,
                      model_solver=model_solver,
                      state=model_solver.state.state,
                      stats=model_solver.stats,
            )

    def io_optimal_solution_callbacks(self, input_file, output_file, model_solver):
        for io_callback in self.__io_callbacks[self.SCOPE_OPTIMAL_SOLUTION]:
            safe_call(io_callback,
                      input_file=input_file, output_file=output_file,
                      model_solver=model_solver,
                      optimization_result=model_solver.get_optimization_result(),
                      state=model_solver.state.state,
                      stats=model_solver.stats,
            )

    def io_end_callbacks(self, input_file, output_file, model_solver):
        for io_callback in self.__io_callbacks[self.SCOPE_END]:
            safe_call(io_callback,
                      input_file=input_file, output_file=output_file,
                      model_solver=model_solver,
                      state=model_solver.state.state,
                      stats=model_solver.stats,
            )

    def io_begin(self, input_file, output_file, model_solver):
        self.io_begin_callbacks(input_file, output_file, model_solver)

    def io_init(self, input_file, output_file):
        macros = self.expand_macros(force_input_const_eval=False)
        for io_obj in self.__io[self.SCOPE_BEGIN]:
            io_obj(input_file, output_file, **macros)
            if isinstance(io_obj, SatInput):
                macros = self.expand_macros(force_input_const_eval=False)

    def io_solution(self, input_file, output_file, model_solver, solution):
        data = self._get_data(model_solver=model_solver, solution=solution)
        data['_SOLUTION'] = solution
        data['_INDEX'] = data['_COUNT'] - 1
        self._do_io(self.SCOPE_SOLUTION, input_file, output_file, data)
        self.io_solution_callbacks(input_file, output_file, model_solver, solution)

    def io_optimal_solution(self, input_file, output_file, model_solver):
        model = model_solver.model
        if not model.has_objectives():
            return
        optimization_result = model_solver.get_optimization_result()
        if optimization_result.solution is not None:
            stats = model_solver.stats
            solution = optimization_result.solution
            if optimization_result.is_optimal:
                optimal = 'optimal'
            else:
                optimal = 'sub-optimal'
            data = self._get_data(model_solver=model_solver, solution=solution)
            data['_IS_OPTIMAL'] = optimization_result.is_optimal
            data['_OPTIMAL'] = optimal
            self._do_io(self.SCOPE_OPTIMAL_SOLUTION, input_file, output_file, data)
        self.io_optimal_solution_callbacks(input_file, output_file, model_solver)

    def io_end(self, input_file, output_file, model_solver):
        data = self._get_data(model_solver=model_solver)
        self._do_io(self.SCOPE_END, input_file, output_file, data)
        self.io_end_callbacks(input_file, output_file, model_solver)

    def add_sat_constraint(self, p, constraint):
        self.__constraints.append(constraint)
        return constraint

    def add_sat_all_different_constraint(self, p, constraint_type, var_list):
        self.__constraints.append(AllDifferentConstraint([var_name for var_name in var_list]))
        return var_list

    def add_sat_objective(self, p, objective_name, expression):
        self.__objectives.append(OBJECTIVE[objective_name](expression))

    def set_sat_option(self, p, option_name, option_value):
        if option_name == 'select_var':
            if not hasattr(SelectVar, option_value):
                raise SatSyntaxError("illegal SelectVar {!r}".format(option_value))
            self.__solver_options[option_name] = getattr(SelectVar, option_value)
        elif option_name == 'select_value':
            if not hasattr(SelectValue, option_value):
                raise SatSyntaxError("illegal SelectValue {!r}".format(option_value))
            self.__solver_options[option_name] = getattr(SelectValue, option_value)
        elif option_name in ['timeout']:
            if not isinstance(option_value, (int, float)):
                raise SatSyntaxError("illegal {} = {!r} of type {}".format(option_name, option_value, type(option_value).__name__))
            self.__solver_options[option_name] = option_value
        elif option_name in ['limit', 'reduce_max_depth']:
            if not isinstance(option_value, int):
                raise SatSyntaxError("illegal {} = {!r} of type {}".format(option_name, option_value, type(option_value).__name__))
            self.__solver_options[option_name] = option_value
        elif option_name in ['compile_constraints', 'discard_const_vars']:
            if not isinstance(option_value, (int, bool)):
                raise SatSyntaxError("illegal option {} = {!r} of type {}".format(option_name, option_value, type(option_value).__name__))
            self.__solver_options[option_name] = bool(option_value)
        else:
            raise SatSyntaxError("unknown option {!r}".format(option_name))


class SatLexer:
    # List of token names.   This is always required
    tokens = (
       'INTEGER',
       'FLOATING_POINT',
       'PLUS',
       'MINUS',
       'TIMES',
       'DIVIDE',
       'POW',
       'LT',
       'LE',
       'GT',
       'GE',
       'EQ',
       'NE',
       'AND',
       'OR',
       'NOT',
       'LPAREN',
       'RPAREN',
       'L_SQUARE_BRACKET',
       'R_SQUARE_BRACKET',
       'COMMA',
       'COLON',
       'EQUALS',
       'DEF_VAR',
       'DEF_MACRO',
       'DEF_OPTION',
       'NEWLINE',
       'OBJECTIVE',
       'ALL_DIFFERENT_CONSTRAINT',
       'SYMBOL',
       'MULTILINE_STRING',
       'STRING',
       'OUTPUT',
       'INPUT',
       'DEFINE',
    )
    
    # Regular expression rules for simple tokens
    t_PLUS                     = r'\+'
    t_MINUS                    = r'-'
    t_TIMES                    = r'\*'
    t_DIVIDE                   = r'/'
    t_POW                      = r'\*\*'
    t_GT                       = r'\>'
    t_GE                       = r'\>\='
    t_LT                       = r'\<'
    t_LE                       = r'\<\='
    t_EQ                       = r'\=\='
    t_NE                       = r'\!\='
    t_AND                      = r'\&'
    t_OR                       = r'\|'
    t_NOT                      = r'\!'
    t_LPAREN                   = r'\('
    t_RPAREN                   = r'\)'
    t_COLON                    = r'\:'
    t_COMMA                    = r'\,'
    t_SYMBOL                   = r'[a-zA-Z]\w*'
    t_L_SQUARE_BRACKET         = r'\['
    t_R_SQUARE_BRACKET         = r'\]'
    t_EQUALS                   = r'\='
    t_DEF_VAR                  = r'\:\:'
    t_DEF_MACRO                = r'\:\='
    t_DEFINE                  = r'\[define'

    def t_DEF_OPTION(self, t):
        r'option'
        return t

    def t_OBJECTIVE(self, t):
        r'minimize|maximize'
        return t

    def t_ALL_DIFFERENT_CONSTRAINT(self, t):
        r'all_different'
        return t

    def t_INTEGER(self, t):
        r'\d+(?!\.)'
        t.value = int(t.value)    
        return t
    
    def t_FLOATING_POINT(self, t):
        r'\d*\.\d+|\d+\.\d*'
        t.value = float(t.value)    
        return t
    
    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
        return t
    
    def t_COMMENT(self, t):
        r'\#.*\n*'
        v = t.value.rstrip('\n')
        t.lexer.lineno += len(t.value) - len(v)
        pass

    @lex.TOKEN(r'|'.join(re.escape(x) for x in Sat.SCOPES))
    def t_OUTPUT(self, t):
        return t

    def t_INPUT(self, t):
        r'\[input\]'
        return t

    def t_MULTILINE_STRING(self, t):
        r'\[\[\[\s*\n(.|\n)*?\]\]\]'
        t.lexer.lineno += t.value.count('\n')
        value = t.value[3:-3]
        value.strip(' ')
        if value[0] == '\n':
            value = value[1:]
        if value[-1] == '\n':
            value = value[:-1]
        t.value = value
        return t

    def t_STRING(self, t):
        r'\".*\"|\'([^\']|(?<=\\)\')*\''
        t.value = t.value[1:-1]
        return t

    t_ignore  = ' \t'
    
    # Error handling rule
    def t_error(self, t):
        raise SatSyntaxError("illegal character {!r} at line {}".format(
            t.value[0], t.lineno))
        # t.lexer.skip(1)

    def parse(self, source):
        self.lexer.input(source)
        while True:
            tok = self.lexer.token()
            if not tok: 
                break
            yield tok

    def __init__(self):
        self.lexer = lex.lex(module=self)


class SatParser:
    tokens = SatLexer.tokens
    start = 'code'

    def __init__(self, sat=None):
        self.lexer = SatLexer()
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        if sat is None:
            sat = Sat()
        self.sat = sat

    def parse(self, source):
        result = self.parser.parse(source, lexer=self.lexer.lexer)
        return result

    def p_empty(self, p):
        'empty :'
        pass

    # def p_code_single_line(self, p):
    #     'code : code_line'
    #     p[0] = p[1]

    def p_code_multiple_lines(self, p):
        '''code : code_line NEWLINE code
                | code_line NEWLINE empty
        '''
        if p[3]:
            p[0] = p[1] +  p[3]
        else:
            p[0] = p[1]

    def p_code_line(self, p):
        '''code_line :
                     | option_definition
                     | domain_definition
                     | var_definition
                     | macro_definition
                     | constraint_definition
                     | objective_definition
                     | output_definition
                     | globals_definition
        '''
        p[0] = [p[1]]

    ### OUTPUT DEFINITION
    def p_output_definition(self, p):
        '''output_definition : OUTPUT STRING
                             | OUTPUT MULTILINE_STRING
        '''
        scope = p[1]
        output_line = p[2]
        p[0] = self.sat.define_sat_output(p, scope, output_line)
        
    ### INPUT DEFINITION
    def p_input_definition(self, p):
        '''input_definition : INPUT STRING
                            | INPUT MULTILINE_STRING
        '''
        prompt = p[2]
        p[0] = self.sat.define_sat_input(prompt, value_type=int)

    ### GLOBALS DEFINITION
    def p_globals_vars_all(self, p):
        '''globals_vars : DEFINE R_SQUARE_BRACKET'''
        p[0] = None
        
    def p_globals_vars_none(self, p):
        '''globals_vars : DEFINE COLON R_SQUARE_BRACKET'''
        p[0] = []
        
    def p_globals_vars_selected(self, p):
        '''globals_vars : DEFINE COLON var_list R_SQUARE_BRACKET'''
        p[0] = p[3]

    def p_globals_definition(self, p):
        '''globals_definition : globals_vars STRING
                              | globals_vars MULTILINE_STRING
        '''
        p[0] = self.sat.define_sat_globals(p, source=p[2], variables=p[1])

    ### DOMAIN DEFINITION
    def p_const_integer(self, p):
        '''const_integer : SYMBOL
                         | INTEGER
        '''
        p[0] = make_const_value(self.sat, p[1])

    def p_domain_definition(self, p):
        'domain_definition : SYMBOL EQUALS domain'
        p[0] = self.sat.define_sat_domain(p, p[1], p[3])

    def p_domain(self, p):
        'domain : L_SQUARE_BRACKET domain_content R_SQUARE_BRACKET'
        p[0] = p[2]

    def p_domain_content_term(self, p):
        'domain_content : domain_term'
        p[0] = flatten(SatTerm.build(p[1]))

    def p_domain_content_list(self, p):
        'domain_content : domain_term COMMA domain_content'
        p[0] = flatten(p[1], p[3])

    def p_domain_term_number(self, p):
        'domain_term : const_integer'
        p[0] = SatTerm.build(p[1])

    def p_domain_term_range(self, p):
        '''domain_term : const_integer COLON const_integer'''
        start = p[1]
        stop = p[3]
        p[0] = SatRange(start, stop)

    def p_domain_term_range_stride(self, p):
        '''domain_term : const_integer COLON const_integer COLON const_integer'''
        start = p[1]
        stop = p[3]
        stride = p[5]
        p[0] = SatRange(start, stop, stride)

    ### OPTIONS
    def p_set_option(self, p):
        '''option_definition : DEF_OPTION LPAREN option_name COMMA option_value RPAREN'''
        self.sat.set_sat_option(p, p[3], p[5])

    def p_option_name(self, p):
        '''option_name : SYMBOL'''
        p[0] = p[1]

    def p_option_value(self, p):
        '''option_value : SYMBOL
                        | INTEGER
                        | FLOATING_POINT
        '''
        p[0] = p[1]

    ### VAR DEFINITION
    def p_domain_value(self, p):
        '''domain_value : domain
        '''
        p[0] = p[1]

    def p_var_definition_single(self, p):
        '''var_definition : var_list DEF_VAR var_domain
        '''
        p[0] = self.sat.define_sat_vars(p, p[3], *p[1])

    def p_var_list_single(self, p):
        '''var_list : SYMBOL'''
        p[0] = [p[1]]

    def p_var_list_multiple(self, p):
        '''var_list : var_list COMMA SYMBOL'''
        p[0] = p[1] + [p[3]]

    def p_var_domain(self, p):
        '''var_domain : SYMBOL
                      | domain_value
        '''
        p[0] = p[1]

    ### CONSTRAINT
    def p_constraint_definition(self, p):
        '''constraint_definition : expression'''
        p[0] = self.sat.add_sat_constraint(p, p[1])

    def p_all_different_constraint_definition(self, p):
        '''constraint_definition : ALL_DIFFERENT_CONSTRAINT LPAREN var_list RPAREN'''
        p[0] = self.sat.add_sat_all_different_constraint(p, p[1], p[3])

    precedence = (
        ('nonassoc', 'LE', 'LT', 'GE', 'GT', 'EQ', 'NE'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('right', 'POW'),
        ('right', 'UNARY'),
    )

    def p_expression(self, p):
        '''expression : expr_binop
                      | expr_unop
                      | paren_expression
                      | SYMBOL
                      | INTEGER
                      | input_definition
                      | function_call
        '''
        p[0] = make_value(self.sat, p[1])

    def p_paren_expression(self, p):
        '''paren_expression : LPAREN expression RPAREN'''
        p[0] = make_value(self.sat, p[2])

    def p_expr_unop(self, p):
        '''expr_unop : PLUS expression  %prec UNARY
                     | MINUS expression %prec UNARY
                     | NOT expression   %prec UNARY
        '''
        if p[1] == '+':
            value = p[2]
        elif p[1] == '-':
            value = -p[2]
        elif p[1] == '!':
            value = not p[2]
        else:
            raise RuntimeError('internal error: unexpected operator {}'.format(p[1]))
        p[0] = make_value(self.sat, value)

    def p_expr_binop(self, p):
        '''expr_binop : expression PLUS expression
                      | expression MINUS expression
                      | expression TIMES expression
                      | expression DIVIDE expression
                      | expression POW expression
                      | expression AND expression
                      | expression OR expression
                      | expression GT expression
                      | expression GE expression
                      | expression LT expression
                      | expression LE expression
                      | expression EQ expression
                      | expression NE expression
        '''
        p[0] = make_binop(self.sat, p[1], p[2], p[3])

    def p_function_call_posonly(self, p):
        '''function_call : SYMBOL LPAREN parameters RPAREN'''
        p[0] = self.sat.define_sat_function_call(p, p[1], p[3], {})

    def p_function_call_named(self, p):
        '''function_call : SYMBOL LPAREN parameters COMMA named_parameters RPAREN'''
        p[0] = self.sat.define_sat_function_call(p, p[1], p[3], p[5])

    def p_parameter(self, p):
        '''parameter : expression'''
        p[0] = p[1]

    def p_parameters_single(self, p):
        '''parameters : parameter'''
        p[0] = [p[1]]

    def p_parameters_multiple(self, p):
        '''parameters : parameters COMMA parameter'''
        p[0] = p[1] + [p[3]]

    def p_named_parameter(self, p):
        '''named_parameter : SYMBOL EQUALS expression'''
        p[0] = {p[1]: p[3]}

    def p_named_parameters_single(self, p):
        '''named_parameters : named_parameter'''
        p[0] = p[1]

    def p_named_parameters_multiple(self, p):
        '''named_parameters : named_parameters COMMA named_parameter'''
        p[0] = {**p[1], **p[3]}

    def p_macro(self, p):
        'macro_definition : SYMBOL DEF_MACRO expression'
        p[0] = self.sat.define_sat_macro(p, p[1], p[3])

    ### OBJECTIVE
    def p_objective_definition(self, p):
        'objective_definition : OBJECTIVE LPAREN expression RPAREN'
        p[0] = self.sat.add_sat_objective(p, p[1], p[3])
 
    ### ERROR:
    def p_error(self, t):
        raise SatSyntaxError("syntax error at line {}, token {!r} [{}]".format(t.lineno, t.value, t.type))


def sat_compile(source, **kwargs):
    if not isinstance(source, str):
        source = source.read()
    sat = Sat(**kwargs)
    parser = SatParser(sat)
    parser.parse(source)
    return sat
