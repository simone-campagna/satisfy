#!/usr/bin/env python

import collections.abc
import itertools
import logging
import operator

import ply.lex as lex
import ply.yacc as yacc
 
from .model import Model
from .solver import Solver, SelectVar, SelectValue
from .objective import Maximize, Minimize
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


def domain(*terms):
    domain = []
    for term in terms:
        if isinstance(term, list):
            domain.extend(term)
        else:
            domain.append(term)
    return domain


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


def make_binop(sat, l, op, r):
    return BINOP[op](make_value(sat, l), make_value(sat, r))




class SatProxy(collections.abc.Mapping):
    def __init__(self, sat, solution):
        self.__sat = sat
        self.__solution = solution

    def __getitem__(self, key):
        return self.__solution(key)

    def __len__(self):
        return len(self.__solution)

    def __iter__(self):
        yield from self.__solution

    @property
    def solution(self):
        return self.__solution

    @property
    def variables(self):
        return dict(self.__sat.variables())

    @property
    def constraints(self):
        return list(self.__sat.constraints())

    @property
    def objectives(self):
        return list(self.__sat.objectives())

    def __repr__(self):
        return repr(self.__solution)

    def __str__(self):
        return str(self.__solution)


class Sat(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domains = {}
        self.vars = {}
        self.macros = {}
        self.__select_var = SelectVar.max_bound
        self.__select_value = SelectValue.min_value
        self.__limit = None
        self.__timeout = None
        self.output = []

    def get_symbol(self, symbol):
        return self.vars[symbol]

    def render_solution(self, solution):
        if self.output:
            fmt = '\n'.join(self.output)
            data = {
                '_': SatProxy(self, solution),
            }
            for macro, expression in self.macros.items():
                data[macro] = expression.evaluate(solution)
            return fmt.format(**solution, **data)
        else:
            return solution

    def define_sat_output(self, p, output):
        self.output.append(output)

    def define_sat_domain(self, p, name, domain):
        # print("DEFINE DOMAIN {} := {!r}".format(name, domain))
        self.domains[name] = domain
        return domain

    def define_sat_vars(self, p, domain, *names):
        if isinstance(domain, str):
            domain = self.domains[domain]
        variables = []
        for name in names:
            # print("DEFINE VAR {} :: {!r}".format(name, domain))
            var = self.add_int_variable(domain, name=name)
            self.vars[name] = var
            variables.append(var)
        return variables

    def define_sat_macro(self, p, name, expression):
        self.vars[name] = expression
        self.macros[name] = expression

    def add_sat_constraint(self, p, constraint):
        self.add_constraint(constraint)
        return constraint

    def add_sat_all_different_constraint(self, p, constraint_type, var_list):
        self.add_all_different_constraint([self.vars[var] for var in var_list])
        return var_list

    def add_sat_objective(self, p, objective_name, expression):
        self.add_objective(OBJECTIVE[objective_name](expression))

    def set_sat_option(self, p, option_name, option_value):
        if option_name == 'select_var':
            if not hasattr(SelectVar, option_value):
                raise SatSyntaxError("illegal SelectVar {!r}".format(option_value))
            self.__select_var = getattr(SelectVar, option_value)
        elif option_name == 'select_value':
            if not hasattr(SelectValue, option_value):
                raise SatSyntaxError("illegal SelectValue {!r}".format(option_value))
            self.__select_option_value = getattr(SelectValue, option_value)
        elif option_name == 'limit':
            if not isinstance(option_value, int):
                raise SatSyntaxError("illegal limit {!r} of type {}".format(option_value, type(option_value).__name__))
            self.__limit = option_value
        elif option_name == 'timeout':
            if not isinstance(option_value, (int, float)):
                raise SatSyntaxError("illegal limit {!r} of type {}".format(option_value, type(option_value).__name__))
            self.__timeout = option_value
        else:
            raise SatSyntaxError("unknown option {!r}".format(option_name))

    def solver(self, **kwargs):
        args = dict(
            limit=kwargs.pop('limit', self.__limit),
            timeout=kwargs.pop('timeout', self.__timeout),
            select_var=kwargs.pop('select_var', self.__select_var),
            select_value=kwargs.pop('select_value', self.__select_value),
            **kwargs
        )
        return Solver(**args)


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
       'DEF_DOMAIN',
       'DEF_VAR',
       'DEF_MACRO',
       'DEF_OPTION',
       'DEF_OUTPUT',
       'NEWLINE',
       'OBJECTIVE',
       'ALL_DIFFERENT_CONSTRAINT',
       'SYMBOL',
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
    t_DEF_DOMAIN               = r'\='
    t_DEF_VAR                  = r'\:\:'
    t_DEF_MACRO                = r'\:\='

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

    def t_DEF_OUTPUT(self, t):
        r'\!.*'
        v = t.value.rstrip('\n')
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
        '''
        p[0] = [p[1]]

    ### OUTPUT DEFINITION
    def p_output_definition(self, p):
        'output_definition : DEF_OUTPUT'
        output_line = p[1].lstrip()[1:]
        p[0] = self.sat.define_sat_output(p, output_line)
        
    ### DOMAIN DEFINITION
    def p_domain_definition(self, p):
        'domain_definition : SYMBOL DEF_DOMAIN domain'
        #print("DEF", p[1], p[3])
        p[0] = self.sat.define_sat_domain(p, p[1], p[3])

    def p_domain(self, p):
        'domain : L_SQUARE_BRACKET domain_content R_SQUARE_BRACKET'
        p[0] = p[2]

    def p_domain_content_term(self, p):
        'domain_content : domain_term'
        p[0] = domain(p[1])

    def p_domain_content_list(self, p):
        'domain_content : domain_term COMMA domain_content'
        p[0] = domain(p[1], p[3])

    def p_domain_term_number(self, p):
        'domain_term : INTEGER'
        p[0] = p[1]

    def p_domain_term_range(self, p):
        '''domain_term : INTEGER COLON INTEGER'''
        p[0] = list(range(p[1], p[3] + 1))

    def p_domain_term_range_stride(self, p):
        '''domain_term : INTEGER COLON INTEGER COLON INTEGER'''
        p[0] = list(range(p[1], p[3] + 1, p[5]))

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
                      | INTEGER'''
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

    def p_macro(self, p):
        'macro_definition : SYMBOL DEF_MACRO expression'
        #print("DEF", p[1], p[3])
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


# def main():
#     source = '''\
#     D2 := [4:40]
# 
#     x, y, z :: D2
# 
#     x < y
#     y < z
#     x + y + z == 62
#     x * y * z == 2880
#     '''
#     
#     # # Give the lexer some input
#     # lexer = SatLexer()
#     # 
#     # for tok in lexer.parse(source):
#     #     print(tok)
# 
#     sat = Sat()
#     parser = SatParser(sat)
#     result = parser.parse(source)
#     for solution in sat:
#         print(solution)
# 
# if __name__ == "__main__":
#     main()
