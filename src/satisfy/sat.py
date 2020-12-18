#!/usr/bin/env python

import logging
import operator

import ply.lex as lex
import ply.yacc as yacc
 
from satisfy import ModelSolver, Maximize, Minimize


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
        v = sat.vars[v]
    return v


def make_binop(sat, l, op, r):
    return BINOP[op](make_value(sat, l), make_value(sat, r))


class Sat(ModelSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domains = {}
        self.vars = {}

    def define_domain(self, name, domain):
        # print("DEFINE DOMAIN {} := {!r}".format(name, domain))
        self.domains[name] = domain
        return domain

    def define_vars(self, domain, *names):
        if isinstance(domain, str):
            domain = self.domains[domain]
        variables = []
        for name in names:
            # print("DEFINE VAR {} :: {!r}".format(name, domain))
            var = self.model.add_int_variable(domain, name=name)
            self.vars[name] = var
            variables.append(var)
        return variables

    def add_constraint(self, constraint):
        self.model.add_constraint(constraint)
        return constraint

    def add_objective(self, objective_name, expression):
        self.model.add_objective(OBJECTIVE[objective_name](expression))


class SatLexer:
    # List of token names.   This is always required
    tokens = (
       'NUMBER',
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
       'SYMBOL',
       'DEF_DOMAIN',
       'DEF_VAR',
       'DEF_OBJECTIVE',
       'NEWLINE',
    )
    
    # Regular expression rules for simple tokens
    t_PLUS               = r'\+'
    t_MINUS              = r'-'
    t_TIMES              = r'\*'
    t_DIVIDE             = r'/'
    t_POW                = r'\*\*'
    t_GT                 = r'\>'
    t_GE                 = r'\>\='
    t_LT                 = r'\<'
    t_LE                 = r'\<\='
    t_EQ                 = r'\=\='
    t_NE                 = r'\!\='
    t_AND                = r'\&'
    t_OR                 = r'\|'
    t_NOT                = r'\!'
    t_LPAREN             = r'\('
    t_RPAREN             = r'\)'
    t_COLON              = r'\:'
    t_COMMA              = r'\,'
    t_SYMBOL             = r'[a-zA-Z]\w*'
    t_L_SQUARE_BRACKET   = r'\['
    t_R_SQUARE_BRACKET   = r'\]'
    t_DEF_DOMAIN         = r'\:\='
    t_DEF_VAR            = r'\:\:'
    t_DEF_OBJECTIVE      = r'objective\:'
    
    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)    
        return t
    
    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
        return t
    
    def t_COMMENT(self, t):
        r'\#.*\n*'
        pass

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

    def p_code_single_line(self, p):
        'code : code_line'
        p[0] = p[1]

    def p_code_multiple_lines(self, p):
        '''code : code_line NEWLINE code
                | code_line NEWLINE empty'''
        if p[3]:
            p[0] = p[1] +  p[3]
        else:
            p[0] = p[1]

    def p_code_line(self, p):
        '''code_line : domain_definition
                     | var_definition
                     | constraint_definition
                     | objective_definition'''
        p[0] = [p[1]]

    ### DOMAIN DEFINITION
    def p_domain_definition(self, p):
        'domain_definition : SYMBOL DEF_DOMAIN domain'
        #print("DEF", p[1], p[3])
        p[0] = self.sat.define_domain(p[1], p[3])

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
        'domain_term : NUMBER'
        p[0] = p[1]

    def p_domain_term_range(self, p):
        'domain_term : NUMBER COLON NUMBER'
        p[0] = list(range(p[1], p[3] + 1))

    ### VAR DEFINITION
    def p_domain_value(self, p):
        '''domain_value : domain
                        | SYMBOL'''
        p[0] = p[1]

    def p_var_definition_single(self, p):
        'var_definition : SYMBOL DEF_VAR domain_value'
        p[0] = self.sat.define_vars(p[3], *p[1])

    def p_var_definition_multiple(self, p):
        'var_definition : var_list DEF_VAR SYMBOL'
        p[0] = self.sat.define_vars(p[3], *p[1])

    def p_var_list_single(self, p):
        'var_list : SYMBOL'
        p[0] = [p[1]]

    def p_var_list_multiple(self, p):
        '''var_list : var_list COMMA var_list'''
        p[0] = p[1] + p[3]

    ### CONSTRAINT
    def p_constraint_definition(self, p):
        '''constraint_definition : comp_binop'''
        p[0] = self.sat.add_constraint(p[1])

    precedence = (
        ('nonassoc', 'LE', 'LT', 'GE', 'GT', 'EQ', 'NE'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('right', 'POW'),
    )
                       # | expression LT expression
                       # | expression LE expression
                       # | expression GT expression
                       # | expression GE expression
                       # | expression EQ expression
                       # | expression NE expression
                       # | expression AND expression
                       # | expression OR expression

    def p_expression(self, p):
        ''' expression : expr_binop
                       | comp_binop
                       | LPAREN expression RPAREN
                       | SYMBOL
                       | NUMBER'''
        p[0] = make_value(self.sat, p[1])

    def p_expr_binop(self, p):
        '''expr_binop : expression PLUS expression
                      | expression MINUS expression
                      | expression TIMES expression
                      | expression DIVIDE expression
                      | expression POW expression
        '''
        p[0] = make_binop(self.sat, p[1], p[2], p[3])

    def p_comp_binop(self, p):
        '''comp_binop : expression AND expression
                      | expression OR expression
                      | expression GT expression
                      | expression GE expression
                      | expression LT expression
                      | expression LE expression
                      | expression EQ expression
                      | expression NE expression
        '''
        p[0] = make_binop(self.sat, p[1], p[2], p[3])

    ### OBJECTIVE
    def p_objective_definition(self, p):
        'objective_definition : DEF_OBJECTIVE SYMBOL LPAREN expression RPAREN'
        p[0] = self.sat.add_objective(p[2], p[4])
 
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