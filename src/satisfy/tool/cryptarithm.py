import itertools
import operator
import re

from ..solver import ModelSolver, VarSelectionPolicy

__all__ = [
    'CryptarithmSolver',
]


    
class CryptarithmSolver(ModelSolver):
    def __init__(self, source, avoid_leading_zeros=True, **args):
        #if args.get('var_selection_policy', None) is None:
        #    args['var_selection_policy'] = VarSelectionPolicy.ORDERED
        super().__init__(**args)
        source = source.upper()
        numbers = set()
        r = re.compile(r'[A-Z]+')
        for m in r.finditer(source):
            numbers.add(m.group())
        letters = set()
        non_zero_letters = set()
        if avoid_leading_zeros:
            for number in numbers:
                non_zero_letters.add(number[0])
                letters.update(number)
        digits = tuple(range(10))
        non_zero_digits = digits[1:]
        variables = {}
        model = self._model
        for letter in letters:
            if letter in non_zero_letters:
                domain = non_zero_digits
            else:
                domain = digits
            variables[letter] = model.add_int_variable(domain=domain, name=letter)
        expressions = {}
        for number in numbers:
            n_expr = 0
            for ipow, letter in enumerate(reversed(number)):
                n_expr += variables[letter] * (10 ** ipow)
            expressions[number] = n_expr
        expr = eval(source, expressions)
        model.add_all_different_constraint(variables.values())
        model.add_constraint(expr)
        self._expr = expr
        self._source = source

    @property
    def source(self):
        return self._source

    @property
    def expr(self):
        return self._expr
