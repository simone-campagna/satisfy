import collections
import itertools
import operator
import re

from .solver import ModelSolver, VarSelectionPolicy

__all__ = [
    'CryptarithmSolver',
]


    
class CryptarithmSolver(ModelSolver):
    def __init__(self, source, avoid_leading_zeros=True, **args):
        if args.get('var_selection_policy', None) is None:
            args['var_selection_policy'] = VarSelectionPolicy.MIN_BOUND
        super().__init__(**args)
        source = source.upper()
        numbers = set()
        word = re.compile(r'[A-Z0-9]+')
        mod_source = source
        offset = 0
        letters = set()
        non_zero_letters = set()
        if avoid_leading_zeros:
            for number in numbers:
                non_zero_letters.add(number[0])
                letters.update(number)
        variables = {}
        for m in word.finditer(source):
            number = m.group()
            if not number[0].isdigit():
                non_zero_letters.add(number[0])
            begin, end = m.span()
            parts = []
            for numeric, group in itertools.groupby(number, lambda x: x.isdigit()):
                parts.append((numeric, ''.join(group)))
            p10 = 1
            num_part = 0
            ldict = collections.defaultdict(list)
            for numeric, part in reversed(parts):
                if numeric:
                    num_part += int(part) * p10
                else:
                    for c, letter in enumerate(reversed(part)):
                        cp10 = p10 * (10 ** c)
                        ldict[letter].append(p10 * (10 ** c))
                    letters.update(part)
                p10 *= 10 ** len(part)
            expr_list = [str(num_part)]
            for letter, coeffs in ldict.items():
                factor = sum(coeffs)
                if factor == 1:
                    expr_list.append(letter)
                else:
                    expr_list.append('({} * {})'.format(letter, factor))
            expr = '(' + ' + '.join(expr_list) + ')'
            mod_source = mod_source[:offset + begin] + expr + mod_source[offset + end:]
            offset += len(expr) - (end - begin)
            numbers.add(m.group())
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
        # expressions = {}
        # for number in numbers:
        #     n_expr = 0
        #     for ipow, letter in enumerate(reversed(number)):
        #         n_expr += variables[letter] * (10 ** ipow)
        #     expressions[number] = n_expr
        # expr = eval(source, expressions)
        expr = eval(mod_source, variables.copy())
        model.add_all_different_constraint(list(variables.values()))
        model.add_constraint(expr)
        self._expr = expr
        self._source = source

    @property
    def source(self):
        return self._source

    @property
    def expr(self):
        return self._expr
