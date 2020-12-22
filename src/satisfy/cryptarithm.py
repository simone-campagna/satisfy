import collections
import itertools
import operator
import re

from .model import Model, SelectVar, SelectValue
from .solver import Solver, SelectVar, SelectValue

__all__ = [
    'Cryptarithm',
]


    
class Cryptarithm(Model):
    def __init__(self, system, avoid_leading_zeros=True, **args):
        if isinstance(system, str):
            system = [system]
        numbers = set()
        word = re.compile(r'[A-Z0-9]+')
        letters = set()
        non_zero_letters = set()
        variables = {}
        super().__init__(**args)
        self._system = []
        self._expressions = []
        mod_sources = []
        for source in system:
            source = source.upper().replace('\n', ' ')
            mod_source = source
            offset = 0
            for m in word.finditer(source):
                number = m.group()
                if avoid_leading_zeros and len(number) > 1 and not number[0].isdigit():
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
            mod_sources.append(mod_source)
        digits = tuple(range(10))
        non_zero_digits = digits[1:]
        variables = {}
        for letter in letters:
            if letter in non_zero_letters:
                domain = non_zero_digits
            else:
                domain = digits
            variables[letter] = self.add_int_variable(domain=domain, name=letter)
        # expressions = {}
        # for number in numbers:
        #     n_expr = 0
        #     for ipow, letter in enumerate(reversed(number)):
        #         n_expr += variables[letter] * (10 ** ipow)
        #     expressions[number] = n_expr
        # expr = eval(source, expressions)
        elist = []
        slist = []
        for source, mod_source in zip(system, mod_sources):
            expression = eval(mod_source, variables.copy())
            self.add_all_different_constraint(list(variables.values()))
            self.add_constraint(expression)
            elist.append(expression)
            slist.append(source)
        self._expressions = tuple(elist)
        self._system = tuple(slist)


    def solver(self, **kwargs):
        return Solver(
            select_var=kwargs.pop('select_var', SelectVar.max_bound),
            select_value=kwargs.pop('select_value', SelectValue.max_value),
            **kwargs
        )

    @property
    def system(self):
        return self._system

    @property
    def expressions(self):
        return self._expressions
