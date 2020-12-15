import collections
import itertools

from ..solver import ModelSolver, SelectVar, SelectValue

__all__ = [
    'FourRingsSolver',
]


class FourRingsSolver(ModelSolver):
    __keys__ = list('abcdefg')
    __task__ = """\
Replace a, b, c, d, e, f, and g with the decimal digits LOW -> HIGH
such that the sum of the letters inside of each of the four large squares add up to the same sum.
"""
    __draw__ = """\

            ╔══════════════╗      ╔══════════════╗
            ║              ║      ║              ║
            ║      a       ║      ║      e       ║
            ║              ║      ║              ║
            ║          ┌───╫──────╫───┐      ┌───╫─────────┐
            ║          │   ║      ║   │      │   ║         │
            ║          │ b ║      ║ d │      │ f ║         │
            ║          │   ║      ║   │      │   ║         │
            ║          │   ║      ║   │      │   ║         │
            ╚══════════╪═══╝      ╚═══╪══════╪═══╝         │
                       │       c      │      │      g      │
                       │              │      │             │
                       │              │      │             │
                       └──────────────┘      └─────────────┘
"""
    def __init__(self, low=0, high=9, unique=True, **args):
        if args.get('select_var', None) is None:
            args['select_var'] = SelectVar.min_bound
        if args.get('select_value', None) is None:
            args['select_value'] = SelectValue.min_value
        super().__init__(**args)
        model = self._model
        v_domain = list(range(low, high + 1))
        v = {key: model.add_int_variable(domain=v_domain, name=key) for key in 'abcdefg'}
        if unique:
            model.add_all_different_constraint(v.values())
            min_3_value = sum(v_domain[:3])
            max_2_value = sum(v_domain[-2:])
        else:
            min_3_value = 3 * v_domain[0]
            max_2_value = 2 * v_domain[-1]

        model.add_constraint(v['b'] + v['c'] + v['d'] == v['a'] + v['b'])
        model.add_constraint(v['d'] + v['e'] + v['f'] == v['a'] + v['b'])
        model.add_constraint(v['f'] + v['g'] == v['a'] + v['b'])
        model.add_constraint(v['a'] + v['b'] >= min_3_value)
        model.add_constraint(v['f'] + v['g'] >= min_3_value)
        model.add_constraint(v['b'] + v['c'] + v['d'] <= max_2_value)
        model.add_constraint(v['d'] + v['e'] + v['f'] <= max_2_value)
