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
        # if args.get('select_value', None) is None:
        #     args['select_value'] = SelectValue.max_value
        super().__init__(**args)
        model = self._model
        v_domain = list(range(low, high + 1))
        v = {key: model.add_int_variable(domain=v_domain, name=key) for key in 'abcdefg'}
        if unique:
            model.add_all_different_constraint(v.values())

        model.add_constraint(v['b'] + v['c'] + v['d'] == v['a'] + v['b'])
        model.add_constraint(v['d'] + v['e'] + v['f'] == v['a'] + v['b'])
        model.add_constraint(v['f'] + v['g'] == v['a'] + v['b'])

    def __iter__(self):
        model = self._model
        solver = self._solver
        for solution in solver.solve(model):
            yield solution
