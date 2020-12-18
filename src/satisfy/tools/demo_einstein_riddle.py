from .demo_utils import (
    iter_solutions,
)

from .einstein_riddle import EinsteinRiddleSolver

__all__ = [
    'einstein',
]


def print_einstein_riddle_solution(solution):
    ordinal = {
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth',
    }
    fmt = "{found} The {owner}, who lives in the {hordinal} {color} house, drinks {drink}, smokes {smoke} and has {animal}."
    for hindex, var_d in solution.items():
        if var_d['animal'] == 'a zebra':
            found = "[*]"
        else:
            found = "   "
        print(fmt.format(hordinal=ordinal[hindex], found=found, **var_d))


def einstein(timeout, limit, show_model, show_stats, profile, compact):
    model_solver = EinsteinRiddleSolver(timeout=timeout, limit=limit)
    print(model_solver.riddle())

    for solution in iter_solutions(model_solver, show_model=show_model, show_stats=show_stats,
                                   profile=profile, compact=compact):
        print_einstein_riddle_solution(model_solver.create_riddle_solution(solution))
