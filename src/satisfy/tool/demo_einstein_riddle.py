from .demo_utils import (
    print_model,
    print_solve_stats,
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


def einstein(timeout, limit, show_model, show_stats):
    einstein_solver = EinsteinRiddleSolver(timeout=timeout, limit=limit)
    print(einstein_solver.riddle())

    if show_model:
        print_model(einstein_solver.model)

    num_solutions = 0
    for solution in einstein_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_einstein_riddle_solution(solution)
    if show_stats:
        print_solve_stats(einstein_solver.get_stats())
