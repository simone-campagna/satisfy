from .demo_utils import (
    print_model,
    print_solve_stats,
)

__all__ = [
    'four_rings',
]

from .four_rings import FourRingsSolver

__all__ = [
    'four_rings',
    'four_rings_description',
]


def print_four_rings(fr_solver, solution, compact=False):
    keys = fr_solver.__keys__
    if compact:
        print([solution[key] for key in keys])
        return
    draw = fr_solver.__draw__
    s_data = {key: str(val) for key, val in solution.items()}
    max_len = max(len(val) for val in s_data.values())
    if max_len <= 3:
        s_draw = draw
        for key in keys:
            s_draw = s_draw.replace(' {} '.format(key), '{:^3s}'.format(s_data[key]))
        print(s_draw)
    else:
        print(draw)
        for key in keys:
            print("    {key} = {val}".format(key=key, val=s_data[key]))


def four_rings(low, high, unique, compact, timeout, limit, show_model, show_stats):
    fr_solver = FourRingsSolver(low=low, high=high, unique=unique, timeout=timeout, limit=limit)

    if show_model:
        print_model(fr_solver.model)

    num_solutions = 0
    for solution in fr_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_four_rings(fr_solver, solution, compact)
    if show_stats:
        print_solve_stats(fr_solver.get_stats())


def four_rings_description():
    return '\n'.join([FourRingsSolver.__task__, FourRingsSolver.__draw__])
