from .demo_utils import (
    iter_solutions,
)

__all__ = [
    'four_rings',
]

from .four_rings import FourRingsSolver

__all__ = [
    'four_rings',
    'four_rings_description',
]


def print_four_rings(fr_solver, solution):
    keys = fr_solver.__keys__
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


def four_rings(low, high, unique, timeout, limit, show_model, show_stats, profile, compact):
    model_solver = FourRingsSolver(low=low, high=high, unique=unique, timeout=timeout, limit=limit)
    for solution in iter_solutions(model_solver, show_model=show_model, show_stats=show_stats,
                                   profile=profile, compact=compact):
        print_four_rings(model_solver, solution)


def four_rings_description():
    return '\n'.join([FourRingsSolver.__task__, FourRingsSolver.__draw__])
