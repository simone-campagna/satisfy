from .cli_utils import (
    solve,
)

from .einstein_riddle import EinsteinRiddle

__all__ = [
    'einstein',
]


def render_riddle_solution(solution):
    ordinal = {
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth',
    }
    fmt = "{found} The {owner}, who lives in the {hordinal} {color} house, drinks {drink}, smokes {smoke} and has {animal}."
    lines = []
    for hindex, var_d in solution.items():
        if var_d['animal'] == 'a zebra':
            found = "[*]"
        else:
            found = "   "
        lines.append(fmt.format(hordinal=ordinal[hindex], found=found, **var_d))
    return '\n'.join(lines)


def einstein(timeout, limit, show_model, show_stats, profile, compact):
    model = EinsteinRiddle()
    print(model.riddle())

    def render_riddle(solution):
        return render_riddle_solution(model.create_riddle_solution(solution))

    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, compact=compact,
          render_solution=render_riddle)
