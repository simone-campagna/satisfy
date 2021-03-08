from .cli_utils import (
    solve,
)

__all__ = [
    'four_rings',
]

from .four_rings import FourRings

__all__ = [
    'four_rings',
    'four_rings_description',
]


def render_four_rings(model, solution):
    keys = model.__keys__
    draw = model.__draw__
    s_data = {key: str(val) for key, val in solution.items()}
    max_len = max(len(val) for val in s_data.values())
    lines = []
    if max_len <= 3:
        s_draw = draw
        for key in keys:
            s_draw = s_draw.replace(' {} '.format(key), '{:^3s}'.format(s_data[key]))
        lines.append(s_draw)
    else:
        lines.append(draw)
        for key in keys:
            lines.append("    {key} = {val}".format(key=key, val=s_data[key]))
    return '\n'.join(lines)


def four_rings(low, high, unique, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
    model = FourRings(low=low, high=high, unique=unique)

    def render_solution(solution):
        return render_four_rings(model, solution)

    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode,
          output_file=output_file, render_solution=render_solution)



def four_rings_description():
    return '\n'.join([FourRings.__task__, FourRings.__draw__])
