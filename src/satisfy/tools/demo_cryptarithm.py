from .cli_utils import (
    solve,
)

from ..cryptarithm import Cryptarithm

__all__ = [
    'cryptarithm',
]


DEFAULT_CRYPTARITHM_SYSTEM = ["SEND+MORE==MONEY"]


def default_cryptarithm_system():
    return DEFAULT_CRYPTARITHM_SYSTEM


def render_cryptarithm_system(system, header=''):
    lines = []
    if isinstance(system, str):
        system = [system]
    if len(system) == 1:
        fmt = "{header}{source}"
    else:
        fmt = "{header}{count}) {source}"
    for count, source in enumerate(system):
        lines.append(fmt.format(header=header, count=count, source=source))
        header = ' ' * len(header)
    return '\n'.join(lines)


def render_cryptarithm_solution(solution, system):
    lines = []
    for key in sorted(solution):
        lines.append("{} = {}".format(key, solution[key]))
    subst_system = []
    for source in system:
        subst_source = source
        for key in sorted(solution):
            subst_source = subst_source.replace(key, str(solution[key]))
        subst_system.append(subst_source)
    lines.append(render_cryptarithm_system(subst_system, header='===> '))
    return '\n'.join(lines)
    

def cryptarithm(system, avoid_leading_zeros, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
    if not system:
        system = default_cryptarithm_system()
        print("No input source - using default cryptarithm example:")
    print(render_cryptarithm_system(system))

    model = Cryptarithm(system, avoid_leading_zeros=avoid_leading_zeros)

    def render_cryptarithm(solution):
        return render_cryptarithm_solution(solution, system)

    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode,
          output_file=output_file, render_solution=render_cryptarithm)
