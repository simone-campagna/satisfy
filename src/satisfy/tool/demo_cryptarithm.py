from .demo_utils import (
    print_model,
    print_solve_stats,
)

from ..cryptarithm import CryptarithmSolver

__all__ = [
    'cryptarithm',
]


DEFAULT_CRYPTARITHM_SYSTEM = ["SEND+MORE==MONEY"]


def default_cryptarithm_system():
    return DEFAULT_CRYPTARITHM_SYSTEM


def print_system(system, header=''):
    if isinstance(system, str):
        system = [system]
    if len(system) == 1:
        fmt = "{header}{source}"
    else:
        fmt = "{header}{count}) {source}"
    for count, source in enumerate(system):
        print(fmt.format(header=header, count=count, source=source))
        header = ' ' * len(header)


def print_cryptarithm_solution(solution, system):
    for key in sorted(solution):
        print("{} = {}".format(key, solution[key]))
    subst_system = []
    for source in system:
        subst_source = source
        for key in sorted(solution):
            subst_source = subst_source.replace(key, str(solution[key]))
        subst_system.append(subst_source)
    print_system(subst_system, header='===> ')
    

def cryptarithm(system, avoid_leading_zeros, timeout, limit, show_model, show_stats):
    if not system:
        system = default_cryptarithm_system()
        print("""\
No input source - using default cryptarithm example:
{example}
""".format(example=system))
    else:
        print("system:", system)

    cryptarithm_solver = CryptarithmSolver(system, avoid_leading_zeros=avoid_leading_zeros, timeout=timeout, limit=limit)

    if show_model:
        print_model(cryptarithm_solver.model)

    num_solutions = 0
    for solution in cryptarithm_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_cryptarithm_solution(solution, system)
    if show_stats:
        print_solve_stats(cryptarithm_solver.get_stats())
