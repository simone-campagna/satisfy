from .demo_utils import (
    iter_solutions,
)

from ..cryptarithm import CryptarithmSolver

__all__ = [
    'cryptarithm',
]


DEFAULT_CRYPTARITHM_SYSTEM = ["SEND+MORE==MONEY"]


def default_cryptarithm_system():
    return DEFAULT_CRYPTARITHM_SYSTEM


def print_cryptarithm_system(system, header=''):
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
    print_cryptarithm_system(subst_system, header='===> ')
    

def cryptarithm(system, avoid_leading_zeros, timeout, limit, show_model, show_stats, profile, compact):
    if not system:
        system = default_cryptarithm_system()
        print("No input source - using default cryptarithm example:")
    print_cryptarithm_system(system)

    model_solver = CryptarithmSolver(system, avoid_leading_zeros=avoid_leading_zeros, timeout=timeout, limit=limit)
    for solution in iter_solutions(model_solver, show_model=show_model, show_stats=show_stats,
                                   profile=profile, compact=compact):
        print_cryptarithm_solution(solution, system)
