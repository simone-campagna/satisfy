from .demo_utils import (
    print_model,
    print_solve_stats,
)

from .cryptarithm import CryptarithmSolver

__all__ = [
    'cryptarithm',
]


DEFAULT_CRYPTARITHM_SOURCE = "SEND+MORE==MONEY"


def default_cryptarithm_source():
    return DEFAULT_CRYPTARITHM_SOURCE


def print_cryptarithm_solution(solution, source):
    s = source
    for key in sorted(solution):
        print("{} = {}".format(key, solution[key]))
        s = s.replace(key, str(solution[key]))
    print("==> {}".format(s))
    

def cryptarithm(source, avoid_leading_zeros, timeout, limit, show_model):
    if source is None:
        source = default_cryptarithm_source()
        print("""\
No input source - using default cryptarithm example:
{example}
""".format(example=source))
    else:
        print(source)

    cryptarithm_solver = CryptarithmSolver(source, avoid_leading_zeros=avoid_leading_zeros, timeout=timeout, limit=limit)

    if show_model:
        print_model(cryptarithm_solver.model)

    num_solutions = 0
    for solution in cryptarithm_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_cryptarithm_solution(solution, source)
    print_solve_stats(cryptarithm_solver.get_stats())
