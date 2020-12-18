import argparse
import sys

from .demo_utils import iter_solutions
from ..sat import sat_compile

__all__ = [
    'sat_tool',
]


def print_solution(model_solver, solution):
    if model_solver.has_objectives():
        opt_result = solution
        solution = opt_result.solution
        if opt_result.is_optimal:
            msg = " [OPTIMAL]"
        else:
            msg = " [sub-optimal]"
    else:
        msg = ""
    if solution is None:
        sol = None
    else:
        sol = " ".join('{}={!r}'.format(var, solution[var]) for var in sorted(solution))
    print("{}{}".format(sol, msg))


def sat_tool(input_file, timeout, limit, show_model, show_stats, profile, compact):
    source = input_file.read()
    model_solver = sat_compile(source, timeout=timeout, limit=limit)
    for solution in iter_solutions(model_solver, show_model=show_model, show_stats=show_stats,
                                   profile=profile, compact=compact):
        print_solution(model_solver, solution)
