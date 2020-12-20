import json

from ..knapsack import Knapsack

from .cli_utils import (
    solve,
)

__all__ = [
    'knapsack',
    'default_knapsack_source',
]


DEFAULT_KNAPSACK_SOURCE = """
{
    "capacities": [8, 6],
    "values": [
        10,
        12,
        20,
        23,
        15,
        14,
        16
    ],
    "weights": [
        [1, 1],
        [2, 1],
        [5, 3],
        [4, 4],
        [2, 2],
        [2, 2],
        [2, 3]
    ]
}
"""


def default_knapsack_source():
    return DEFAULT_KNAPSACK_SOURCE


def knapsack(input_file, timeout, limit, show_model, show_stats, profile, compact):
    if input_file is None:
        source = default_knapsack_source()
        print("""\
No input file - using default data:
{example}
""".format(example=source))
        data = json.loads(source)
    else:
        data = json.load(input_file)

    values = data['values']
    capacities = data['capacities']
    weights = data['weights']

    def solution_string(solution):
        selected = []
        for item in solution:
            if item:
                selected.append("X")
            else:
                selected.append("_")
        return " ".join(selected)

    model = Knapsack(values, capacities, weights)

    def render_knapsack_solution(solution):
        knapsack_solution = model.make_knapsack_solution(solution)
        return "{} [{:d}]".format(solution_string(knapsack_solution.solution), knapsack_solution.value)
        
    # REM def print_knapsack_result(optimization_result):
    # REM     if optimization_result.solution is not None:
    # REM         if optimization_result.is_optimal:
    # REM             sol_type = 'optimal'
    # REM         else:
    # REM             sol_type = 'sub-optimal'
    # REM         print("Found {} solution:".format(sol_type))
    # REM         print_knapsack_solution(optimization_result.count, optimization_result.solution)
    # REM     else:
    # REM         print("No solution found")

    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, compact=compact,
          render_solution=render_knapsack_solution)
