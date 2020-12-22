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


def knapsack(input_file, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
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
        
    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode,
          output_file=output_file, render_solution=render_knapsack_solution)
