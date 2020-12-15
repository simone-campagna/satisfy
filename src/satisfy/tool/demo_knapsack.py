import json

from ..knapsack import KnapsackOptimizer

from .demo_utils import (
    optimize,
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

    model_optimizer = KnapsackOptimizer(values, capacities, weights, timeout=timeout, limit=limit)
    for result in optimize(model_optimizer, show_model=show_model, show_stats=show_stats,
                           profile=profile, compact=compact):
        print("is_optimal:", result.is_optimal)
        if result.solution is None:
            print("no solution found")
        else:
            print("solution:", repr(solution_string(result.solution)))
            print("value:", result.value)
            print("weights:", result.weights)
