import json

from ..knapsack import KnapsackOptimizer

from .demo_utils import (
    print_model,
    print_model,
    print_optimization_stats,
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


def knapsack(input_file, timeout, limit, show_model, show_stats):
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

    knapsack_optimizer = KnapsackOptimizer(values, capacities, weights, timeout=timeout, limit=limit)
    if show_model:
        print_model(knapsack_optimizer.model)

    knapsack_result = knapsack_optimizer()

    print("=== optimal_solution ===")
    print("is_optimal:", knapsack_result.is_optimal)
    if knapsack_result.solution is None:
        print("no solution found")
    else:
        print("solution:", repr(solution_string(knapsack_result.solution)))
        print("value:", knapsack_result.value)
        print("weights:", knapsack_result.weights)

    if show_stats:
        print_optimization_stats(knapsack_optimizer.get_stats(), optimal=knapsack_result.is_optimal)
