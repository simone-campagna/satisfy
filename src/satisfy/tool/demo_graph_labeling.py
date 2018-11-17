import json

import networkx as nx

from ..graph_labeling import GraphLabelingSolver

from .demo_utils import (
    print_model,
    print_solve_stats,
)

__all__ = [
    'graph_labeling',
    'default_graph_labeling_source',
]


DEFAULT_GRAPH_LABELING_SOURCE = """
{
    "directed": false,
    "graph": {},
    "links": [
        {
            "source": "node1",
            "target": "node3"
        },
        {
            "source": "node1",
            "target": "node2"
        },
        {
            "source": "node1",
            "target": "node0"
        },
        {
            "source": "node3",
            "target": "node0"
        },
        {
            "source": "node3",
            "target": "node4"
        },
        {
            "source": "node2",
            "target": "node0"
        },
        {
            "source": "node2",
            "target": "node4"
        },
        {
            "source": "node0",
            "target": "node4"
        }
    ],
    "multigraph": false,
    "nodes": [
        {
            "id": "node1"
        },
        {
            "id": "node3"
        },
        {
            "id": "node2"
        },
        {
            "id": "node0"
        },
        {
            "id": "node4"
        }
    ]
}
"""


def default_graph_labeling_source():
    return DEFAULT_GRAPH_LABELING_SOURCE


def graph_labeling(input_file, labels, timeout, limit, show_model):
    if input_file is None:
        source = default_graph_labeling_source()
        print("""\
No input file - using default data:
{example}
""".format(example=source))
        data = json.loads(source)
    else:
        data = json.load(input_file)

    graph = nx.node_link_graph(data)

    graph_labeling_solver = GraphLabelingSolver(graph, labels, timeout=timeout, limit=limit)
    if show_model:
        print_model(graph_labeling_solver.model)

    num_solutions = 0
    for solution in graph_labeling_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print(solution)
    print_solve_stats(graph_labeling_solver.get_stats())
