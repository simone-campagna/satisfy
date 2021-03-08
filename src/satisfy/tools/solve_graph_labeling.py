import json

import networkx as nx

from ..graph_labeling import GraphLabeling

from .cli_utils import (
    solve,
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


def graph_labeling(input_file, labels, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
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

    model = GraphLabeling(graph, labels)

    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode,
          output_file=output_file)
