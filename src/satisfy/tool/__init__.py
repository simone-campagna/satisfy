import argparse

from ..utils import INFINITY

from .demo_cryptarithm import (
    cryptarithm,
    default_cryptarithm_source,
)
from .einstein_riddle import (
    EinsteinRiddleSolver,
)
from .demo_einstein_riddle import (
    einstein,
)
from .demo_graph_labeling import (
    graph_labeling,
    default_graph_labeling_source,
)
from .demo_knapsack import (
    knapsack,
    default_knapsack_source,
)
from .demo_ascii_map_coloring import (
    ascii_map_coloring,
    default_ascii_map_coloring_source,
)
from .demo_queens import (
    queens,
)
from .demo_sudoku import (
    sudoku,
    default_sudoku_source,
)


__all__ = [
    'main',
]


def type_on_off(x):
    x = x.lower()
    if x in {'on', 'true'}:
        return True
    elif x in {'off', 'false'}:
        return False
    else:
        return bool(int(x))


def main():
    common_args = {
        'formatter_class': argparse.RawDescriptionHelpFormatter
    }
    top_level_parser = argparse.ArgumentParser(
        description="""\
Satisfy tool - show some examples.

* knapsack: solve a generic multidimensional knapsack problem
* sudoku: solve a sudoku schema
* graph_labeling: solve a graph labeling problem
* ascii_map_coloring: solve a map coloring problem based on a simple ascii map
* queens: solve the N-queens problem, for a chessboard with side N
* einstein: solve the Einstein's riddle
* cryptarithm: solve cryptarithms, i.e. arithmetic equations where some numbers
  are substituted with letters (for instance 'AA3*55==6CAB')
""",
        **common_args)

    solve_args = ["timeout", "limit", "show_model"]

    subparsers = top_level_parser.add_subparsers()
    top_level_parser.set_defaults(
        function=top_level_parser.print_help,
        function_args=[])

    knapsack_parser = subparsers.add_parser(
        "knapsack",
        description="""\
Solve a N-dimensional knapsack problem.

The input file is a JSON file containing values, weights and capacities;
for instance, this is a 2-dimensional knapsack problem with 7 items:

{example}

""".format(example=default_knapsack_source()),
        **common_args)
    knapsack_parser.set_defaults(
        function=knapsack,
        function_args=["input_file"] + solve_args)

    sudoku_parser = subparsers.add_parser(
        "sudoku",
        description="""\
Solve a sudoku schema.

The input file is a JSON file containing a sudoku schema, where 0 marks
an unknown value. For instance:

{example}

""".format(example=default_sudoku_source()),
        **common_args)
    sudoku_parser.set_defaults(
        function=sudoku,
        function_args=["input_file"] + solve_args)

    graph_labeling_parser = subparsers.add_parser(
        "graph_labeling",
        description="""\
Solve a graph labeling problem.

The input file is a JSON file containing the problem in NetworkX node link format.
For instance:

{example}

""".format(example=default_graph_labeling_source()),
        **common_args)
    graph_labeling_parser.set_defaults(
        function=graph_labeling,
        function_args=["input_file", "labels"] + solve_args)

    ascii_map_coloring_parser = subparsers.add_parser(
        "ascii_map_coloring",
        description="""\
Solve a map_coloring problem.

The input file is a simple ascii map, for instance:

{example}

""".format(example=default_ascii_map_coloring_source()),
        **common_args)
    ascii_map_coloring_parser.set_defaults(
        function=ascii_map_coloring,
        function_args=["input_file", "colors"] + solve_args)

    for parser in knapsack_parser, sudoku_parser, graph_labeling_parser, ascii_map_coloring_parser:
        parser.add_argument(
            "-i", "--input-file",
            metavar="F",
            default=None,
            type=argparse.FileType('r'),
            help="input filename")

    graph_labeling_parser.add_argument(
        "-L", "--labels",
        nargs='+',
        default=['red', 'blue', 'green'],
        help="set node labels")

    ascii_map_coloring_parser.add_argument(
        "-C", "--colors",
        nargs='+',
        default=['red', 'blue', 'green', 'yellow'],
        help="set map colors")

    queens_parser = subparsers.add_parser(
        "queens",
        description="""\
Solve the N-queens problem: given an NxN chessboard, try to place N non
attacking queens.

""",
        **common_args)
    queens_parser.set_defaults(
        function=queens,
        function_args=["board_size"] + solve_args)

    queens_parser.add_argument(
        "-b", "--board-size",
        metavar="S",
        default=8,
        type=int,
        help="board size")

    einstein_parser = subparsers.add_parser(
        "einstein",
        description="""\
Solve the Einstein's riddle:
""" + EinsteinRiddleSolver.riddle(),
        **common_args)
    einstein_parser.set_defaults(
        function=einstein,
        function_args=[] + solve_args)

    cryptarithm_parser = subparsers.add_parser(
        "cryptarithm",
        description="""\
Solve cryptarithms, for instance:

{example}

""".format(example=default_cryptarithm_source()),
        **common_args)
    cryptarithm_parser.set_defaults(
        function=cryptarithm,
        function_args=["source", "avoid_leading_zeros"] + solve_args)

    cryptarithm_parser.add_argument(
        "-z", "--avoid-leading-zeros",
        type=type_on_off,
        nargs='?', const='on',
        default=True,
        help="avoid leading zeros in numbers")

    cryptarithm_parser.add_argument(
        "source",
        nargs='?', default=None,
        help="cryptarithm source")

    solve_parsers = [sudoku_parser, queens_parser, einstein_parser, knapsack_parser,
                     graph_labeling_parser, ascii_map_coloring_parser,
                     cryptarithm_parser]
    for parser in solve_parsers:
        parser.add_argument(
            "-t", "--timeout",
            metavar="S",
            default=None,
            nargs='?', const=INFINITY,
            type=float,
            help="solve timeout")

        parser.add_argument(
            "-l", "--limit",
            metavar="N",
            default=None,
            nargs='?', const=INFINITY,
            type=int,
            help="max number of solutions")

        parser.add_argument(
            "-s", "--show-model",
            default=False,
            action="store_true",
            help="show model variables and constraints")

    namespace = top_level_parser.parse_args()

    function = namespace.function
    kwargs = {
        arg: getattr(namespace, arg) for arg in namespace.function_args
    }
    function(**kwargs)
    return

