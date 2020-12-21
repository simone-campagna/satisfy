import argparse
import json
import sys

import argcomplete

from ..utils import INFINITY

from .cli_utils import ShowMode

from .demo_cryptarithm import (
    cryptarithm,
    default_cryptarithm_system,
)
from .einstein_riddle import (
    EinsteinRiddle,
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
from .demo_nonogram import (
    nonogram,
    default_nonogram_source,
)
from .demo_queens import (
    queens,
)
from .demo_sudoku import (
    sudoku,
    default_sudoku_source,
)
from .demo_four_rings import (
    four_rings,
    four_rings_description,
)
from .sat_tool import (
    sat_tool,
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


def json_file(filename):
    with open(filename, "r") as f_in:
        return json.load(f_in)


def main():
    common_args = {
        'formatter_class': argparse.RawDescriptionHelpFormatter
    }
    top_level_parser = argparse.ArgumentParser(
        description="""\
Satisfy tool - show some examples.

* knapsack: solve a generic multidimensional knapsack problem
* sudoku: solve a sudoku schema
* graph-labeling: solve a graph labeling problem
* ascii-map-coloring: solve a map coloring problem based on a simple ascii map
* queens: solve the N-queens problem, for a chessboard with side N
* einstein-riddle: solve the Einstein's riddle
* cryptarithm: solve cryptarithms, i.e. arithmetic equations where some numbers
  are substituted with letters (for instance 'AA3*55==6CAB')
* nonogram: solve nonograms
* 4-rings: solve the 4 rings problem
""",
        **common_args)

    solve_args = ["timeout", "limit", "show_model", "show_stats", "profile", "show_mode"]

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
        "graph-labeling",
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
        "ascii-map-coloring",
        description="""\
Solve a map_coloring problem.

The input file is a simple ascii map, for instance:

{example}

""".format(example=default_ascii_map_coloring_source()),
        **common_args)
    ascii_map_coloring_parser.set_defaults(
        function=ascii_map_coloring,
        function_args=["input_file", "colors"] + solve_args)

    nonogram_parser = subparsers.add_parser(
        "nonogram",
        description="""\
Solve a nonogram.

The input file is a JSON file containing a nonogram definition.
For instance:

{example}

""".format(example=default_nonogram_source()),
        **common_args)
    nonogram_parser.set_defaults(
        function=nonogram,
        function_args=["input_file", "input_image"] + solve_args)

    input_group = nonogram_parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-i", "--input-file",
        metavar="F",
        default=None,
        type=argparse.FileType('r'),
        help="input filename")
    input_group.add_argument(
        "-I", "--input-image",
        metavar="F",
        default=None,
        type=argparse.FileType('r'),
        help="input image")

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
        "-B", "--board-size",
        metavar="S",
        default=8,
        type=int,
        help="board size")

    einstein_parser = subparsers.add_parser(
        "einstein-riddle",
        description="""\
Solve the Einstein's riddle:
""" + EinsteinRiddle.riddle(),
        **common_args)
    einstein_parser.set_defaults(
        function=einstein,
        function_args=[] + solve_args)

    cryptarithm_parser = subparsers.add_parser(
        "cryptarithm",
        description="""\
Solve cryptarithms, for instance:

{example}

""".format(example=' '.join(default_cryptarithm_system())),
        **common_args)
    cryptarithm_parser.set_defaults(
        function=cryptarithm,
        function_args=["system", "avoid_leading_zeros"] + solve_args)

    cryptarithm_parser.add_argument(
        "-z", "--avoid-leading-zeros",
        type=type_on_off,
        nargs='?', const='on',
        default=True,
        help="avoid leading zeros in numbers")

    input_group = cryptarithm_parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-i", "--input-file",
        dest="system",
        metavar="F",
        default=None,
        type=json_file,
        help="input JSON filename")
    input_group.add_argument(
        "-e", "--equation",
        dest="system",
        action="append",
        default=[],
        help="cryptarithm equation")

    four_rings_parser = subparsers.add_parser(
        "4-rings",
        description="""\
Solve the 4-rings problem:

{description}

""".format(description=four_rings_description()),
        **common_args)
    four_rings_parser.set_defaults(
        function=four_rings,
        function_args=["low", "high", "unique"] + solve_args)

    four_rings_parser.add_argument(
        '-L', '--low',
        type=int, default=0,
        help='minimum value for the variables (defaults to 0)')

    four_rings_parser.add_argument(
        '-H', '--high',
        type=int, default=9,
        help='maximum value for the variables (defaults to 0)')
    fr_unique_mgrp = four_rings_parser.add_mutually_exclusive_group()
    fr_unique_kwargs = {'dest': 'unique', 'default': True}
    fr_unique_mgrp.add_argument(
        '-u', '--unique',
        action='store_true',
        help='unique variable values (default)',
        **fr_unique_kwargs)
    fr_unique_mgrp.add_argument(
        '-U', '--not-unique',
        action='store_false',
        help='allows not unique variable values',
        **fr_unique_kwargs)

    sat_parser = subparsers.add_parser(
        "sat",
        description="""\
Solve a generic model described as SAT file.

The SAT syntax is simple; for instance:

  ### DOMAIN DEFINITION ######################################
  # you can optionally create domains with a given name;
  # domain definition can contain
  # - integers [1, 2, 3]
  # - ranges [10:15, 20:30]
  # - ranges with strides [1:100:3]
  
  # create domain 'D0' containing [1, 2, 10, 11, 12, 13, 14, 15]
  D0 := [1, 2, 10:15]
  
  ### VARIABLE DEFINITION ####################################
  # you must define the variables that you're going to use
  # in constraints and optimization objectives;
  # the variable definition requires the variable domain:
  # - a named domain
  # - a domain expression
  # you can declare a list of variables with the same domain
  
  # create variable 'x' with domain 'D0'
  x :: D0
  
  # create variables 'y', 'z' with domain [1, 3, 5]
  y, z :: [1:5:2]
  
  ### CONSTRAINTS ############################################
  # you can create constraints:
  
  # create constraint x + y + z == 10
  x + y + z > 10
  
  # create constraint x + y + z == 10
  x * y * z < 100
  
  # create all_different constraint
  all_different(x, y, z)
  
  ### OPTIMIZATION ###########################################
  # you can optionally set optimization objectives:
  # - minimize
  # - maximize
  
  # maximize x - z
  maximize(x - z)
  
  ### OPTIONS ################################################
  # you can optionally set solver options:
  # - select_var [SelectVar]
  # - select_value [SelectValue]
  # - limit [int]
  # - timeout [float]
  
  # set select_var = SelectVar.in_order
  option(select_var, in_order)
  
  # set select_value = SelectValue.random
  option(select_value, random)
  
  # set limit = 3
  # option(limit, 3)
  
  # set timeout = 0.5
  # option(timeout, 0.5)
""",
        **common_args)
    sat_parser.set_defaults(
        function=sat_tool,
        function_args=["input_file"] + solve_args)

    sat_parser.add_argument(
        'input_file',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='SAT input file (defaults to STDIN)')

    for parser in [knapsack_parser, graph_labeling_parser, ascii_map_coloring_parser, sudoku_parser]:
        parser.add_argument(
            "-i", "--input-file",
            metavar="F",
            default=None,
            type=argparse.FileType('r'),
            help="input filename")
# 
#     solve_parsers = [sudoku_parser, queens_parser, einstein_parser, knapsack_parser,
#                      graph_labeling_parser, ascii_map_coloring_parser,
#                      cryptarithm_parser, nonogram_parser, four_rings_parser, sat_parser]
    solve_parsers = [knapsack_parser, nonogram_parser, cryptarithm_parser, einstein_parser,
                     graph_labeling_parser, ascii_map_coloring_parser, queens_parser,
                     sudoku_parser, four_rings_parser, sat_parser]

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

        def _default(b_value):
            if b_value:
                return " (default)"
            return ""

        show_model_group = parser.add_mutually_exclusive_group()
        default_show_model = False
        show_model_group.add_argument(
            "-m", "--show-model",
            dest='show_model',
            default=default_show_model,
            action="store_true",
            help="show model variables and constraints" + _default(default_show_model))
        show_model_group.add_argument(
            "-M", "--no-show-model",
            dest='show_model',
            default=default_show_model,
            action="store_false",
            help="show model variables and constraints" + _default(not default_show_model))

        show_stats_group = parser.add_mutually_exclusive_group()
        default_show_stats = True
        show_stats_group.add_argument(
            "-s", "--show-stats",
            dest='show_stats',
            default=default_show_stats,
            action="store_true",
            help="show solver statistics" + _default(default_show_stats))
        show_stats_group.add_argument(
            "-S", "--no-show-stats",
            dest='show_stats',
            default=default_show_stats,
            action="store_false",
            help="show solver statistics" + _default(not default_show_stats))

        profile_group = parser.add_mutually_exclusive_group()
        profile_group.add_argument(
            "-p", "--profile",
            dest='profile',
            action="store_true", default=False,
            help="enable profiling")
        profile_group.add_argument(
            "-P", "--no-profile",
            dest='profile',
            action="store_false", default=False,
            help="disable profiling")

        show_mode_group = parser.add_argument_group("show solution mode")
        show_mode_kwargs = {'dest': 'show_mode', 'default': ShowMode.DEFAULT}
        show_mode_group.add_argument(
            '-c', '--compact',
            action='store_const', const=ShowMode.COMPACT,
            help='compact output',
            **show_mode_kwargs)
        show_mode_group.add_argument(
            '-b', '--brief',
            action='store_const', const=ShowMode.BRIEF,
            help='minimal output',
            **show_mode_kwargs)
        show_mode_group.add_argument(
            '-q', '--quiet',
            action='store_const', const=ShowMode.QUIET,
            help='do not show solutions',
            **show_mode_kwargs)

    argcomplete.autocomplete(top_level_parser)
    namespace = top_level_parser.parse_args()
    profile = namespace.profile

    function = namespace.function
    kwargs = {
        arg: getattr(namespace, arg) for arg in namespace.function_args
    }
    function(**kwargs)
    return

