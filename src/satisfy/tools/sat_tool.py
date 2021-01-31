import argparse
import functools
import sys

import argcomplete

from .cli_utils import sat_solve, add_solve_arguments, solve_arguments
from ..sat import sat_compile

__all__ = [
    'sat_tool',
]


def sat_tool(input_file, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
    source = input_file.read()
    model = sat_compile(source)
    
    sat_solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode, output_file=output_file)


def create_sat_parser(subparsers=None, name='sat', *, formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):
    if subparsers:
        function = functools.partial(subparsers.add_parser, name)
    else:
        function = argparse.ArgumentParser
    sat_parser = function(
        description="""\
Solve a generic model described as SAT file.

The SAT syntax is simple; for instance:

  # Lines beginning with <| are show at the beginning:
  <|A rectangular garden is to be constructed using a rock wall as one side of the garden,
  <|and wire fencing for the other three sides.
  <|        y
  <|   +-----------+
  <|   |           |
  <|   |           | x
  <|   |           |
  <|   +###########+
  <|       rock 
  <|Given  100 m of wire fencing, determine the dimensions that would create a garden of
  <|maximum area. What is the maximum area?
  <|
  
  # Lines beginning with !| are shown for each found solution:
  !|solution[{_stats.count:2d}]: x={x} y={y} area={area} [elapsed: {_stats.elapsed:.2f}s]
  
  # Lines beginning with $| are shown only for optimization problems:
  $|=== {_optimal} solution:
  $|
  $|        {y:^3d}
  $|   +-----------+
  $|   |           |
  $|   |           | {x:^3d} [area: {area}]
  $|   |           |
  $|   +###########+
  $|       rock 
  $|
  
  # Lines beginning with >| are show at the end:
  >|All done! [elapsed: {_stats.elapsed:.2f}s]
  
  ### DOMAIN definition:
  D = [0:100]
  
  ### VARIABLES definition:
  x, y :: D
  
  ### MACROS definition:
  area := x * y
  fence_length := 2 * x + y
  
  ### CONSTRAINTS:
  fence_length == 100
  
  ### OPTIMIZATION:
  maximize(area)

The command has bash autocompletion; to enable it run this command:

  $ eval "$(register-python-argcomplete satisfy)"

""",
        formatter_class=formatter_class,
        **kwargs)
    sat_parser.set_defaults(
        function=sat_tool,
        function_args=["input_file"] + solve_arguments())

    sat_parser.add_argument(
        'input_file',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='SAT input file (defaults to STDIN)')

    add_solve_arguments(sat_parser)
    return sat_parser


def main():
    parser = create_sat_parser()
    argcomplete.autocomplete(parser)
    namespace = parser.parse_args()
    profile = namespace.profile

    function = namespace.function
    kwargs = {
        arg: getattr(namespace, arg) for arg in namespace.function_args
    }
    function(**kwargs)
    return
