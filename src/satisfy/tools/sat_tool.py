import argparse
import functools
import sys

from pathlib import Path

import argcomplete

from .cli_utils import sat_solve, add_solve_arguments, solve_arguments
from ..sat import sat_compile

__all__ = [
    'sat_tool',
]


DATA_DIR = Path(__file__).parent / 'data'

class HelpSyntaxAction(argparse.Action):
    """Syntax help"""
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help="show syntax help"):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        with open(DATA_DIR / 'syntax.txt', 'r') as f_handle:
            print(f_handle.read())
        parser.exit(0)


def sat_tool(source_file, timeout, limit, show_model, show_stats, profile, show_mode, output_file, input_file):
    source = source_file.read()
    model_builder = sat_compile(source)
    
    sat_solve(model_builder, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode,
          input_file=input_file, output_file=output_file)


def create_sat_parser(subparsers=None, name='sat', *, formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):
    if subparsers:
        function = functools.partial(subparsers.add_parser, name)
    else:
        function = argparse.ArgumentParser
    sat_parser = function(
        description="""\
Solve a generic model described as SAT file.

The SAT syntax is simple; for instance:

# ------------------------------------------------------------------------------
# Print a description of the problem:
[begin] <<<
A rectangular garden is to be constructed using a rock wall as one side of the garden,
and wire fencing for the other three sides.
        y
   +-----------+
   |           |
   |           | x
   |           |
   +###########+
       rock 

Given N meters of wire fencing, determine the dimensions that would create a garden of
maximum area. What is the maximum area?

>>>

# Read input parameter (fence length)
N := [input] "How many meters of fence? "

# Print a line for each found solution:
[solution] "solution[{_COUNT:2d}]: x={x} y={y} area={area} [elapsed: {_ELAPSED:.2f}s]"

# Print the optimal solution:
[optimal-solution] <<<
=== {_OPTIMAL} solution:

        {y:^3d}
   +-----------+
   |           |
   |           | {x:^3d} [area: {area}]
   |           |
   +###########+
       rock 

>>>

# Print something at the end:
[end] "All done! [elapsed: {_ELAPSED:.2f}s]"

### DOMAIN definition:
D = [1:N]

### VARIABLES definition:
x, y :: D

### MACROS definition:
area := x * y
fence_length := 2 * x + y

### CONSTRAINTS:
fence_length == N

### OPTIMIZATION:
maximize(area)
# ------------------------------------------------------------------------------

The command has bash autocompletion; to enable it run this command:

  $ eval "$(register-python-argcomplete satisfy)"

""",
        formatter_class=formatter_class,
        **kwargs)
    sat_parser.set_defaults(
        function=sat_tool,
        function_args=["source_file", "input_file"] + solve_arguments())

    sat_parser.add_argument(
        'source_file',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='SAT source file (defaults to STDIN)')

    add_solve_arguments(sat_parser)
    sat_parser.add_argument(
        '-i', '--input-file',
        default=sys.stdin,
        type=argparse.FileType('r'),
        help='input filename')

    sat_parser.add_argument(
        "-H", "--help-syntax",
        action=HelpSyntaxAction)

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
