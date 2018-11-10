import argparse
import json

from .utils import INFINITY
from .knapsack import KnapsackOptimizer
from .queens import QueensSolver
from .sudoku import SudokuSolver
from .einstein_riddle import EinsteinRiddleSolver

__all__ = [
    'main',
]


KNAPSACK_EXAMPLE = """
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

SUDOKU_EXAMPLE = """
[
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0]
]
"""


def print_model(model):
    print("\n=== model variables: ===")
    for var_index, (var_name, var_info) in enumerate(model.variables().items()):
        print(" {:4d}) {!r} domain: {}".format(var_index, var_name, var_info.domain))
    print("\n=== model constraints: ===")
    for c_index, constraint in enumerate(model.constraints()):
        print(" {:4d}) {}".format(c_index, constraint))


def print_solve_stats(stats):
    if stats.count == 1:
        suffix = ''
    else:
        suffix = 's'
    if stats.interrupt:
        fmt = "Found {s.count} partial solution{suffix} in {s.elapsed:.3f} seconds [{s.interrupt} reached]"
    else:
        if stats.count == 1:
            fmt = "Found unique solution{suffix} in {s.elapsed:.3f} seconds"
        else:
            fmt = "Found all {s.count} solution{suffix} in {s.elapsed:.3f} seconds"
    print("\n" + fmt.format(suffix=suffix, s=stats))


def print_optimization_stats(stats, optimal=None):
    if optimal is None:
        optimal = stats.interrupt is None

    if optimal:
        kind = 'optimal'
    else:
        kind = 'sub-optimal'

    if stats.count == 1:
        suffix = ''
    else:
        suffix = 's'
    fmt = "Found {kind} solution in {s.elapsed:.3f} seconds after {s.count} solve iteration{suffix}"
    if stats.interrupt:
        fmt += " [{s.interrupt} reached]"
    print("\n" + fmt.format(suffix=suffix, kind=kind, s=stats))


def knapsack(input_file, timeout, limit, show_model):
    if input_file is None:
        print("""\
No input file - using default data:
{example}
""".format(example=KNAPSACK_EXAMPLE))
        data = json.loads(KNAPSACK_EXAMPLE)
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

    print_optimization_stats(knapsack_optimizer.get_stats(), optimal=knapsack_result.is_optimal)


def print_sudoku_schema(schema):
    values = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    def convert_value(value):
        if value in values:
            return str(value)
        else:
            return "."

    lst = [list(convert_value(value) for value in row) for row in schema]
    table = []
    hdr = ['-' for _ in range(9)]
    hdr.insert(6, '+')
    hdr.insert(3, '+')
    hdr_line = '-'.join(hdr)
    for lst_row in lst:
        lst_row.insert(6, '|')
        lst_row.insert(3, '|')
        table.append(' '.join(lst_row))
    mlen = max(len(row) for row in table)
    table.insert(6, hdr_line)
    table.insert(3, hdr_line)
    for row in table:
        print(row)


def sudoku(input_file, timeout, limit, show_model):
    if input_file is None:
        print("""\
No input file - using default schema:
{example}
""".format(example=SUDOKU_EXAMPLE))
        schema = json.loads(SUDOKU_EXAMPLE)
    else:
        schema = json.load(input_file)

    print("=== sudoku schema ===")
    print_sudoku_schema(schema)
    num_solutions = 0
    sudoku_solver = SudokuSolver(schema, timeout=timeout, limit=limit)
    if show_model:
        print_model(sudoku_solver.model)
    for schema in sudoku_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_sudoku_schema(schema)
    print_solve_stats(sudoku_solver.get_stats())


def print_queens_board(board_size, board):
    queen = " ♛ "
    empty = "   "
    uline = '┌' + '┬'.join(['───'] * board_size) + '┐'
    mline = '├' + '┼'.join(['───'] * board_size) + '┤'
    bline = '└' + '┴'.join(['───'] * board_size) + '┘'
    sep = '+' * (board_size + 1)
    hline = uline
    for board_row in board:
        print(hline)
        hline = mline
        row = []
        for is_queen in board_row:
            if is_queen:
                row.append(queen)
            else:
                row.append(empty)
        print('│' + '│'.join(row) + '│')
    print(bline)


def queens(board_size, timeout, limit, show_model):
    num_solutions = 0
    queens_solver = QueensSolver(board_size, timeout=timeout, limit=limit)

    if show_model:
        print_model(queens_solver.model)

    for board in queens_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_queens_board(board_size, board)
    print_solve_stats(queens_solver.get_stats())


def print_einstein_riddle_solution(solution):
    ordinal = {
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth',
    }
    fmt = "{found} The {owner}, who lives in the {hordinal} {color} house, drinks {drink}, smokes {smoke} and has {animal}."
    for hindex, var_d in solution.items():
        if var_d['animal'] == 'a zebra':
            found = "[*]"
        else:
            found = "   "
        print(fmt.format(hordinal=ordinal[hindex], found=found, **var_d))


def einstein(timeout, limit, show_model):
    einstein_solver = EinsteinRiddleSolver(timeout=timeout, limit=limit)
    print(einstein_solver.riddle())

    if show_model:
        print_model(einstein_solver.model)

    num_solutions = 0
    for solution in einstein_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_einstein_riddle_solution(solution)
    print_solve_stats(einstein_solver.get_stats())


def main():
    common_args = {
        'formatter_class': argparse.RawDescriptionHelpFormatter
    }
    top_level_parser = argparse.ArgumentParser(
        description="""\
Satisfy tool - show some examples.

* knapsack: solve a generic multidimensional knapsack problem
* sudoku: solve a sudoku schema
* queens: solve the N-queens problem, for a chessboard with side N
* einstein: solve the Einstein's riddle
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

""".format(example=KNAPSACK_EXAMPLE),
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

""".format(example=SUDOKU_EXAMPLE),
        **common_args)
    sudoku_parser.set_defaults(
        function=sudoku,
        function_args=["input_file"] + solve_args)

    for parser in knapsack_parser, sudoku_parser:
        parser.add_argument(
            "-i", "--input-file",
            metavar="F",
            default=None,
            type=argparse.FileType('r'),
            help="input filename")

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

    solve_parsers = [sudoku_parser, queens_parser, einstein_parser, knapsack_parser]
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

