import json

from ..sudoku import SudokuSolver

from .demo_utils import (
    print_model,
    print_solve_stats,
)

__all__ = [
    'sudoku',
    'default_sudoku_source',
]


DEFAULT_SUDOKU_SOURCE = """
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

def default_sudoku_source():
    return DEFAULT_SUDOKU_SOURCE


def print_sudoku_schema(schema):
    values = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    def convert_value(value):
        if value in values:
            return str(value)
        else:
            return "."

    uline = '┌─────────┬─────────┬─────────┐'
    mline = '├─────────┼─────────┼─────────┤'
    bline = '└─────────┴─────────┴─────────┘'

    hline = uline
    def conv(v):
        if v:
            return str(v)
        else:
            return "."

    for br in range(3):
        print(hline)
        hline = mline
        for sr in range(3):
            r = (br * 3) + sr
            lst = [" {} ".format(conv(v)) for v in schema[r]]
            lst.insert(9, '│')
            lst.insert(6, '│')
            lst.insert(3, '│')
            lst.insert(0, '│')
            print(''.join(lst))
    print(bline)


def sudoku(input_file, timeout, limit, show_model, show_stats):
    if input_file is None:
        source = default_sudoku_source()
        print("""\
No input file - using default schema:
{example}
""".format(example=source))
        schema = json.loads(source)
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
    if show_stats:
        print_solve_stats(sudoku_solver.get_stats())
