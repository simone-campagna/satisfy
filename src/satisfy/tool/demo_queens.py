from .demo_utils import (
    print_model,
    print_solve_stats,
)

__all__ = [
    'queens',
]

from .queens import QueensSolver

__all__ = [
    'queens',
]


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


def queens(board_size, timeout, limit, show_model, show_stats):
    num_solutions = 0
    queens_solver = QueensSolver(board_size, timeout=timeout, limit=limit)

    if show_model:
        print_model(queens_solver.model)

    for board in queens_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_queens_board(board_size, board)
    if show_stats:
        print_solve_stats(queens_solver.get_stats())
