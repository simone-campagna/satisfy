from .demo_utils import (
    iter_solutions,
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


def queens(board_size, timeout, limit, show_model, show_stats, profile, compact):
    model_solver = QueensSolver(board_size, timeout=timeout, limit=limit)
    for solution in iter_solutions(model_solver, show_model=show_model, show_stats=show_stats,
                                   profile=profile, compact=compact):
        print_queens_board(board_size, model_solver.create_board(solution))
