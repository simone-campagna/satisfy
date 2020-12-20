from .cli_utils import (
    solve,
)

__all__ = [
    'queens',
]

from .queens import Queens

__all__ = [
    'queens',
]


def render_queens_board(board_size, board):
    queen = " ♛ "
    empty = "   "
    uline = '┌' + '┬'.join(['───'] * board_size) + '┐'
    mline = '├' + '┼'.join(['───'] * board_size) + '┤'
    bline = '└' + '┴'.join(['───'] * board_size) + '┘'
    sep = '+' * (board_size + 1)
    hline = uline
    lines = []
    for board_row in board:
        lines.append(hline)
        hline = mline
        row = []
        for is_queen in board_row:
            if is_queen:
                row.append(queen)
            else:
                row.append(empty)
        lines.append('│' + '│'.join(row) + '│')
    lines.append(bline)
    return '\n'.join(lines)


def queens(board_size, timeout, limit, show_model, show_stats, profile, compact):
    model = Queens(board_size)

    def render_queens(solution):
        return render_queens_board(board_size, model.create_board(solution))

    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, compact=compact,
          render_solution=render_queens)
