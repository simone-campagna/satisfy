import collections
import itertools

from ..model import Model
from ..solver import Solver, SelectVar, SelectValue

__all__ = [
    'Queens',
]


class Queens(Model):
    def __init__(self, board_size, **args):
        super().__init__(**args)
        queens = [self.add_int_variable(domain=range(board_size), name="q_{}".format(r)) for r in range(board_size)]

        self.add_all_different_constraint(queens)

        for r in range(board_size):
            diag_ul_br = []
            diag_ur_bl = []
            for c in range(board_size):
                var = self.add_int_variable(domain=range(2 * board_size + 1), name="diag_ul_br_{}_{}".format(r, c))
                diag_ul_br.append(var)
                self.add_constraint(var == queens[c] + c)
                var = self.add_int_variable(domain=range(-board_size, board_size + 1), name="diag_ur_bl_{}_{}".format(r, c))
                diag_ur_bl.append(var)
                self.add_constraint(var == queens[c] - c)
            self.add_all_different_constraint(diag_ul_br)
            self.add_all_different_constraint(diag_ur_bl)

        self._board_size = board_size
        self._queens = queens

    def solver(self, **kwargs):
        return Solver(
            select_var=kwargs.pop('select_var', SelectVar.group_prio),
            select_value=kwargs.pop('select_value', SelectValue.max_value),
            **kwargs
        )

    def create_board(self, solution):
        board_size = self._board_size
        queens = self._queens
        board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        for r, queen in enumerate(queens):
            board[r][solution[queen.name]] = 1
        return board
