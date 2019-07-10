import collections
import itertools

from ..solver import ModelSolver, SelectVar, SelectValue

__all__ = [
    'QueensSolver',
]


class QueensSolver(ModelSolver):
    def __init__(self, board_size, **args):
        if args.get('select_var', None) is None:
            args['select_var'] = SelectVar.group_prio
        if args.get('select_value', None) is None:
            args['select_value'] = SelectValue.max_value
        super().__init__(**args)
        model = self._model
        queens = [model.add_int_variable(domain=range(board_size), name="q_{}".format(r)) for r in range(board_size)]

        model.add_all_different_constraint(queens)

        for r in range(board_size):
            diag_ul_br = []
            diag_ur_bl = []
            for c in range(board_size):
                var = model.add_int_variable(domain=range(2 * board_size + 1), name="diag_ul_br_{}_{}".format(r, c))
                diag_ul_br.append(var)
                model.add_constraint(var == queens[c] + c)
                var = model.add_int_variable(domain=range(-board_size, board_size + 1), name="diag_ur_bl_{}_{}".format(r, c))
                diag_ur_bl.append(var)
                model.add_constraint(var == queens[c] - c)
            model.add_all_different_constraint(diag_ul_br)
            model.add_all_different_constraint(diag_ur_bl)

        self._board_size = board_size
        self._queens = queens

    def __iter__(self):
        board_size = self._board_size
        model = self._model
        solver = self._solver
        queens = self._queens
        for solution in solver.solve(model):
            board = [[0 for _ in range(board_size)] for _ in range(board_size)]
            for r, queen in enumerate(queens):
                board[r][solution[queen.name]] = 1
            yield tuple(board)
