import collections
import itertools

from .model import Model
from .solver import Solver, SelectVar, SelectValue

__all__ = [
    'SudokuSolver',
]


Cell = collections.namedtuple(
    "Cell",
    "row_index col_index block_index domain variables")


class Sudoku(Model):
    def __init__(self, schema, **args):
        super().__init__(**args)
        block_size = 3
        size = block_size ** 2
        indices = list(range(size))
        values = set(range(1, size + 1))
        null_values = {0, None}
        accepted_values = values.union(null_values)
        schema = tuple(tuple(row) for row in schema)
        if len(schema) != size:
            raise ValueError("wrong row number {}".format(len(schema)))
        matrix = [[None for _ in range(size)] for _ in range(size)]
        row_neighbors = [[] for _ in range(size)]
        col_neighbors = [[] for _ in range(size)]
        block_neighbors = [[] for _ in range(size)]
        variables = []

        for row_index, row in enumerate(schema):
            block_offset = (row_index // block_size) * block_size
            for col_index, value in enumerate(row):
                if value not in accepted_values:
                    raise ValueError("bad value {!r}".format(value))
                block_index = block_offset + (col_index // 3)
                cell = Cell(row_index, col_index, block_index, domain=set(values), variables=[])
                matrix[row_index][col_index] = cell
                row_neighbors[row_index].append(cell)
                col_neighbors[col_index].append(cell)
                block_neighbors[block_index].append(cell)

        for matrix_row, row in zip(matrix, schema):
            for cell, value in zip(matrix_row, row):
                if value in accepted_values:
                    for c1 in itertools.chain(row_neighbors[cell.row_index],
                                              col_neighbors[cell.col_index],
                                              block_neighbors[cell.block_index]):
                        c1.domain.discard(value)

        row_variables = [[] for _ in range(size)]
        col_variables = [[] for _ in range(size)]
        block_variables = [[] for _ in range(size)]
        for matrix_row in matrix:
            for cell in matrix_row:
                value = schema[cell.row_index][cell.col_index]
                if value in null_values:
                    variable = self.add_int_variable(domain=cell.domain, name="c_{}_{}".format(cell.row_index, cell.col_index))
                    cell.variables.append(variable)
                    row_variables[cell.row_index].append(variable)
                    col_variables[cell.col_index].append(variable)
                    block_variables[cell.block_index].append(variable)

        for idx, variables in enumerate(row_variables):
            if len(variables) > 1:
                # print("r[{}]: {}".format(idx, ', '.join(v.name for v in variables)))
                self.add_all_different_constraint(variables)
        for idx, variables in enumerate(col_variables):
            if len(variables) > 1:
                # print("c[{}]: {}".format(idx, ', '.join(v.name for v in variables)))
                self.add_all_different_constraint(variables)
        for idx, variables in enumerate(block_variables):
            if len(variables) > 1:
                # print("b[{}]: {}".format(idx, ', '.join(v.name for v in variables)))
                self.add_all_different_constraint(variables)

        self._matrix = matrix
        self._schema = schema

    def solver(self, *, limit=None, **kwargs):
        if limit is None:
            limit = 1
        return Solver(
            limit=limit,
            select_var=kwargs.pop('select_var', SelectVar.min_bound),
            select_value=kwargs.pop('select_value', SelectValue.min_value),
            **kwargs
        )

    def create_schema(self, solution):
        matrix = self._matrix
        schema = []
        for matrix_row, row in zip(matrix, self._schema):
            schema_row = []
            for cell, value in zip(matrix_row, row):
                if cell.variables:
                    value = solution[cell.variables[0].name]
                schema_row.append(value)
            schema.append(tuple(schema_row))
        return schema
