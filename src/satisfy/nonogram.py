import collections

from .solver import ModelSolver, VarSelectionPolicy

__all__ = [
    'NonogramSolver',
    'pixmap_shape',
    'pixmap_to_nonogram',
]


VarInfo = collections.namedtuple('VarInfo', 'size start_value end_value')
    
class NonogramSolver(ModelSolver):
    def __init__(self, nonogram, **args):
        if args.get('var_selection_policy', None) is None:
            args['var_selection_policy'] = VarSelectionPolicy.MIN_BOUND
        super().__init__(**args)
        model = self._model
        rows = nonogram['rows']
        cols = nonogram['columns']
        num_rows = len(rows)
        num_cols = len(cols)
        var_infos = {}

        # add row vars and constraints:
        row_vars = {r: [] for r in range(num_rows)}
        for r, row in enumerate(rows):
            cur_vars = row_vars[r]
            if row:
                start = 0
                rem_size = sum(row) + len(row) - 1
                for k, size in enumerate(row):
                    offset = size + int(k != len(row) - 1)
                    end = num_cols - rem_size + 1
                    domain = list(range(start, end))
                    var = model.add_int_variable(name='r{}_{}'.format(r, k), domain=domain)
                    var_infos[var.name] = VarInfo(size=size, start_value=start, end_value=end + size)
                    # model.add_constraint(var + size <= num_cols)  # TODO diff SERVE???
                    start += offset
                    rem_size -= offset
                    if cur_vars:
                        prev_var = cur_vars[-1]
                        constraint = var > prev_var + var_infos[prev_var.name].size
                        model.add_constraint(constraint)
                    cur_vars.append(var)

        # add col vars and constraints:
        col_vars = {c: [] for c in range(num_cols)}
        for c, col in enumerate(cols):
            cur_vars = col_vars[c]
            if col:
                start = 0
                rem_size = sum(col) + len(col) - 1
                for k, size in enumerate(col):
                    offset = size + int(k != len(col) - 1)
                    end = num_rows - rem_size + 1
                    domain = list(range(start, end))
                    var = model.add_int_variable(name='c{}_{}'.format(c, k), domain=domain)
                    var_infos[var.name] = VarInfo(size=size, start_value=start, end_value=end + size)
                    # model.add_constraint(var + size <= num_rows)  # TODO diff SERVE???
                    start += offset
                    rem_size -= offset
                    if cur_vars:
                        prev_var = cur_vars[-1]
                        constraint = var > prev_var + var_infos[prev_var.name].size
                        model.add_constraint(constraint)
                    cur_vars.append(var)

        # add row<>col constraints:
        for r in range(num_rows):
            for c in range(num_cols):
                r_expr_list = []
                for var in row_vars[r]:
                    size = var_infos[var.name].size
                    var_info = var_infos[var.name]
                    if var_info.start_value <= c < var_info.end_value:
                        r_expr_list.append((var <= c) & (c < var + size))
                    # else:
                    #     print("r: {}: discard {} ({})".format(var.name, c, var_info), model.get_var_domain(var))
                c_expr_list = []
                for var in col_vars[c]:
                    size = var_infos[var.name].size
                    var_info = var_infos[var.name]
                    if var_info.start_value <= r < var_info.end_value:
                        c_expr_list.append((var <= r) & (r < var + size))
                    # else:
                    #     print("c: {}: discard {} ({})".format(var.name, r, var_info), model.get_var_domain(var))
                if r_expr_list or c_expr_list:
                    if r_expr_list:
                        r_expr = sum(r_expr_list)
                    else:
                        r_expr = 0
                    if c_expr_list:
                        c_expr = sum(c_expr_list)
                    else:
                        c_expr = 0
                    constraint = (sum(r_expr_list) == sum(c_expr_list))
                    model.add_constraint(constraint)

        # instance attributes:
        self._var_infos = var_infos
        self._shape = (num_rows, num_cols)
        self._row_vars = row_vars
        self._col_vars = col_vars

    @property
    def source(self):
        return self._source

    @property
    def expr(self):
        return self._expr

    def __iter__(self):
        model = self._model
        solver = self._solver
        num_rows, num_cols = self._shape
        var_infos = self._var_infos
        row_vars = self._row_vars
        for solution in solver.solve(model):
            pixmap = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
            for r, cur_vars in row_vars.items():
                for var in cur_vars:
                    start = solution[var.name] 
                    size = var_infos[var.name].size
                    for c in range(start, start + size):
                        pixmap[r][c] = 1
            yield pixmap


def pixmap_shape(pixmap):
    num_rows = len(pixmap)
    if pixmap:
        num_cols = max(len(row) for row in pixmap)
    else:
        num_cols = 0
    return num_rows, num_cols


def pixmap_to_nonogram(pixmap):
    num_rows, num_cols = pixmap_shape(pixmap)
    rows = []
    for r, pixmap_row in enumerate(pixmap):
        row = []
        count = 0
        for c, cell in enumerate(pixmap_row):
            if cell:
                count += 1
            else:
                if count:
                    row.append(count)
                count = 0
        if count:
            row.append(count)
        rows.append(row)
    cols = []
    for c in range(num_cols):
        col = []
        count = 0
        for r in range(num_rows):
            cell = pixmap[r][c]
            if cell:
                count += 1
            else:
                if count:
                    col.append(count)
                count = 0
        if count:
            col.append(count)
        cols.append(col)
    return {'rows': rows, 'columns': cols}
