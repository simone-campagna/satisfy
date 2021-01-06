import collections
import itertools

from .model import Model
from .solver import Solver, SelectVar, StaticVarSelector, SelectValue

__all__ = [
    'Nonogram',
    'pixmap_shape',
    'pixmap_to_nonogram',
]


VarInfo = collections.namedtuple('VarInfo', 'size start_value end_value')
    


# @SelectVar.__register__('nonogram')
# class NonogramVarSelector(StaticVarSelector):
#     def sort_var_names(self, unbound_var_names, model_info):
#         model = model_info.model
#         num_rows, num_cols = model.shape
#         row_vars = model.row_vars
#         col_vars = model.col_vars
#         var_infos = model.var_infos
#         var_value = {}
#         for r, r_vars in enumerate(row_vars):
#             for var in r_vars:
#                 var_value[var.name] = len(r_vars)
#         for c, c_vars in enumerate(col_vars):
#             for var in c_vars:
#                 var_value[var.name] = len(c_vars)
#         key_fn = lambda v: var_value[v]
#         all_vars = sorted(var_value, key=key_fn, reverse=True)
#         unbound_var_names.clear()
# 
#         def var_key_fn(v):
#             v_infos = var_infos[v]
#             return v_infos.end_value - v_infos.start_value
# 
#         for dummy, var_group in itertools.groupby(all_vars, key=key_fn):
#             var_group = list(var_group)
#             print("...", dummy, var_group)
#             var_group.sort(key=var_key_fn, reverse=True)
#             print("  .", dummy, var_group)
#             unbound_var_names.extend(var_group)
#         print(unbound_var_names)
            
        
class Nonogram(Model):
    def __init__(self, nonogram, **args):
        super().__init__(**args)
        rows = nonogram['rows']
        cols = nonogram['columns']
        num_rows = len(rows)
        num_cols = len(cols)
        var_infos = {}

        # add row vars and constraints:
        row_vars = [[] for r in range(num_rows)]
        for r, row in enumerate(rows):
            cur_vars = row_vars[r]
            if row:
                start = 0
                rem_size = sum(row) + len(row) - 1
                for k, size in enumerate(row):
                    offset = size + int(k != len(row) - 1)
                    end = num_cols - rem_size + 1
                    domain = list(range(start, end))
                    var = self.add_int_variable(name='r{}_{}'.format(r, k), domain=domain)
                    var_infos[var.name] = VarInfo(size=size, start_value=start, end_value=end + size)
                    # self.add_constraint(var + size <= num_cols)  # TODO diff SERVE???
                    start += offset
                    rem_size -= offset
                    if cur_vars:
                        prev_var = cur_vars[-1]
                        constraint = var > prev_var + var_infos[prev_var.name].size
                        self.add_constraint(constraint)
                    cur_vars.append(var)

        # add col vars and constraints:
        col_vars = [[] for c in range(num_cols)]
        for c, col in enumerate(cols):
            cur_vars = col_vars[c]
            if col:
                start = 0
                rem_size = sum(col) + len(col) - 1
                for k, size in enumerate(col):
                    offset = size + int(k != len(col) - 1)
                    end = num_rows - rem_size + 1
                    domain = list(range(start, end))
                    var = self.add_int_variable(name='c{}_{}'.format(c, k), domain=domain)
                    var_infos[var.name] = VarInfo(size=size, start_value=start, end_value=end + size)
                    # self.add_constraint(var + size <= num_rows)  # TODO diff SERVE???
                    start += offset
                    rem_size -= offset
                    if cur_vars:
                        prev_var = cur_vars[-1]
                        constraint = var > prev_var + var_infos[prev_var.name].size
                        self.add_constraint(constraint)
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
                    #     print("r: {}: discard {} ({})".format(var.name, c, var_info), self.get_var_domain(var))
                c_expr_list = []
                for var in col_vars[c]:
                    size = var_infos[var.name].size
                    var_info = var_infos[var.name]
                    if var_info.start_value <= r < var_info.end_value:
                        c_expr_list.append((var <= r) & (r < var + size))
                    # else:
                    #     print("c: {}: discard {} ({})".format(var.name, r, var_info), self.get_var_domain(var))
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
                    self.add_constraint(constraint)

        # instance attributes:
        self._var_infos = var_infos
        self._shape = (num_rows, num_cols)
        self._row_vars = row_vars
        self._col_vars = col_vars

    @property
    def shape(self):
        return self._shape

    @property
    def row_vars(self):
        return self._row_vars

    @property
    def col_vars(self):
        return self._col_vars

    @property
    def var_infos(self):
        return self._var_infos

    def solver(self, **kwargs):
        return Solver(
            select_var=kwargs.pop('select_var', SelectVar.min_boundmax),
            select_value=kwargs.pop('select_value', SelectValue.min_value),
            **kwargs
        )

    @property
    def source(self):
        return self._source

    @property
    def expr(self):
        return self._expr

    def create_pixmap(self, solution):
        num_rows, num_cols = self._shape
        row_vars = self._row_vars
        var_infos = self._var_infos
        pixmap = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        for r, cur_vars in enumerate(row_vars):
            for var in cur_vars:
                start = solution[var.name] 
                size = var_infos[var.name].size
                for c in range(start, start + size):
                    pixmap[r][c] = 1
        return pixmap


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
