import collections
import itertools

from .model import Model
from .solver import Solver, SelectVar, StaticVarSelector, SelectValue, Algorithm

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
            
        
class Stripe:
    def __init__(self, start_begin, start_end, size):
        self.start_begin = start_begin
        self.start_end = start_end
        self.size = size
        assert self.start_end > self.start_begin

    def __contains__(self, pos):
        return self.start_begin <= pos < self.start_end + self.size

    def known_points(self):
        yield from range(self.start_end - 1, self.start_begin + self.size)

    def domain_len(self):
        return self.start_end - self.start_begin

    def domain(self):
        return list(range(self.start_begin, self.start_end))

    def must_contain(self, pos):
        # orig = self.start_begin, self.start_end, self.domain()
        orig = (self.start_begin, self.start_end)
        self.start_begin = max(pos - self.size, self.start_begin)
        self.start_end = min(pos + 1, self.start_end)
        assert self.start_end > self.start_begin
        return (self.start_begin, self.start_end) != orig
        # new = self.start_begin, self.start_end, self.domain()
        # if orig[-1] != new[-1]:
        #     print("<<<", orig)
        #     print("must contain:", pos)
        #     print(">>>", new)

    def __repr__(self):
        return "{}(start_begin={!r}, start_end={!r}, size={!r})".format(
            type(self).__name__, self.start_begin, self.start_end, self.size)


def _dbg_print_matrix(*matrices):
        dct = {0: '_', 1: '#'}
        print('-' * 80)
        for matrix in matrices:
            for row in matrix:
                print(''.join(dct[r] for r in row))
            print('-' * 80)
        input("---")


class Nonogram(Model):
    def __init__(self, nonogram, **args):
        super().__init__(**args)
        rows = nonogram['rows']
        cols = nonogram['columns']
        num_rows = len(rows)
        num_cols = len(cols)
        var_infos = {}

        # matrix of known points:
        r_matrix = []
        c_matrix = []
        for _ in range(num_rows):
            r_matrix.append([0] * num_cols)
            c_matrix.append([0] * num_cols)

        # inspect rows:
        row_stripes = []
        for r, row in enumerate(rows):
            stripes = []
            row_stripes.append(stripes)
            if row:
                start_begin = 0
                rem_size = sum(row) + len(row) - 1
                for k, size in enumerate(row):
                    offset = size + int(k != len(row) - 1)
                    start_end = num_cols - rem_size + 1
                    stripe = Stripe(start_begin, start_end, size)
                    stripes.append(stripe)
                    for c in stripe.known_points():
                        r_matrix[r][c] = 1
                    start_begin += offset
                    rem_size -= offset
        # _dbg_print_matrix(r_matrix)

        # inspect cols:
        col_stripes = []
        for c, col in enumerate(cols):
            stripes = []
            col_stripes.append(stripes)
            if col:
                start_begin = 0
                rem_size = sum(col) + len(col) - 1
                for k, size in enumerate(col):
                    offset = size + int(k != len(col) - 1)
                    start_end = num_rows - rem_size + 1
                    stripe = Stripe(start_begin, start_end, size)
                    stripes.append(stripe)
                    for r in stripe.known_points():
                        c_matrix[r][c] = 1
                    start_begin += offset
                    rem_size -= offset
        # _dbg_print_matrix(c_matrix)

        repeat = True
        while repeat:
            repeat = False
            # reduce stripe domains:
            for r in range(num_rows):
                for c in range(num_cols):
                    if c_matrix[r][c]:
                        r_stripes = []
                        for stripe in row_stripes[r]:
                            if c in stripe:
                                r_stripes.append(stripe)
                        if len(r_stripes) == 1:
                            stripe = r_stripes[0]
                            if stripe.must_contain(c):
                                repeat = True
                    if r_matrix[r][c]:
                        c_stripes = []
                        for stripe in col_stripes[c]:
                            if r in stripe:
                                c_stripes.append(stripe)
                        if len(c_stripes) == 1:
                            stripe = c_stripes[0]
                            if stripe.must_contain(r):
                                repeat = True
            for r, stripes in enumerate(row_stripes):
                for stripe in stripes:
                    if stripe.domain_len() == 1:
                        for c in range(stripe.start_begin, stripe.start_begin + stripe.size):
                            c_matrix[r][c] = 1
            for c, stripes in enumerate(col_stripes):
                for stripe in stripes:
                    if stripe.domain_len() == 1:
                        for r in range(stripe.start_begin, stripe.start_begin + stripe.size):
                            r_matrix[r][c] = 1

            # input("repeat {}...".format(repeat))
            # _dbg_print_matrix(r_matrix, c_matrix)

        # add row vars and constraints:
        row_vars = [[] for r in range(num_rows)]
        for r, stripes in enumerate(row_stripes):
            cur_vars = row_vars[r]
            for k, stripe in enumerate(stripes):
                domain = stripe.domain()
                var = self.add_int_variable(name='r{}_{}'.format(r, k), domain=domain)
                var_infos[var.name] = VarInfo(
                    size=stripe.size,
                    start_value=stripe.start_begin,
                    end_value=stripe.start_end + stripe.size)
                if cur_vars:
                    prev_var = cur_vars[-1]
                    constraint = var > prev_var + var_infos[prev_var.name].size
                    self.add_constraint(constraint)
                cur_vars.append(var)
            if cur_vars:
                self.add_all_different_constraint(cur_vars)

        # add col vars and constraints:
        col_vars = [[] for r in range(num_cols)]
        for c, stripes in enumerate(col_stripes):
            cur_vars = col_vars[c]
            for k, stripe in enumerate(stripes):
                domain = stripe.domain()
                var = self.add_int_variable(name='c{}_{}'.format(k, c), domain=domain)
                var_infos[var.name] = VarInfo(
                    size=stripe.size,
                    start_value=stripe.start_begin,
                    end_value=stripe.start_end + stripe.size)
                if cur_vars:
                    prev_var = cur_vars[-1]
                    constraint = var > prev_var + var_infos[prev_var.name].size
                    self.add_constraint(constraint)
                cur_vars.append(var)
            if cur_vars:
                self.add_all_different_constraint(cur_vars)

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
            select_var=kwargs.pop('select_var', SelectVar.min_domain),
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
