import itertools

import networkx as nx
import termcolor

from ..graph_labeling import GraphLabelingSolver

__all__ = [
    'AsciiMapColoringSolver',
]


class AsciiMapColoringSolver(GraphLabelingSolver):
    def __init__(self, ascii_map, colors=('red', 'green', 'blue', 'yellow'), **args):
        if isinstance(ascii_map, str):
            ascii_map = ascii_map.split('\n')

        OFFSETS = [
            (-1, -1),
            (-1,  0),
            (-1, +1),
            ( 0, -1),
            ( 0, +1),
            (+1, -1),
            (+1,  0),
            (+1, +1),
        ]

        def search_and_cancel_group(rows, value, pos, pos_limits):
            r, c = pos
            rstop, cstop = pos_limits
            positions = set([pos])
            group = set()
            while positions:
                new_positions = set()
                for r, c in positions:
                    if rows[r][c] == value:
                        rows[r][c] = "."
                        group.add((r, c))
                        for roffset, coffset in OFFSETS:
                            pos_offset = r + roffset, c + coffset
                            ro, co = pos_offset
                            if 0 <= ro < rstop and 0 <= co < cstop:
                                if pos_offset not in group:
                                    new_positions.add(pos_offset)
                positions = new_positions
            return group

        rows = [list(row) for row in ascii_map]
        self._ascii_map = tuple(tuple(row) for row in ascii_map)
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows)
        pos_limits = (num_rows, num_cols)
        rows = [(row + ([' '] * (num_cols - len(row)))) for row in rows]
        
        r_start = 0
        groups = []
        graph = nx.Graph()
        while True:
            for r, c in itertools.product(range(r_start, num_rows), range(num_cols)):
                value = rows[r][c]
                if value not in {' ', '.'}:
                    group = search_and_cancel_group(rows, value, (r, c), pos_limits)
                    group_id = len(groups)
                    groups.append(group)
                    graph.add_node(group_id)
                    r_start = r
                    break
            else:
                break


        for gid0, group0 in enumerate(groups[:-1]):
            graph.add_node(gid0)
            g0_neighbors = set()
            for r, c in group0:
                for roffset, coffset in OFFSETS:
                    g0_neighbors.add((r + roffset, c + coffset))
            for rel_gid1, group1 in enumerate(groups[gid0 + 1:]):
                if group1.intersection(g0_neighbors):
                    gid1 = gid0 + 1 + rel_gid1
                    graph.add_edge(gid0, gid1)

        self._groups = groups
        if args.get('limit', None) is None:
            args['limit'] = 1
        super().__init__(graph=graph, labels=colors, **args)

    @property
    def ascii_map(self):
        return self._ascii_map

    def __iter__(self):
        orig_ascii_map = self._ascii_map
        groups = self._groups
        colors = self._labels
        glabels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for solution in super().__iter__():
            ascii_map = [[' ' for _ in row] for row in orig_ascii_map]
            for gid, positions in enumerate(groups):
                color = solution[gid]
                value = termcolor.colored(str(glabels[gid % len(glabels)]), color)
                for r, c in positions:
                    ascii_map[r][c] = value
            yield ascii_map

