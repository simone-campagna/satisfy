from .model import Model
from .solver import Solver, SelectVar, SelectValue

__all__ = [
    'GraphLabeling',
]


class GraphLabeling(Model):
    def __init__(self, graph, labels, **args):
        super().__init__(**args)
        self._graph = graph
        self._labels = {label_id: label for label_id, label in enumerate(labels)}
        label_ids = tuple(self._labels)
        variables = {}
        for node in self._graph.nodes():
            variables[node] = self.add_int_variable(domain=label_ids, name='n_{}'.format(len(variables)))
        for node0, node1 in self._graph.edges():
            self.add_constraint(variables[node0] != variables[node1])
        self._gl_variables = variables

    def solver(self, **kwargs):
        return Solver(
            select_var=kwargs.pop('select_var', SelectVar.in_order),
            select_value=kwargs.pop('select_value', SelectValue.min_value),
            **kwargs
        )

    def create_node_labels(self, solution):
        variables = self._gl_variables
        labels = self._labels
        node_labels = {}
        for node, variable in variables.items():
            node_labels[node] = labels[solution[variable.name]]
        return  node_labels
