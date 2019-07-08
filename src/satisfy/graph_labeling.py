from .model import Model
from .solver_legacy import ModelSolver, VarSelectionPolicy

__all__ = [
    'GraphLabelingSolver',
]


class GraphLabelingSolver(ModelSolver):
    def __init__(self, graph, labels, **args):
        if args.get('var_selection_policy', None) is None:
            args['var_selection_policy'] = VarSelectionPolicy.ORDERED
        super().__init__(**args)
        self._graph = graph
        self._labels = {label_id: label for label_id, label in enumerate(labels)}
        label_ids = tuple(self._labels)
        model = self._model
        variables = {}
        for node in self._graph.nodes():
            variables[node] = model.add_int_variable(domain=label_ids, name='n_{}'.format(len(variables)))
        for node0, node1 in self._graph.edges():
            model.add_constraint(variables[node0] != variables[node1])
        self._variables = variables

    def __iter__(self):
        variables = self._variables
        labels = self._labels
        for solution in self._solver.solve(self._model):
            node_labels = {}
            for node, variable in variables.items():
                node_labels[node] = labels[solution[variable.name]]
            yield node_labels
