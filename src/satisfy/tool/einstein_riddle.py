import collections
import itertools

from ..model import Model
from ..solver import ModelSolver, VarSelectionPolicy

__all__ = [
    'EinsteinRiddleSolver',
]


class EinsteinRiddleSolver(ModelSolver):
    __statements__ = (
        "The English man lives in the red house.",
        "The Swede has a dog.",
        "The Dane drinks tea.",
        "The green house is immediately to the left of the white house.",
        "They drink coffee in the green house.",
        "The man who smokes Pall Mall has birds.",
        "In the yellow house they smoke Dunhill.",
        "In the middle house they drink milk.",
        "The Norwegian lives in the first house.",
        "The man who smokes Blend lives in the house next to the house with cats.",
        "In a house next to the house where they have a horse, they smoke Dunhill.",
        "The man who smokes Blue Master drinks beer.",
        "The German smokes Prince.",
        "The Norwegian lives next to the blue house.",
        "They drink water in a house next to the house where they smoke Blend.",
    )

    def __init__(self, **args):
        if args.get('var_selection_policy', None) is None:
            args['var_selection_policy'] = VarSelectionPolicy.ORDERED
        super().__init__(**args)
        model = self._model

        def mkname(*tokens):
            return "_".join(token.replace(' ', '_') for token in tokens)

        indices = tuple(range(1, 6))
        houses = ['house{}'.format(i) for i in indices]

        l_owner = ['English', 'Swede', 'Dane', 'Norwegian', 'German']
        d_owner = {owner: model.add_int_variable(domain=indices, name=mkname('owner', owner)) for owner in l_owner}

        l_color = ['red', 'green', 'white', 'yellow', 'blue']
        d_color = {color: model.add_int_variable(domain=indices, name=mkname('color', color)) for color in l_color}

        l_drink = ['water', 'coffee', 'tea', 'milk', 'beer']
        d_drink = {drink: model.add_int_variable(domain=indices, name=mkname('drink', drink)) for drink in l_drink}

        l_smoke = ['Pall Mall', 'Prince', 'Dunhill', 'Blend', 'Blue Master']
        d_smoke = {smoke: model.add_int_variable(domain=indices, name=mkname('smoke', smoke)) for smoke in l_smoke}

        l_animal = ['cats', 'a dog', 'birds', 'a horse', 'a zebra']
        d_animal = {animal: model.add_int_variable(domain=indices, name=mkname('animal', animal)) for animal in l_animal}

        variables = [
            ('owner',  d_owner),
            ('color',  d_color),
            ('drink',  d_drink),
            ('smoke',  d_smoke),
            ('animal', d_animal),
        ]

        model.add_all_different_constraint(d_owner.values())
        model.add_all_different_constraint(d_color.values())
        model.add_all_different_constraint(d_drink.values())
        model.add_all_different_constraint(d_smoke.values())
        model.add_all_different_constraint(d_animal.values())

        ################################################################################

        #  1. The English man lives in the red house.
        model.add_constraint(d_owner['English'] == d_color['red'])

        #  2. The Swede has a dog.
        model.add_constraint(d_owner['Swede'] == d_animal['a dog'])

        #  3. The Dane drinks tea.
        model.add_constraint(d_owner['Dane'] == d_drink['tea'])

        #  4. The green house is immediately to the left of the white house.
        model.add_constraint(d_color['green'] == (d_color['white'] - 1))

        #  5. They drink coffee in the green house.
        model.add_constraint(d_color['green'] == d_drink['coffee'])

        #  6. The man who smokes Pall Mall has birds.
        model.add_constraint(d_smoke['Pall Mall'] == d_animal['birds'])

        #  7. In the yellow house they smoke Dunhill.
        model.add_constraint(d_color['yellow'] == d_smoke['Dunhill'])

        #  8. In the middle house they drink milk.
        model.add_constraint(d_drink['milk'] == 3)

        #  9. The Norwegian lives in the first house.
        model.add_constraint(d_owner['Norwegian'] == 1)

        # 10. The man who smokes Blend lives in the house next to the house with cats.
        pm_a = model.add_int_variable(domain=[-1, 1], name='Blend_cats')
        model.add_constraint(d_smoke['Blend'] - d_animal['cats'] == pm_a)

        # 11. In a house next to the house where they have a horse, they smoke Dunhill.
        pm_b = model.add_int_variable(domain=[-1, 1], name='Dunhill_horse')
        model.add_constraint(d_smoke['Dunhill'] - d_animal['a horse'] == pm_b)

        # 12. The man who smokes Blue Master drinks beer.
        model.add_constraint(d_smoke['Blue Master'] == d_drink['beer'])

        # 13. The German smokes Prince.
        model.add_constraint(d_owner['German'] == d_smoke['Prince'])

        # 14. The Norwegian lives next to the blue house.
        pm_c = model.add_int_variable(domain=[-1, 1], name='Norwegian_blue')
        model.add_constraint(d_owner['Norwegian'] - d_color['blue'] == pm_c)

        # 15. They drink water in a house next to the house where they smoke Blend.
        pm_d = model.add_int_variable(domain=[-1, 1], name='water_Blend')
        model.add_constraint(d_drink['water'] - d_smoke['Blend'] == pm_d)

        ################################################################################
        self._variables = variables
        self._indices = indices

    @classmethod
    def statements(cls):
        return cls.__statements__

    @classmethod
    def riddle(cls):
        lst = ["{:>2d}. {}".format(i, statement) for i, statement in enumerate(cls.statements())]
        lst.append("")
        lst.append("Who owns the zebra?")
        lst.append("(see http://rosettacode.org/wiki/Zebra_puzzle)")
        return '\n'.join(lst)

    def __iter__(self):
        solver = self._solver
        model = self._model
        indices = self._indices
        variables = self._variables
        for solution in solver.solve(model):
            z_sol = {}
            for index in indices:
                z_sol[index] = {}
            for var_name, var_d in variables:
                for var_value, var in var_d.items():
                    house_index = solution[var.name]
                    z_sol[house_index][var_name] = var_value
            yield z_sol
