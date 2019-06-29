import json

from ..nonogram import NonogramSolver, pixmap_shape, pixmap_to_nonogram

from .demo_utils import (
    print_model,
    print_solve_stats,
)

__all__ = [
    'nonogram',
    'default_nonogram_data',
]


__all__ = [
    'nonogram',
]


DEFAULT_NONOGRAM_SOURCE = """
{
    "rows": [[], [4], [6], [2, 2], [2, 2], [6], [4], [2], [2], [2], [], []],
    "columns": [[], [9], [9], [2, 2], [2, 2], [4], [4], []]
}
"""

def default_nonogram_source():
    return DEFAULT_NONOGRAM_SOURCE


def print_nonogram(nonogram):
    print(json.dumps(nonogram))


def print_nonogram_pixmap(pixmap):
    print(pixmap_to_image(pixmap))


ZEROES = frozenset({' ', '_'})
ZERO = ' '
ONE = '#'

def image_to_pixmap(image, zeroes=ZEROES):
    lines = image.split('\n')
    num_rows = len(lines)
    num_cols = max(len(line) for line in lines)
    pixmap = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    for r, line in enumerate(lines):
        for c, cell in enumerate(line):
            if cell not in zeroes:
                pixmap[r][c] = 1
    return pixmap


def image_to_nonogram(image, zeroes=ZEROES):
    return pixmap_to_nonogram(image_to_pixmap(image, zeroes=zeroes))


def pixmap_to_image(pixmap, zero=ZERO, one=ONE):
    num_rows, num_cols = pixmap_shape(pixmap)
    image = [[zero for _ in range(num_cols)] for _ in range(num_rows)]
    for r, row in enumerate(pixmap):
        for c, cell in enumerate(row):
            if cell:
                image[r][c] = one
    return '\n'.join(''.join(line) for line in image)


def nonogram(input_file, input_image, timeout, limit, show_model):
    if input_file is None:
        if input_image:
            nonogram = image_to_nonogram(input_image.read())
        else:
            source = default_nonogram_source()
            print("""\
No input file - using default nonogram:
{example}
""".format(example=source))
            nonogram = json.loads(source)
    else:
        nonogram = json.load(input_file)

    num_solutions = 0
    nonogram_solver = NonogramSolver(nonogram, timeout=timeout, limit=limit)
    if show_model:
        print_model(nonogram_solver.model)
    for pixmap in nonogram_solver:
        num_solutions += 1
        print("\n=== solution {} ===".format(num_solutions))
        print_nonogram_pixmap(pixmap)
    print_solve_stats(nonogram_solver.get_stats())
