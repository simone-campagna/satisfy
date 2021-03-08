import json

from ..nonogram import Nonogram, pixmap_shape, pixmap_to_nonogram

from .cli_utils import (
    solve,
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
    "rows": [
        [3], [4, 2], [6, 6], [6, 2, 1], [1, 4, 2, 1], [6, 3, 2],
        [6, 7], [6, 8], [1, 10], [1, 10], [1, 10], [1, 1, 4, 4],
        [3, 4, 4], [4, 4], [4, 4], [], []
    ],
    "columns": [
        [1], [11], [3, 3, 1], [7, 2], [7], [15], [1, 5, 7],
        [2, 8], [14], [9], [1, 6], [1, 9], [1, 9], [1, 10], [12]
    ]
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


def nonogram(input_file, input_image, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
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

    model = Nonogram(nonogram)

    def render_nonogram(solution):
        pixmap = model.create_pixmap(solution)
        return pixmap_to_image(pixmap)
        
    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode,
          output_file=output_file, render_solution=render_nonogram)
