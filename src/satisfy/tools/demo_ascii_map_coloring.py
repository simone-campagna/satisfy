from .ascii_map_coloring import AsciiMapColoring

from .cli_utils import (
    solve,
)

__all__ = [
    'ascii_map_coloring',
    'default_ascii_map_coloring_source',
]


DEFAULT_ASCII_MAP_COLORING_SOURCE = """\
           0001111111144
        000011111111114433     33
      000111111111111111333333333333332222 
    22222111111111111111333333333333333322222
   2222333311111111111111333333333333333222222
   22223333666111111111111333333333333332222222
  3333333336666611111111111FFFFFHHHHHH3222222222
 333333333366666666611111FFFFFFHHHHHHHHZZZZZZZZZZ 
33333333333666666666FFFFFFFFFFFHHHHHHHHZZZZZZZZZZYYY  XXXXX
4444446666666666666FFFFFFFFFFFHHHHHHHHHZZZZZZZZZYYYYXXXXXXX
 444446666666BBCDDDEEFFFFFFFFFHHHHHHHHHZZZZZZZZYYYYYYYXXXX
  5556666666ABBBCDDEEEEEEEEEEEGHHHHHHHHIZZZZZZZZYYYYXXXX
   6667888AAABBBCDDDEEEEEEEEEGGGHHHHHHHIIZZZZZZYYYYXXX 
    7779999AABBBBCDDEEEEEGGGGGGGHHHIIIIIIZZZZVVVYYXX
        999AABBBBCDDEEGGGGGGGGIIIIIIIIIII0UUUUVVVVV
                      GGGGGGGIIIIIIIIII11UUUUUVVVV
                     KKKLLLLJJIIIIIIJJJUUUUUUUUVV
                      KKLLLJJJJJJJJJJJJUUUUUUUUV
                       KLLJJJJJJJJJJJJJUUUUUUUU
                       MMMMMJJJJJJJJJJJUUUUUUUU
                       MMMMMMMJJJJJJJJTTTTTUUUU
                       MMMMMMMMMMJJJJJTTTTTSSSS
                       MMMMMMMMMMMJTTTTTTSSSSSS
                      MMMMMMMMMMMTTTTTSSSSSSSS
                       NNNNNNNNNNNQQQQQSSSSS 
                        NNNNNNQQQQQQQOOOSSS
                         NNNNNQQQOOOOOOOORR
                         NNNNNOOOOOOOOPPOR
                          NNNNOOOOOOOOOO
                           OOOOOOOOOOO 
                             OOOOO    
"""


def default_ascii_map_coloring_source():
    return DEFAULT_ASCII_MAP_COLORING_SOURCE


def ascii_map_coloring(input_file, colors, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
    if input_file is None:
        source = default_ascii_map_coloring_source()
        print("""\
No input file - using default data:
{example}
""".format(example=source))
        data = source
    else:
        with open(input_file, "r") as fh:
            data = fh.read()

    model = AsciiMapColoring(data, colors)

    def render_map(solution):
        return model.create_ascii_map(solution)

    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode,
          output_file=output_file, render_solution=render_map)
