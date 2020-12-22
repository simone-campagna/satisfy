import argparse
import sys

from .cli_utils import solve
from ..sat import sat_compile

__all__ = [
    'sat_tool',
]


def sat_tool(input_file, timeout, limit, show_model, show_stats, profile, show_mode, output_file):
    source = input_file.read()
    model = sat_compile(source)
    
    solve(model, timeout=timeout, limit=limit,
          show_model=show_model, show_stats=show_stats, profile=profile, show_mode=show_mode, output_file=output_file)
          #render_solution=render_solution)
