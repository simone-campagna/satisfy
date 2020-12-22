import abc
import cProfile
import contextlib
import enum
import functools
import io
import json
import pstats
import sys

from ..solver import State

__all__ = [
    'ShowMode',
    'print_model',
    'print_solve_stats',
    'print_optimization_stats',
    'iter_solutions',
]


class Renderer(abc.ABC):
    def __init__(self, model, model_solver, render_solution=None, output_file=sys.stdout):
        self.model = model
        self.model_solver = model_solver
        if render_solution is None:
            render_solution = self._impl_render_solution
        self.render_solution = render_solution
        self.state = self.model_solver.state
        self.stats = self.state.stats
        self.output_file = output_file
        self.print = functools.partial(print, file=output_file)

    def _impl_render_solution(self, solution):
        return solution

    def show_model(self):
        pass

    def show_soluton(self, solution):
        pass

    def show_optimization_result(self):
        if self.model.has_objectives():
            optimization_result = self.model_solver.get_optimization_result()
            self._impl_show_optimization_result(optimization_result)

    def _impl_show_optimization_result(self, optimization_result):
        pass

    def show_stats(self, stats):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


class JsonRenderer(Renderer):
    def __init__(self, *args, **kwargs):
        self.data = {}
        super().__init__(*args, **kwargs)

    def show_model(self):
        model = self.model
        variables = {}
        for var_name, var_info in model.variables().items():
            variables[var_name] = var_info.domain
        constraints = []
        for constraint in model.constraints():
            constraints.append(str(constraint))
        self.data['model'] = {
            'variables': variables,
            'constraints': constraints,
        }
        if model.has_objectives():
            self.data['model']['objectives'] = [
                str(objective) for objective in model.objectives()
            ]

    def _impl_show_optimization_result(self, optimization_result):
        self.data['optimization_result'] = {
            'solution': optimization_result.solution,
            'is_optimal': optimization_result.is_optimal,
        }

    def show_stats(self):
        state = self.state
        stats = self.stats
        self.data['stats'] = {
            'solver_state': state.state.name,
            'solution_count': stats.count,
            'elapsed_seconds': stats.elapsed,
        }

    def show_solution(self, solution):
        self.data.setdefault('solutions', []).append(solution)

    def close(self):
        self.print(json.dumps(self.data, indent=4))


class TextRenderer(Renderer):
    def show_model(self):
        model = self.model
        self.print("=== model variables: ===")
        for var_index, (var_name, var_info) in enumerate(model.variables().items()):
            self.print(" {:4d}) {!r} domain: {}".format(var_index, var_name, var_info.domain))
        self.print()
        self.print("=== model constraints: ===")
        for c_index, constraint in enumerate(model.constraints()):
            self.print(" {:4d}) {}".format(c_index, constraint))
        self.print()
        if model.has_objectives():
            self.print("=== model objectives: ===")
            for c_index, objective in enumerate(model.objectives()):
                self.print(" {:4d}) {}".format(c_index, objective))
            self.print()

    def show_stats(self):
        state = self.state
        stats = self.stats
        if stats.count == 1:
            suffix = ''
        else:
            suffix = 's'
        if state.state is State.DONE:
            if stats.count == 1:
                fmt = "Found unique solution{suffix} in {stats.elapsed:.3f} seconds"
            else:
                fmt = "Found all {stats.count} solution{suffix} in {stats.elapsed:.3f} seconds"
        else:
            fmt = "Found {stats.count} partial solution{suffix} in {stats.elapsed:.3f} seconds [{state} reached]"
        state_name_tr = {
            State.INTERRUPT_TIMEOUT: 'timeout',
            State.INTERRUPT_LIMIT: 'limit',
        }
        state_name = state_name_tr.get(state.state, state.state.name)
        self.print(fmt.format(suffix=suffix, stats=stats, state=state_name))

    def _impl_render_solution(self, solution):
        dct = {}
        for key in sorted(solution):
            dct[key] = solution[key]
        return str(dct)

    def _impl_show_optimization_result(self, optimization_result):
        stats = self.stats
        if optimization_result.solution is not None:
            if optimization_result.is_optimal:
                sol_type = 'optimal'
            else:
                sol_type = 'sub-optimal'
            if stats.count == 1:
                suffix = ''
            else:
                suffix = 's'
            fmt = "Found {sol_type} solution in {s.elapsed:.3f} seconds after {s.count} solve iteration{suffix}"
            self.print(fmt.format(sol_type=sol_type, suffix=suffix, s=stats))
            if optimization_result.solution is not None:
                self.show_solution(optimization_result.solution)


class DefaultTextRenderer(TextRenderer):
    def show_solution(self, solution):
        render_solution = self.render_solution
        stats = self.stats
        self.print("=== solution {} ===".format(stats.count))
        self.print(render_solution(solution))
        self.print()


class CompactTextRenderer(TextRenderer):
    def show_solution(self, solution):
        render_solution = self.render_solution
        stats = self.stats
        prefix = '{:8d}) '.format(stats.count)
        for line in render_solution(solution).split('\n'):
            self.print(prefix + line)
            prefix = ' ' * len(prefix)


class QuietTextRenderer(CompactTextRenderer):
    def show_solution(self, solution):
        pass


class ShowMode(enum.Enum):
    DEFAULT = 0
    COMPACT = 1
    BRIEF = 2
    QUIET = 3
    JSON = 4


@contextlib.contextmanager
def profiling(enabled=True):
    if enabled:
        prof = cProfile.Profile()
        prof.enable()
    try:
        yield
    finally:
        if enabled:
            prof.disable()
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
            ps.print_stats()
            print(s.getvalue())


def solve(model, timeout, limit, profile=False, show_stats=False, show_model=False, show_mode=ShowMode.DEFAULT,
          output_file=sys.stdout, render_solution=None):
    renderer_class = DefaultTextRenderer
    if show_mode is ShowMode.QUIET:
        renderer_class = QuietTextRenderer
        render_solution = None
    elif show_mode is ShowMode.BRIEF:
        renderer_class = CompactTextRenderer
        render_solution = None
    elif show_mode is ShowMode.COMPACT:
        renderer_class = CompactTextRenderer
    elif show_mode is ShowMode.JSON:
        renderer_class = JsonRenderer
        render_solution = None
    else:
        renderer_class = DefaultTextRenderer

    kwargs = {}
    if limit is not None:
        kwargs['limit'] = limit
    if timeout is not None:
        kwargs['timeout'] = timeout
    with model.solve(**kwargs) as model_solver:
        with renderer_class(model, model_solver, render_solution, output_file=output_file) as renderer:
            if show_model:
                renderer.show_model()

            with profiling(profile):
                for solution in model_solver:
                    renderer.show_solution(solution)
            renderer.show_optimization_result()
            if show_stats:
                renderer.show_stats()
