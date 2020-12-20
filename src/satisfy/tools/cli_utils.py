import cProfile
import contextlib
import io
import pstats

from ..solver import State

__all__ = [
    'print_model',
    'print_solve_stats',
    'print_optimization_stats',
    'iter_solutions',
]


def print_model(model):
    print("\n=== model variables: ===")
    for var_index, (var_name, var_info) in enumerate(model.variables().items()):
        print(" {:4d}) {!r} domain: {}".format(var_index, var_name, var_info.domain))
    print("\n=== model constraints: ===")
    for c_index, constraint in enumerate(model.constraints()):
        print(" {:4d}) {}".format(c_index, constraint))


def print_solve_stats(state):
    stats = state.stats
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
    print("\n" + fmt.format(suffix=suffix, stats=stats, state=state_name))


def render_solution(solution):
    return ' '.join('{}={!r}'.format(key, solution[key]) for key in sorted(solution))


def print_solution(solution, stats, renderer=render_solution, compact=False):
    if compact:
        print("{:8d}) {}".format(stats.count, renderer(solution)))
    else:
        print("=== solution {} ===".format(stats.count))
        print(renderer(solution))
        print()


def print_optimization_result(optimization_result, stats, renderer=render_solution, compact=False):
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
        print(fmt.format(sol_type=sol_type, suffix=suffix, s=stats))
        if renderer and optimization_result.solution is not None:
            print_solution(optimization_result.solution, stats, renderer=renderer, compact=compact)

    
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


def solve(model, timeout, limit, profile=False, show_stats=False, show_model=False, compact=False,
          render_solution=render_solution, print_optimization_result=print_optimization_result):
    if show_model:
        print_model(model)

    with model.solve(timeout=timeout, limit=limit) as model_solver:
        state = model_solver.state
        stats = state.stats
        with profiling(profile):
            for solution in model_solver:
                if render_solution:
                    print_solution(solution, stats, render_solution, compact=compact)
        if model.has_objectives() and print_optimization_result:
            optimization_result = model_solver.get_optimization_result()
            print_optimization_result(optimization_result, stats, render_solution, compact=compact)
        if show_stats:
            print_solve_stats(state)
