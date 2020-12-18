import cProfile
import contextlib
import io
import pstats

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


def print_solve_stats(stats):
    if stats.count == 1:
        suffix = ''
    else:
        suffix = 's'
    if stats.interrupt:
        fmt = "Found {s.count} partial solution{suffix} in {s.elapsed:.3f} seconds [{s.interrupt} reached]"
    else:
        if stats.count == 1:
            fmt = "Found unique solution{suffix} in {s.elapsed:.3f} seconds"
        else:
            fmt = "Found all {s.count} solution{suffix} in {s.elapsed:.3f} seconds"
    print("\n" + fmt.format(suffix=suffix, s=stats))


def print_optimization_stats(stats, optimal=None):
    if optimal is None:
        optimal = stats.interrupt is None

    if optimal:
        kind = 'optimal'
    else:
        kind = 'sub-optimal'

    if stats.count == 1:
        suffix = ''
    else:
        suffix = 's'
    fmt = "Found {kind} solution in {s.elapsed:.3f} seconds after {s.count} solve iteration{suffix}"
    if stats.interrupt:
        fmt += " [{s.interrupt} reached]"
    print("\n" + fmt.format(suffix=suffix, kind=kind, s=stats))


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


def iter_solutions(model_solver, profile=False, show_stats=False, show_model=False, compact=False):
    if show_model:
        print_model(model_solver.model)

    with profiling(profile):
        if model_solver.has_objectives():
            result = model_solver.optimize()

            print("=== optimal_solution ===")
            if compact:
                print(result)
            else:
                yield result
            if show_stats:
                print_optimization_stats(model_solver.get_stats())
        else:
            num_solutions = 0
            for solution in model_solver:
                num_solutions += 1
                print("\n=== solution {} ===".format(num_solutions))
                if compact:
                    print(solution)
                else:
                    yield solution
            if show_stats:
                print_solve_stats(model_solver.get_stats())

