__all__ = [
    'print_model',
    'print_solve_stats',
    'print_optimization_stats',
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
