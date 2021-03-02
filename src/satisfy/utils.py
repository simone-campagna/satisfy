import collections
import inspect
import time

__all__ = [
    'INFINITY',
    'InfiniTY',
    'UNDEFINED',
    'Undefined',
    'Timer',
    'Stats',
    'safe_call',
]


class _Singleton(object):
    __instance__ = None

    def __new__(cls):
        if cls.__instance__ is None:
            instance = super().__new__(cls)
            cls.__instance__ = instance
        return cls.__instance__

    def name(self):
        return type(self).__name__

    def __repr__(self):
        return "{}()".format(self.name())

    def __str__(self):
        return self.name


class Infinity(_Singleton):
    __instance__ = None


class Undefined(_Singleton):
    __instance__ = None

INFINITY = Infinity()
UNDEFINED = Undefined()


class Stats:
    def __init__(self, count=0, elapsed=0.0):
        self.count = count
        self.elapsed = elapsed

    def __repr__(self):
        return "{}(count={!r}, elapsed={!r})".format(type(self).__name__, self.count, self.elapsed)


class Timer(object):
    def __init__(self):
        self._t_start = None
        self.stats = Stats(count=0, elapsed=0.0)

    def running(self):
        return self._t_start is not None

    def start(self):
        if self.running():
            raise RuntimeError("already started")
        self._t_start = time.time()

    def stop(self, *, count=1):
        if not self.running():
            raise RuntimeError("not started")
        t_elapsed = time.time() - self._t_start
        self.stats.count += count
        self.stats.elapsed += t_elapsed
        self._t_start = None
        return t_elapsed

    def count(self):
        return self._count

    def abort(self):
        if self.running():
            self.stop(count=0)

    def elapsed(self):
        if self.running():
            return self.stats.elapsed + (time.time() - self._t_start)
        else:
            return self.stats.elapsed


def safe_call(function, **kwargs):
    parameters = inspect.signature(function).parameters
    for p in parameters.values():
        if p.kind == p.VAR_KEYWORD:
            return function(**kwargs)
    safe_kwargs = {}
    for key, value in kwargs.items():
        if key in parameters:
            safe_kwargs[key] = value
    return function(**safe_kwargs)
