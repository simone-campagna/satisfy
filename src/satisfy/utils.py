import collections
import time

__all__ = [
    'INFINITY',
    'InfiniTY',
    'UNDEFINED',
    'Undefined',
    'Timer',
    'Stats',
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

Stats = collections.namedtuple(
    "Stats", "count elapsed")


SolveStats = collections.namedtuple(
    "SolveStats", "count elapsed interrupt")


class Timer(object):
    def __init__(self):
        self._t_elapsed = 0.0
        self._t_start = None
        self._count = 0

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
        self._t_elapsed += t_elapsed
        self._t_start = None
        self._count += count
        return t_elapsed

    def count(self):
        return self._count

    def abort(self):
        if self.running():
            self.stop(count=0)

    def elapsed(self):
        if self.running():
            return self._t_elapsed + (time.time() - self._t_start)
        else:
            return self._t_elapsed

    def stats(self):
        return Stats(
            count=self._count,
            elapsed=self._t_elapsed)
