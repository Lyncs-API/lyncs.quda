"""
Interface to timer.h
"""


__all__ = [
    "default_profiler",
    "TimeProfile",
]

from .lib import lib

PROFILER = None


def default_profiler():
    "Returns the default profiler which gets initialized once"
    global PROFILER
    if PROFILER is None:
        PROFILER = TimeProfile("Default")
    return PROFILER


class TimeProfile:
    "Mimics quda::TimeProfile"

    __slots__ = ["quda"]
    last = 0

    @classmethod
    def default_name(cls):
        "Returns a default and unique name for the profiler"
        name = f"Profiler{cls.last}"
        cls.last += 1
        return name

    def __init__(self, name=None, use_global=False):
        if name is None:
            name = self.default_name()
        if not isinstance(name, str):
            raise TypeError
        self.quda = lib.TimeProfile(name, use_global)

    @property
    def name(self):
        "Name of the profiler"
        return self.quda.fname

    def print(self):
        "Prints the content of the profiler to stdout"
        self.quda.Print()
