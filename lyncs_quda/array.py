"""
Interface to array.h
"""

__all__ = [
    "lat_dims",
    "Array",
]

from functools import cache
from lyncs_utils import isiterable
from .lib import lib


@cache
def lat_dims(elems=(1, 1, 1, 1)):
    # QUDA_MAX_DIM = 6 by default
    return Array(int, lib.QUDA_MAX_DIM, elems).qarray


class Array:
    "mimics template <typename T, int n> struct array"

    def __init__(self, typename, size, elems=None):
        self._qarray = lib.quda.array[typename, size]()

        if elems != None:
            if isiterable(elems):
                if len(elems) > size:
                    raise ValueError()
                for idx, val in enumerate(elems):
                    self[idx] = val
            else:
                for i in range(size):
                    self[i] = elems

    @property
    def qarray(self):
        return self._qarray

    def __len__(self):
        return self.qarray.N

    def __getitem__(self, idx):
        # TODO: suport slicing
        return self.qarray[idx]

    def __setitem__(self, idx, val):
        self.qarray[idx] = val
