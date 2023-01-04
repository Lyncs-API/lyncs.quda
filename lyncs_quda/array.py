"""
Interface to array.h
"""

__all__ = [
    "lat_dims",
    "Array",
]

from lyncs_utils import isiterable
from .lib import lib

def lat_dims(elems=[1,1,1,1]):
    # QUDA_MAX_DIM = 6
    return Array(int, 6, elems).qarray

class Array:
    "mimics template <typename T, int n> struct array"

    def __init__(self, typename, size, elems=None):
        self._size = size
        self._qarray = lib.quda.array[typename, size]()

        if elems != None:
            if isiterable(elems):
                if len(elems) > size:
                    raise ValueError()
                for i,e in enumerate(elems):
                    self._qarray[i] = e
            else:
                self._qarray[0] = elems
        print(type(self._qarray))

    @property
    def qarray(self):
        return self._qarray
    
    def __len__(self):
        return self._size

    def __getitem__(self, i):
        # TODO: suport slicing
        return self.qarray[i]

    def __setitem(self,i , val):
        self.qarray[i] = val
        
