"Struct base class and tools"

__all__ = [
    "Struct",
    "to_human",
    "to_code",
]

from lyncs_cppyy import nullptr
from lyncs_utils import isiterable, setitems
from .lib import lib
from . import enums


def to_human(val, typ=None):
    "Converts C-value to human-readable"
    if typ is None:
        return val
    if typ in dir(enums):
        return getattr(enums, typ)[val]
    if isinstance(val, (int, float)):
        return val
    if "*" in typ and val == nullptr:
        return 0
    return val


def to_code(val, typ=None):
    "Convert human-readable to C-value"
    if typ is None:
        return val
    if typ in dir(enums):
        if isinstance(val, int):
            return val
        return getattr(enums, typ)[val]
    if typ in ["int", "float", "double"]:
        return val
    if "char" in typ:
        if val in [0, None]:
            return b"\0"
        return str(val)
    if "*" in typ:
        if val in [0, None]:
            return nullptr
    return val


class Struct:
    "Struct base class"
    _types = {}

    def __init__(self, *args, **kwargs):
        self._params = getattr(lib, type(self).__name__)() #? recursive?

        for arg in args:
            self.update(arg)
        self.update(kwargs)

    def keys(self):
        "List of keys in the structure"
        return self._types.keys()

    def items(self):
        "Tuple of (key, value) in the structure"
        return ((key, getattr(self, key)) for key in self.keys())

    def update(self, params):
        "Updates values of the structure"
        if not hasattr(params, "items"): #? in __init__, it takes *args, which is a tuple.  expect a tuple of dict's?
            raise TypeError(f"Unsopported type for params: {type(params)}")
        for key, val in params.items():
            setattr(self, key, val)

    def _assign(self, key, val):
        typ = self._types[key]
        val = to_code(val, typ)
        cur = getattr(self._params, key)

        if hasattr(cur, "shape"):
            setitems(cur, val)
        else:
            setattr(self._params, key, val)

    def __dir__(self):
        return list(set(list(super().__dir__()) + list(self._params.keys())))

    def __getattr__(self, key):
        return to_human(getattr(self._params, key), self._types[key])

    def __setattr__(self, key, val):
        if key in self.keys():
            try:
                self._assign(key, val)
            except TypeError:
                raise TypeError(
                    f"Cannot assign '{val}' to '{key}' of type '{self._types[key]}'"
                )
        else:
            super().__setattr__(key, val)

    def __str__(self):
        return str(dict(self.items()))
