"Struct base class and tools"

__all__ = [
    "Struct",
    "to_human",
    "to_code",
]

import numpy as np
from lyncs_cppyy import nullptr, to_pointer, addressof
from lyncs_utils import isiterable
from .lib import lib
from . import enums


def to_human(val, typ=None):
    "Converts C-value to human-readable"
    if typ is None:
        return val
    if typ in dir(enums):
        return str(getattr(enums, typ)[val])
    if isinstance(val, (int, float)):
        return val
    if "*" in typ and val == nullptr:
        return 0
    if "char" in typ:
        return "".join(list(val))
    return val


def to_code(val, typ=None):
    "Convert human-readable to C-value"
    if typ is None:
        return val
    if typ in dir(enums):
        if isinstance(val, int):
            return val
        return int(getattr(enums, typ)[val])
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


def get_dtype(typ):
    if "*" in typ:
        return np.dtype(object)
    typ_dict = {"complex":"c", "unsigned":"u", "float":"single", "char":"byte", "comlex_double":"double"}
    typ_list = typ.split()
    dtype = ""
    for w in typ_list:
        if w in ("complex", "unsigned"):
            dtype = typ_dict[w] + dtype
        dtype += typ_dict.get(w, w)
    if typ in ("bool", "long"):
        dtype += "_"
    if "int" in typ:
        dtype += "c"
    return np.dtype(dtype)
        

def setitems(arr, vals, shape=None, is_string=False):
    "Sets items of an iterable object"
    shape = shape if shape is not None else arr.shape
    size = shape[0] #len(arr)
    if not is_string and type(vals) == str:
        # sometimes, vals is turned into str
        vals = eval(vals)
    if hasattr(vals, "__len__") and type(vals) != bytes:
        if len(vals) > size:
            raise ValueError(
                f"Values size ({len(vals)}) larger than array size ({size})"
            )
    else:
        vals = (vals,) * size
    for i, val in enumerate(vals):
        if len(shape)>1 and hasattr(arr[i], "__len__"):
            is_string = len(shape[1:]) == 1 and type(vals[0]) == str
            setitems(arr[i], val, shape = shape[1:], is_string=is_string)
        else:
            arr[i] = val

            
class Struct:
    "Struct base class"
    _types = {}

    def __init__(self, *args, **kwargs):
        #? is *args necessary? when provided, it causes error in update
        self._quda_params = getattr(lib, "new"+type(self).__name__)()

        # some fields are not set by QUDA's new* function
        default_params = getattr(lib, type(self).__name__)()        
        for key in self.keys():
            # to avoid Enum error due to unexpected key-value pair
            if self._types[key] in dir(enums) and not key in kwargs:
                enm = getattr(enums, self._types[key])
                if not getattr(self._quda_params, key) in enm.values():
                    val = list(enm.values())[-1]
                    self._assign(key, val)
            if "char" in self._types[key]:
                self._assign(key, b"\0")
                
        # temporal fix: newQudaMultigridParam does not assign a default value to n_level
        if "Multigrid" in type(self).__name__:
            n = getattr(self._quda_params, "n_level")
            n = 2 if n < 0 or n > lib.QUDA_MAX_MG_LEVEL else n
            setattr(self._quda_params, "n_level", n)
            
        for arg in args:
            self.update(arg)
        self.update(kwargs)
        self.updated = False
        
    def keys(self):
        "List of keys in the structure"
        return self._types.keys()

    def items(self):
        "Tuple of (key, value) in the structure"
        return ((key, getattr(self, key)) for key in self.keys())

    def update(self, params):
        "Updates values of the structure"
        if not hasattr(params, "items"):
            raise TypeError(f"Unsopported type for params: {type(params)}")
        for key, val in params.items():
            setattr(self, key, val)

    def _assign(self, key, val):
        typ = self._types[key]
        val = to_code(val, typ) 
        cur = getattr(self._quda_params, key)

        if "[" in self._types[key] and not hasattr(cur, "shape"):
            # safeguard against hectic behavior of cppyy
            # QudaEigParam *eig_param[QUDA_MAX_MG_LEVEL] is somehow turned into QudaEigParam **
            raise RuntimeError("cppyy is not happy for now.  Try again!")

        
        if typ.count("[") > 1:
            # cppyy<=3.0.0 cannot handle subviews properly
            #  Trying to manipulate the sub-array either results in error or segfault
            #   => array of arrays is set using glb.memcpy
            # Alternative:
            #  use ctypes (C = ctypes, arr = LowlevelView of array of arrays)
            #    ptr = C.cast(cppyy.ll.addressof(arr), C.POINTER(C.c_int))
            #    narr = np.ctypeslib.as_array(ptr, shape=arr.shape)
            #  This allows to access sub-indicies properly, i.e., narr[2][3] = 9 works
            assert hasattr(cur, "shape")
            if "file" in key:
                #? array = np.zeros(cur.shape, dtype="S1") and remove setitems(array, b"\0"); is this ok?
                #? not sure of this as "" is not b"\0"
                array = np.chararray(cur.shape)
                setitems(array, b"\0") 
                setitems(array, val)
                size = 1
            else:
                dtype = get_dtype(typ[:typ.index("[")].strip())
                array = np.asarray(val, dtype=dtype)
                size = dtype.itemsize
            lib.memcpy(to_pointer(addressof(cur)), to_pointer(array.__array_interface__["data"][0]), int(np.prod(cur.shape))*size)
        elif typ.count("[") == 1:
            assert hasattr(cur, "shape")
            shape = tuple([getattr(lib, macro) for macro in typ.split(" ") if "QUDA_" in macro or macro.isnumeric()]) #not necessary for cppyy3.0.0? 
            cur.reshape(shape) #? not necessary for cppyy3.0.0?
            if "*" in typ:
                if hasattr(val, "__len__"):
                    val = [to_pointer(addressof(v), ctype = typ[:-typ.index("[")].strip()) for v in val]
                else:
                    val = to_pointer(addressof(v), ctype = typ[:-typ.index("[")].strip())
            is_string = True if "char" in typ else False
            if is_string:
                setitems(cur, b"\0") # for printing
            setitems(cur, val, is_string=is_string)
        else:
            if "*" in typ:
                # cannot set nullptr to void *, int *, etc; works for classes such as Enum classes with bind_object
                if val == nullptr:
                    raise ValueError("Cannot cast nullptr to a valid pointer")
                val = to_pointer(addressof(val), ctype = typ)
            setattr(self._quda_params, key, val)

    def __dir__(self):
        return list(set(list(super().__dir__()) + list(self._quda_params.keys())))

    def __getattr__(self, key):
        return to_human(getattr(self._quda_params, key), self._types[key])

    def __setattr__(self, key, val):
        if key in self.keys():
            try:
                self._assign(key, val)
            except TypeError:
                raise TypeError(
                    f"Cannot assign '{val}' to '{key}' of type '{self._types[key]}'"
                )
        else: #should we allow this?
            super().__setattr__(key, val)
        super().__setattr__("updated", True)
            
    def __str__(self):
        return str(dict(self.items()))

    @property
    def quda(self):
        return self._quda_params

    @property
    def address(self):
        return addressof(self.quda)
    
    @property
    def ptr(self):
        return to_pointer(addressof(self.quda), ctype = type(self).__name__ + " *")

    def printf(self):
        getattr(lib, "print"+type(self).__name__)(self._quda_params)
