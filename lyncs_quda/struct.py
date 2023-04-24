"Struct base class and tools"

__all__ = [
    "Struct",
    "to_human",
    "to_code",
]

import io, sys
from io import TextIOWrapper, BytesIO
import cppyy as cp
import cppyy.ll 
from contextlib import redirect_stdout
from lyncs_cppyy import nullptr, to_pointer, addressof, cppdef
from lyncs_utils import isiterable #, setitems
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
    return val


def to_code(val, typ=None):
    "Convert human-readable to C-value"
    if typ is None:
        return val
    if typ in dir(enums):
        if isinstance(val, int):
            return val
        print("to_code",type(getattr(enums, typ)[val]))
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
        

def setitems(arr, vals, shape=None):
    "Sets items of an iterable object"
    shape = shape if shape is not None else arr.shape
    size = shape[0] #len(arr)
    print("set items",arr,size,shape, vals)
    if hasattr(vals, "__len__") and type(vals) != bytes:
        if len(vals) > size:
            raise ValueError(
                f"Values size ({len(vals)}) larger than array size ({size})"
            )
    else:
        vals = (vals,) * size
    for i, val in enumerate(vals):
        if hasattr(arr[i], "__len__") and len(shape)>1:
            setitems(arr[i], val, shape = shape[1:])
        else:
            arr[i] = val
            print("set item", i,"to",val)

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
            # to avoid codec error
            elif 'file' in key and self._types[key].count("[")<0:
                # TODO: think about a better thing to do here
                #  The Problem: attribute access, getattr(self._quda_params, key), raises 'UnicodeDecodeError: 'utf-8' codec can't decode byte...'
                #               if not initialized to 0's if one uses newQuda* functions
                #print(key, self._types[key])
                setattr(self._quda_params, key, getattr(default_params, key))

        # temporal fix: newQudaMultigridParam does not assign a default value to n_level
        if "Multigrid" in type(self).__name__:
            n = getattr(self._quda_params, "n_level")
            n = lib.QUDA_MAX_MG_LEVEL if n < 0 and n > lib.QUDA_MAX_MG_LEVEL else n
            setattr(self._quda_params, "n_level", n)
            
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
        if not hasattr(params, "items"):
            raise TypeError(f"Unsopported type for params: {type(params)}")
        for key, val in params.items():
            setattr(self, key, val)

    def _assign(self, key, val):
        print("assign")
        typ = self._types[key]
        val = to_code(val, typ) 
        cur = getattr(self._quda_params, key)

        if "[" in self._types[key] and not hasattr(cur, "shape"):# not sure if this is needed for cppyy3.0.0
            # safeguard against hectic behavior of cppyy
            raise RuntimeError("cppyy is not happy for now.  Try again!")

        
        if typ.count("[") > 1:
            # cppyy<=3.0.0 cannot handle subviews properly
            assert hasattr(cur, "shape")
            if "file" in key:
                array = np.chararray(cur.shape)
                setitems(array, b"\0")
                setitems(array, vals)
                size = 1
            else:
                dtype = get_dtyp(typ[:-typ.index("[")].strip())
                array = np.empty(cur.shape, dtype=dtype)
                size = dtype.itemsize
            lib.memcpy(to_pointer(addressof(cur)), to_pointer(array.__array_interface__["data"][0]), int(np.prod(cur.shape))*size)
        elif typ.count("[") == 1:
            assert hasattr(cur, "shape")
            shape = tuple([getattr(lib, macro) for macro in typ.split(" ") if "QUDA_" in macro or macro.isnumeric()]) #not necessary for cppyy3.0.0? 
            print("assign (shape)",key,typ,val,cur,cur.shape,  isiterable(cur), shape)
            cur.reshape(shape) #not necessary for cppyy3.0.0?
            if "*" in typ: 
                for i in range(shape[0]):
                    val = to_pointer(addressof(val), ctype = typ[:-typ.index("[")].strip())
            setitems(cur, val)
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
        print("set atr")
        if key in self.keys():
            try:
                self._assign(key, val)
            except TypeError:
                raise TypeError(
                    f"Cannot assign '{val}' to '{key}' of type '{self._types[key]}'"
                )
        else: #should we allow this?
            super().__setattr__(key, val)

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
