"""
Interface to lattice_field.h
"""

__all__ = [
    "LatticeField",
]

from array import array
from contextlib import contextmanager
import numpy
from lyncs_cppyy import nullptr
from lyncs_utils import prod
from .enums import QudaPrecision
from .lib import lib, cupy

from lyncs_cppyy import to_pointer
import ctypes

def get_precision(dtype):
    if dtype in ["float64", "complex128"]:
        return "double"
    if dtype in ["float32", "complex64"]:
        return "single"
    if dtype in ["float16"]:  # , "complex32"
        return "half"
    raise ValueError


def get_ptr(array):
    "Memory pointer"
    if isinstance(array, numpy.ndarray):
        return array.__array_interface__["data"][0]
    return array.data.ptr


def reshuffle(field, N0, N1):
    #### PROPOSAL - template #### not sure if this is better than manual for-loop in Python
    # TODO: write safegurds
    # ASSUME: array in field has shape= dofs+lattice so that self.dofs, etc works; add dofs attribute if not
    # ASSUME: Given array is actually ordered so that parity is slowest running index; currently is default; see below

    xp = field.backend

    sub = []
    sub.append(field.field.reshape((2, -1))[0, :])
    sub.append(field.field.reshape((2, -1))[1, :])

    dof = (1,) + field.dofs if len(field.dofs) == 1 else field.dofs
    idof = prod(field.dofs[1:])
    dof0 = (dof[0],) + (idof // N0,)
    dof1 = (dof[0],) + (idof // N1,)
    for i in range(2):
        sub[i] = xp.transpose(
            xp.transpose(sub[i].reshape(dof0 + (-1, N0)), axes=(0, 2, 1, 3)).reshape(
                (dof[0],) + (-1, dof1[1], N1)
            ),
            axes=(0, 2, 1, 3),
        )

    field.field = xp.concatenate(
        sub[0] + sub[1]
    )  # ATTENTION: this will affects self.dofs, etc


@contextmanager
def backend(device=True):
    if device is False or device is None:
        yield numpy
    else:
        if device is True:
            device = lib.device_id

        if not isinstance(device, int):
            raise TypeError("Expected device to be an integer or None/True/False")

        # ? is this safe?
        lib.device_id = device
        with cupy.cuda.Device(device):
            yield cupy


class LatticeField(numpy.lib.mixins.NDArrayOperatorsMixin):
    "Mimics the quda::LatticeField object"

    @classmethod
    def get_dtype(cls, dtype):
        return numpy.dtype(dtype)

    @classmethod
    def create(cls, lattice, dofs=None, dtype=None, device=True, empty=True, **kwargs):
        "Constructs a new lattice field with default dtype=None, translating into float64"
        if isinstance(lattice, cls):
            return lattice

        with backend(device) as bck:
            if isinstance(lattice, (numpy.ndarray, cupy.ndarray)):
                return cls(bck.array(lattice), **kwargs)

            new = bck.empty if empty else bck.zeros
            shape = tuple(dofs) + tuple(lattice)
            return cls(new(shape, dtype=cls.get_dtype(dtype)), **kwargs)

    def new(self, empty=True, **kwargs):
        "Returns a new empty field based on the current"
        return self.create(
            self.lattice,
            dofs=kwargs.get("dofs", self.dofs),
            dtype=kwargs.get("dtype", self.dtype),
            device=kwargs.get("device", self.device),
            empty=empty,
        )

    def copy(self, other=None, out=None, **kwargs):
        "Returns out, a copy+=kwargs, of other if given, else of self"
        # ASSUME: if out != None, out <=> other/self+=kwargs
        # if other and out are both given, this behaves like a classmethod except out&other are casted into type(self)

        out = self.prepare(out, copy=False, check=False, **kwargs) #check=False => if out!=None, converts from type(out) to type(self); else create a new one with kwargs
        
        if other is None:
            other = self
        other = out.cast(other, copy=False, check=False, **kwargs)
        
        try:
            out.quda_field.copy(other.quda_field)
        except NotImplementedError: # at least, serial version calls exit(1) from qudaError, which is not catched by this
            assert False
            out = out.prepare((other.field.copy()), copy=False) #the orignal code may lead to infinite recursion
        return out

    def equivalent(self, other, switch=False, **kwargs):
        "Whether other is equivalent to self with kwargs"
        # Check if self+kwargs <=> other

        if switch:
            self, other = other, self

        if not isinstance(other, type(self)):
            return False
        dtype = kwargs.get("dtype", self.dtype)
        if other.dtype != dtype:
            return False
        device = kwargs.get("device", self.device)
        if other.device != device:
            return False
        dofs = kwargs.get("dofs", self.dofs) #None) #? to force to specify dof in kwargs?
        if dofs and other.dofs != dofs:
            return False
        return True

    def cast(self, other=None, copy=True, check=False, **kwargs):
        "Cast other (self if not given) into type(self) and copy or check for compatibility"

        if other is None:
            other = self
        return self.prepare(other, copy=copy, check=check, **kwargs)

    """
    def cast(self, other=None, copy=True, check=True, **kwargs):
        "Cast a field into its type and check for compatibility"
        cls = type(self)
        if other is None:
            other = self
        if not isinstance(other, cls):
            other = cls(other)

        # (check, copy)                 expect
        # (T, T): check but not copy      check & copy
        # (T, F): copy but no check       check & not copy
        # (F, T): does nothing            no check & just copy
        # (F, F): does nothing            does nothing
        if check and not self.equivalent(other, **kwargs):
            if copy:
                raise ValueError("The given field is not appropriate")
            return self.copy(other, **kwargs) 
        return other
    """
        
    def prepare(self, *fields, copy=True, check=False, switch=False, **kwargs):
        "Prepares the fields by creating new one if None given else casting them to type(self) then  checking them if compatible with self and/or copying them"
        if not fields:
            return self  # Needed? or raise error
        if len(fields) == 1:
            field = fields[0]
            if field is None:
                return self.new(**kwargs)
            cls = type(self)
            if not isinstance(field, cls):
                field = cls(field)
            if check and not self.equivalent(field, switch=switch, **kwargs):
                raise ValueError("The given field is not appropriate")
            if copy:
                return self.copy(other=field, **kwargs)
            return field
        return tuple(self.prepare(field, **kwargs) for field in fields)

    def __init__(self, field, comm=None):
        self.field = field
        self.comm = comm
        self._quda = None
        self.activate()

    def activate(self):
        "Activates the current field. To be called before using the object in quda"
        lib.set_comm(self.comm)

    @property
    def field(self):
        "The underlaying lattice field"
        return self._field

    @field.setter
    def field(self, field):
        if isinstance(field, LatticeField):
            field = field.field
        if not isinstance(field, (numpy.ndarray, cupy.ndarray)):
            raise TypeError("Supporting only numpy or cupy for field")
        if isinstance(field, cupy.ndarray) and field.device.id != lib.device_id:
            raise TypeError("Field is stored on a different device than the quda lib")
        if len(field.shape) < 4:
            raise ValueError("A lattice field should not have shape smaller than 4")
        self._field = field

    def get(self):
        "Returns the field as numpy array"
        return self.__array__()

    def __array__(self, *args, **kwargs):
        out = self.field
        if self.device is not None:
            out = out.get()
        return out.__array__(*args, **kwargs)

    def complex_view(self):
        "Returns a complex view of the field"
        if self.iscomplex:
            return self.field
        if self.dtype == "float64":
            return self.field.view("complex128")
        elif self.dtype == "float32":
            return self.field.view("complex64")
        raise TypeError(f"Cannot convert {self.dtype} to complex")

    def default_view(self):
        "Returns the default view of the field including reshaping"
        return self.complex_view()

    @property
    def backend(self):
        "The backend of the field: cupy or numpy"
        if isinstance(self.field, cupy.ndarray):
            return cupy
        return numpy

    @property
    def device(self):
        "Device id of the field (None if not on GPUs)"
        if isinstance(self.field, cupy.ndarray):
            return self.field.device.id
        return None

    @property
    def shape(self):
        "Shape of the field"
        return self.field.shape

    @property
    def location(self):
        "Memory location of the field (CPU or CUDA)"
        return "CPU" if isinstance(self.field, numpy.ndarray) else "CUDA"

    @property
    def quda_location(self):
        "Quda enum for memory location of the field (CPU or CUDA)"
        return getattr(lib, f"QUDA_{self.location}_FIELD_LOCATION")

    @property
    def ndims(self):
        "Number of lattice dimensions"
        return 4

    @property
    def dims(self):
        "Shape of the lattice dimensions"
        return self.shape[-self.ndims :]

    lattice = dims

    @property
    def quda_dims(self):
        "Memory array with lattice dimensions including halo width"
        return array("i", self.dims)
    
    @property
    def dofs(self):
        "Shape of the per-site degrees of freedom"
        return self.shape[: -self.ndims]

    @property
    def dtype(self):
        "Field data type"
        return self.field.dtype

    @property
    def iscomplex(self):
        "Whether the field dtype is complex"
        return self.backend.iscomplexobj(self.field)

    @property
    def isreal(self):
        "Whether the field dtype is real"
        return self.backend.isrealobj(self.field)

    @property
    def precision(self):
        "Field data type precision"
        return get_precision(self.dtype)

    @property
    def quda_precision(self):
        "Quda enum for field data type precision"
        return int(QudaPrecision[get_precision(self.dtype)])

    @property
    def ghost_exchange(self):
        "Ghost exchange"
        return "NO"

    @property
    def quda_ghost_exchange(self):
        "Quda enum for ghost exchange"
        return getattr(lib, f"QUDA_GHOST_EXCHANGE_{self.ghost_exchange}")

    @property
    def pad(self):
        "Memory padding"
        return 0

    @property
    def ptr(self):
        "Memory pointer"
        if isinstance(self.field, numpy.ndarray):
            return self.field.__array_interface__["data"][0]
        return self.field.data.ptr

    @property
    def quda_params(self):
        "Returns an instance of quda::LatticeFieldParam"
        return lib.LatticeFieldParam(
            self.ndims,
            self.quda_dims,
            self.pad,
            self.quda_precision,
            self.quda_ghost_exchange,
        )

    # ? this assumes: mem_type(QUDA_MEMORY_DEVICE),
    # siteSubset(QUDA_FULL_SITE_SUBSET) => volumeCB =  volume / 2 = stride (as pad==0)
    # Here, local volume (with no halo) = volume (with halo) as ghost_exchange == NO

    def reduce(self, val, local=False, opr="SUM"):
        if self.comm is None or local:
            return val
        return self.comm.allreduce(val, getattr(lib.MPI, opr))

    @property
    def quda_field(self):
        "Returns an instance of a quda class"
        raise NotImplementedError("Creating a LatticeField")

    @property
    def cpu_field(self):
        "Returns a cpuField class if possible, otherwise nullptr"
        if self.device is None:
            return self.quda_field
        return nullptr

    @property
    def gpu_field(self):
        "Returns a gpuField class if possible, otherwise nullptr"
        if self.device is not None:
            return self.quda_field
        return nullptr

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        prepare = (
            lambda arg: self.cast(arg).field if isinstance(arg, LatticeField) else arg
        )
        args = tuple(map(prepare, args))

        for key, val in kwargs.items():
            if isinstance(val, (tuple, list)):
                kwargs[key] = type(val)(map(prepare, val))
            else:
                kwargs[key] = prepare(val)

        fnc = getattr(ufunc, method)
        return type(self)(fnc(*args, **kwargs))

    def __bool__(self):
        return bool(self.field.all())
