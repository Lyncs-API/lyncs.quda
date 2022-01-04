"""
Interface to lattice_field.h
"""

__all__ = [
    "LatticeField",
]

from array import array
from contextlib import contextmanager
import numpy
from .enums import QudaPrecision
from .lib import lib, cupy


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

@contextmanager
def backend(device=True):
    if device is False or device is None:
        yield numpy
    else:
        if device is True:
            device = lib.device_id

        if not isinstance(device, int):
            raise TypeError("Expected device to be an integer or None/True/False")

        lib.device_id = device
        with cupy.cuda.Device(device):
            yield cupy


class LatticeField:
    "Mimics the quda::LatticeField object"

    @classmethod
    def get_dtype(cls, dtype):
        return numpy.dtype(dtype)

    @classmethod
    def create(cls, lattice, dofs=None, dtype=None, device=True, empty=True, **kwargs):
        "Constructs a new lattice field"
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

    def cast(self, other, **kwargs):
        "Cast a field into its type and check for compatibility"
        cls = type(self)
        if not isinstance(other, cls):
            other = cls(other)
        # TODO: check compatibility using also kwargs
        dofs = (kwargs.get("dofs", self.dofs),)
        assert other.dofs == dofs
        dtype = (kwargs.get("dtype", self.dtype),)
        assert other.dtype == dtype
        device = (kwargs.get("device", self.device),)
        assert other.device == device
        return other

    def prepare(self, *fields, **kwargs):
        "Prepares the fields by creating new one (None) or making them compatible"
        if not fields:
            return self  # Needed? or raise error
        if len(fields) == 1:
            field = fields[0]
            if field is None:
                return self.new(**kwargs)
            return self.cast(field, **kwargs)
        return tuple(self.prepare(field, **kwargs) for field in fields)

    def __init__(self, field, comm=None):
        self.field = field
        self.comm = comm
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
        if not isinstance(field, (numpy.ndarray, cupy.ndarray)):
            raise TypeError("Supporting only numpy or cupy for field")
        if isinstance(field, cupy.ndarray) and field.device.id != lib.device_id:
            raise TypeError("Field is stored on a different device than the quda lib")
        if len(field.shape) < 4:
            raise ValueError("A lattice field should not have shape smaller than 4")
        self._field = field

    def __array__(self, *args, **kwargs):
        out = self.field
        if self.device is not None:
            out = out.get()
        return out.__array__(*args, **kwargs)

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
        "Memory array with lattice dimensions"
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
    #? this assumes: mem_type(QUDA_MEMORY_DEVICE),
    # siteSubset(QUDA_FULL_SITE_SUBSET) => volumeCB =  volume / 2 = stride (as pad==0)
    
    def reduce(self, val, local=False, opr="SUM"):
        if self.comm is None or local:
            return val
        return self.comm.allreduce(val, getattr(lib.MPI, opr))
