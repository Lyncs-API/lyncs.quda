"""
Interface to lattice_field.h
"""

__all__ = [
    "LatticeField",
]

from array import array
from contextlib import contextmanager
import numpy
import cupy
from .lib import lib


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
    def create(cls, lattice, dofs=None, dtype=None, device=True, **kwargs):
        "Constructs a new gauge field"
        if isinstance(lattice, cls):
            return lattice

        with backend(device) as bck:
            if isinstance(lattice, (numpy.ndarray, cupy.ndarray)):
                return cls(bck.array(lattice), **kwargs)

            shape = tuple(dofs) + tuple(lattice)
            return cls(bck.empty(shape, dtype=cls.get_dtype(dtype)), **kwargs)

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
        "Whether the field dtype is complex"
        return self.backend.isrealobj(self.field)

    @property
    def precision(self):
        "Field data type precision"
        if self.dtype in ["float64", "complex128"]:
            return "DOUBLE"
        if self.dtype in ["float32", "complex64"]:
            return "SINGLE"
        if self.dtype in ["float16", "complex32"]:
            return "HALF"
        return "INVALID"

    @property
    def quda_precision(self):
        "Quda enum for field data type precision"
        return getattr(lib, f"QUDA_{self.precision}_PRECISION")

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
        "Returns and instance of quda::LatticeFieldParam"
        return lib.LatticeFieldParam(
            self.ndims,
            self.quda_dims,
            self.pad,
            self.quda_precision,
            self.quda_ghost_exchange,
        )

    def reduce(self, val, local=False, opr="SUM"):
        if self.comm is None or local:
            return val
        return self.comm.allreduce(val, getattr(lib.MPI, opr))
