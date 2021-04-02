"""
Interface to lattice_field.h
"""

__all__ = [
    "LatticeField",
]

from array import array
import numpy
import cupy
from .lib import lib


class LatticeField:
    "Mimics the quda::LatticeField object"

    @classmethod
    def create(cls, lattice, dofs, dtype=None, device=True, comm=None):
        "Constructs a new gauge field"
        shape = tuple(dofs) + tuple(lattice)
        field_kwargs = dict(dtype=dtype)
        cls_kwargs = dict(comm=comm)

        if device is False or device is None:
            return cls(numpy.empty(shape, **field_kwargs), **cls_kwargs)

        if device is True:
            device = lib.device_id
        else:
            lib.device_id = device

        with cupy.cuda.Device(device):
            return cls(cupy.empty(shape, **field_kwargs), **cls_kwargs)

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
    def precision(self):
        "Field data type precision"
        if not str(self.dtype).startswith("float"):
            return "INVALID"
        if str(self.dtype).endswith("64"):
            return "DOUBLE"
        if str(self.dtype).endswith("32"):
            return "SINGLE"
        if str(self.dtype).endswith("16"):
            return "HALF"
        return "INVALID"

    @property
    def quda_precision(self):
        "Quda enum for field data type precision"
        return getattr(lib, f"QUDA_{self.precision}_PRECISION")

    @property
    def order(self):
        "Data order of the field"
        return "FLOAT2"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_GAUGE_ORDER")

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
