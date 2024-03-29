"""
Interface to lattice_field.h
"""

__all__ = [
    "LatticeField",
]

from array import array
from contextlib import contextmanager
from functools import cache
import numpy
from lyncs_cppyy import nullptr
from lyncs_utils import prod
from .enums import QudaPrecision, QudaFieldLocation, QudaGhostExchange
from .lib import lib, cupy
from .array import lat_dims

from lyncs_cppyy import to_pointer
import ctypes
import traceback


def get_precision(dtype):
    if dtype in ["float64", "complex128"]:
        return "double"
    if dtype in ["float32", "complex64"]:
        return "single"
    if dtype in ["float16", "complex32"]:
        return "half"
    raise ValueError


def get_complex_dtype(dtype):
    "Return equivalent complex dtype"
    if dtype in ["float64", "complex128"]:
        return "complex128"
    if dtype in ["float32", "complex64"]:
        return "complex64"
    if dtype in ["float16", "complex32"]:
        return "complex32"
    raise TypeError(f"Cannot convert {dtype} to complex")


def get_float_dtype(dtype):
    "Return equivalent float dtype"
    if dtype in ["float64", "complex128"]:
        return "float64"
    if dtype in ["float32", "complex64"]:
        return "float32"
    if dtype in ["float16", "complex32"]:
        return "float16"
    raise TypeError(f"Cannot convert {dtype} to float")


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
    try:
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
    finally:  # I don't think this will be invked when exception occurs
        cupy.cuda.runtime.setDevice(lib.device_id)


class LatticeField(numpy.lib.mixins.NDArrayOperatorsMixin):
    "Mimics the quda::LatticeField object"

    @classmethod
    def get_dtype(cls, dtype):
        return numpy.dtype(dtype)

    @classmethod
    def create(
        cls,
        lattice,
        dofs=None,
        dtype=None,
        device=True,
        comm=None,
        empty=True,
        **kwargs,
    ):
        "Constructs a new lattice field with default dtype=None, translating into float64"
        # IN: lattice: represetns local lattice only if (nu/cu)py array; else global lattice
        # IN: comm: Cartesian communicator

        if isinstance(lattice, cls):
            return lattice

        if comm is None:
            comm = lib.comm
        # Compute local lattice dims
        if comm is not None and not isinstance(lattice, (numpy.ndarray, cupy.ndarray)):
            global_lattice = lattice
            local_lattice = ()
            procs = comm.dims
            for ldim, cdim in zip(global_lattice, procs):
                if not (ldim / cdim).is_integer():
                    raise ValueError(
                        "Each lattice dim needs to be divisible by the corresponding dim of the Cartesian communicator!"
                    )
                local_lattice += (int(ldim / cdim),)
        else:
            # no commnicator is taken as indication of a single MPI job
            local_lattice = lattice

        with backend(device) as bck:
            if isinstance(lattice, (numpy.ndarray, cupy.ndarray)):
                return cls(bck.array(lattice), comm=comm, **kwargs)

            new = bck.empty if empty else bck.zeros

            shape = tuple(dofs) + tuple(local_lattice)
            return cls(new(shape, dtype=cls.get_dtype(dtype)), comm=comm, **kwargs)

    def new(self, empty=True, **kwargs):
        "Returns a new empty field based on the current"

        out = self.create(
            self.global_lattice,
            dofs=kwargs.pop("dofs", self.dofs),
            dtype=kwargs.pop("dtype", self.dtype),
            device=kwargs.pop("device", self.device),
            comm=kwargs.pop("comm", self.comm),
            empty=empty,
            **kwargs,
        )
        out.__array_finalize__(self)
        return out

    def __array_finalize__(self, obj):
        "Support for __array_finalize__ standard"
        if self.comm is None:
            self.comm = obj.comm

    def copy(self, other=None, out=None, **kwargs):
        "Returns out, a copy+=kwargs, of other if given, else of self"
        # src: other if given; otherwise self
        # dst: out (created anew, if not explicitly given)
        # ASSUME: if out != None, out <=> other/self+=kwargs
        #
        # IF other and out are both given, this behaves like a classmethod
        # where out&other are casted into type(self)

        # check=False => here any output is accepted
        out = self.prepare_out(out, check=False, **kwargs)

        if other is None:
            other = self
        # we prepare other without copying because we do copy here!
        other = out.prepare_in(other, copy=False, check=False, **kwargs)
        try:
            out.quda_field.copy(other.quda_field)
        except:  # NotImplementedError:  #raised if self is LatticeField# at least, serial version calls exit(1) from qudaError, which is not catched by this
            # As last resort trying to copy elementwise
            out.default_view()[:] = other.default_view()

        return out

    def equivalent(self, other, **kwargs):
        "Whether other is equivalent to self with kwargs"
        # Check if metadata of (self+kwargs) == other

        if not isinstance(other, type(self)):
            return False
        dtype = kwargs.get("dtype", self.dtype)
        if other.dtype != dtype:
            return False
        device = kwargs.get("device", self.device)
        if other.device != device:
            return False
        dofs = kwargs.get("dofs", self.dofs)
        if other.dofs != dofs:
            return False
        return True

    def _prepare(self, field, copy=False, check=False, **kwargs):
        if field is self:
            return field
        if field is None:
            return self.new(**kwargs)
        cls = type(self)
        if not isinstance(field, cls):
            field = cls(field)
        if check and not self.equivalent(field, **kwargs):
            if copy:
                return self.copy(other=field, **kwargs)
            raise ValueError("The given field is not appropriate")
        field.__array_finalize__(self)
        return field

    def prepare(self, fields, **kwargs):
        """Prepares the fields by creating new ones if None given;
        else casting them to type(self), then checking them if compatible with self,
        and/or copying them
        """
        if isinstance(fields, (tuple, list)):
            return type(fields)(self.prepare(field, **kwargs) for field in fields)
        return self._prepare(fields, **kwargs)

    def prepare_out(self, fields, **kwargs):
        "Function to call for preparing output(s) to be passed for a calculation"
        # Typically, we do want to check but not copy an output
        kwargs.setdefault("check", True)
        kwargs.setdefault("copy", False)
        if kwargs["copy"]:
            raise ValueError("An output should never be copied")
        return self.prepare(fields, **kwargs)

    def prepare_in(self, fields, **kwargs):
        "Function to call for preparing input(s) to be used for a calculation"
        # Typically, we want to check and copy an input
        kwargs.setdefault("check", True)
        kwargs.setdefault("copy", True)
        return self.prepare(fields, **kwargs)

    def __init__(self, field, comm=None, **kwargs):
        self.field = field
        if comm is False:
            self.comm = comm
        elif comm is None:
            self.comm = lib.comm
        else:
            self.comm = comm
        self._quda = None
        self.activate()

    def activate(self):
        "Activates the current field. To be called before using the object in quda"
        "to make sure the communicator is set for MPI"
        # if self.comm is None, but #ranks>1, this will mess thigns up
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
            raise TypeError(
                f"Supporting only numpy or cupy for field, got {type(field)}"
            )
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
        return self.field.view(get_complex_dtype(self.dtype))

    def float_view(self):
        "Returns a complex view of the field"
        if not self.iscomplex:
            return self.field
        return self.field.view(get_float_dtype(self.dtype))

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
        return int(QudaFieldLocation[self.location])

    @property
    def ndims(self):
        "Number of lattice dimensions"
        return 4

    @property
    def dims(self):
        "Shape of the local lattice dimensions"
        return self.shape[-self.ndims :]

    @property
    def local_lattice(self):
        "Returns the local lattice size"
        return self.dims

    @property
    def global_lattice(self):
        "Returns the global lattice size"
        if self.comm is None:
            return self.local_lattice
        return tuple(int(cdim * ldim) for cdim, ldim in zip(self.comm.dims, self.dims))

    @property
    def quda_dims(self):
        "Memory array with lattice dimensions"
        return lat_dims(tuple(reversed(self.dims)))

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
        return int(QudaPrecision[self.precision])

    @property
    def ghost_exchange(self):
        "Ghost exchange"
        return "NO"

    @property
    def quda_ghost_exchange(self):
        "Quda enum for ghost exchange"
        return int(QudaGhostExchange[self.ghost_exchange])

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

    @staticmethod
    @cache
    def _quda_params(*args):
        "Call wrapper to cache param structures"
        return lib.LatticeFieldParam(*args)

    @property
    def quda_params(self):
        "Returns an instance of quda::LatticeFieldParam"
        return self._quda_params(
            self.ndims,
            self.quda_dims,
            self.pad,
            self.quda_location,
            self.quda_precision,
            self.quda_ghost_exchange,
        )

    # ? this assumes: mem_type(QUDA_MEMORY_DEVICE),
    # siteSubset(QUDA_FULL_SITE_SUBSET) => volumeCB =  volume / 2 = stride (as pad==0)
    # Here, local volume (with no halo) = volume (with halo) as ghost_exchange == NO

    def reduce(self, val, local=False, opr="SUM"):
        # ? may be better to avoid use of cupy's get until this point
        # and convert the result of reduction to dtype of val,
        # which potentially involves device-to-host communication
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
        out = kwargs.get("out", (self,))[0]

        prepare = (
            lambda arg: out.prepare_in(arg).field
            if isinstance(arg, (LatticeField, cupy.ndarray, numpy.ndarray))
            else arg
        )
        args = tuple(map(prepare, args))

        for key, val in kwargs.items():
            if isinstance(val, (tuple, list)):
                kwargs[key] = type(val)(map(prepare, val))
            else:
                kwargs[key] = prepare(val)

        fnc = getattr(ufunc, method)

        return self.prepare_out(fnc(*args, **kwargs), check=False)

    def __bool__(self):
        return bool(self.field.all())
