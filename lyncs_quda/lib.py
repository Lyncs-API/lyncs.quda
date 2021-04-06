"""
Loading the QUDA library
"""
# pylint: disable=C0103

__all__ = [
    "lib",
    "PATHS",
]

from os import environ
from pathlib import Path
from array import array
from appdirs import user_data_dir
from lyncs_cppyy import Lib, nullptr, cppdef
from lyncs_cppyy.ll import addressof, to_pointer, cast
from lyncs_utils import static_property
from . import __path__
from .config import QUDA_MPI, GITVERSION, CUDA_INCLUDE


class QudaLib(Lib):
    "Adds additional enviromental control required by QUDA"

    __slots__ = [
        "_initialized",
        "_device_id",
        "_comm",
    ]

    def __init__(self, *args, **kwargs):
        self._initialized = False
        self._device_id = 0
        self._comm = None
        if not self.tune_dir:
            self.tune_dir = user_data_dir("quda", "lyncs") + "/" + GITVERSION
        super().__init__(*args, **kwargs)

    @static_property
    def MPI():
        if not QUDA_MPI:
            raise RuntimeError("Quda has not been compiled with MPI")
        from mpi4py import MPI

        return MPI

    @property
    def initialized(self):
        "Whether the QUDA library has been initialized"
        return self._initialized

    @property
    def tune_dir(self):
        "The directory where the tuning files are stored"
        return environ.get("QUDA_RESOURCE_PATH")

    @tune_dir.setter
    def tune_dir(self, value):
        environ["QUDA_RESOURCE_PATH"] = value

    @property
    def tune_enabled(self):
        "Returns if tuning is enabled"
        if environ.get("QUDA_ENABLE_TUNING"):
            return environ["QUDA_ENABLE_TUNING"] == "1"
        return True

    @tune_enabled.setter
    def tune_enabled(self, value):
        if value:
            environ["QUDA_ENABLE_TUNING"] = "1"
        else:
            environ["QUDA_ENABLE_TUNING"] = "0"

    @property
    def device_id(self):
        "Device id to use"
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        if value == self.device_id:
            return
        if self.initialized:
            raise RuntimeError(
                f"device_id cannot be changed: current={self.device_id}, given={value}"
            )
        if not isinstance(value, int):
            raise TypeError(f"Unsupported type for device: {type(device)}")
        self._device_id = value

    def get_current_device(self):
        dev = array("i", [0])
        super().__getattr__("cudaGetDevice")(dev)
        return dev[0]

    def init_quda(self, dev=None):
        if self.initialized:
            raise RuntimeError("Quda already initialized")
        # As first we set initialized to True to avoid recursion
        self._initialized = True
        if self.tune_dir:
            Path(self.tune_dir).mkdir(parents=True, exist_ok=True)
        if QUDA_MPI and self.comm is None:
            self.set_comm()
        if QUDA_MPI:
            comm = cast["MPI_Comm"](self.MPI._handleof(self.comm))
            comm_ptr = self._comm_ptr(comm)
            self.setMPICommHandleQuda(comm_ptr)
            dims = array("i", self.comm.dims)
            self.initCommsGridQuda(4, dims, self._comms_map, comm_ptr)

        if dev is None:
            if QUDA_MPI:
                dev = -1
            else:
                dev = self.device_id
        self.initQuda(dev)
        self.device_id = self.get_current_device()

    def set_comm(self, comm=None):
        if comm is not None and not QUDA_MPI:
            raise RuntimeError("Quda has not been compiled with MPI")
        if comm is None and not QUDA_MPI:
            return
        if comm is None:
            comm = self.MPI.COMM_SELF.Create_cart((1, 1, 1, 1))
        if not isinstance(comm, self.MPI.Cartcomm):
            raise TypeError("comm expected to be a Cartcomm")
        if (
            self._comm is not None
            and self.MPI.Comm.Compare(comm, self._comm) <= self.MPI.CONGRUENT
        ):
            return
        if comm.ndim != 4:
            raise ValueError("comm expected to be a 4D Cartcomm")
        if self._comm is not None:
            self.end_quda()
        self._comm = comm

    @property
    def _comm_ptr(self):
        try:
            return self.lyncs_quda_comm_ptr
        except AttributeError:
            cppdef(
                """
                void * lyncs_quda_comm_ptr(MPI_Comm &comm) {
                  return &comm;
                }
                """
            )
        return self.lyncs_quda_comm_ptr

    @property
    def _comms_map(self):
        try:
            return self.lyncs_quda_cart_comms_map
        except AttributeError:
            cppdef(
                """
                int lyncs_quda_cart_comms_map(const int *coords, void *fdata) {
                  auto comm = *static_cast<MPI_Comm *>(fdata);
                  int rank;
                  MPI_Cart_rank(comm, coords, &rank);
                  // printf("coord=(%d,%d,%d,%d) -> rank=%d\\n",
                  //         coords[0],coords[1],coords[2],coords[3],rank);
                  return rank;
                }
                """
            )
        return self.lyncs_quda_cart_comms_map

    @property
    def comm(self):
        "Communicator in use by QUDA"
        return self._comm

    comm.setter(set_comm)

    @property
    def copy_struct(self):
        try:
            return self.lyncs_quda_copy_struct
        except AttributeError:
            cppdef(
                """
                template<typename T1, typename T2>
                void lyncs_quda_copy_struct(T1 &out, T2 &in) {
                  (T2 &) out = in;
                }
                """
            )
        return self.lyncs_quda_copy_struct

    def end_quda(self):
        if not self.initialized:
            raise RuntimeError("Quda has not been initialized")
        self.endQuda()
        self._comm = None
        self._initialized = False

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(f"QudaLib does not have attribute '{key}'")
        if not self.initialized:
            self.init_quda()
        if self.get_current_device() != self.device_id:
            super().__getattr__("cudaSetDevice")(self.device_id)
        return super().__getattr__(key)

    def __del__(self):
        if self.initialized:
            self.end_quda()


libs = []
if QUDA_MPI:
    from lyncs_mpi import lib as libmpi

    libs.append(libmpi)

PATHS = list(__path__)

headers = [
    "quda.h",
    "gauge_field.h",
    "gauge_tools.h",
    "gauge_force_quda.h",
    "gauge_update_quda.h",
    "dirac_quda.h",
]


lib = QudaLib(
    path=PATHS,
    header=headers,
    library=["libquda.so"] + libs,
    check="initQuda",
    include=CUDA_INCLUDE.split(";"),
    namespace="quda",
)


try:
    from pytest import fixture

    @fixture(scope="session")
    def fixlib():
        "A fixture to guarantee that in pytest lib is finalized at the end"
        if not lib.initialized:
            lib.init_quda()
        yield lib
        if lib.initialized:
            lib.end_quda()


except ImportError:
    pass
