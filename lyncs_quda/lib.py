"""
Loading the QUDA library
"""
# pylint: disable=C0103

__all__ = [
    "lib",
    "MPI",
    "PATHS",
]

import atexit
from os import environ
from pathlib import Path
from array import array
from math import prod
from appdirs import user_data_dir
from lyncs_cppyy import Lib, cppdef
from lyncs_utils import lazy_import
from . import __path__
from .config import QUDA_MPI, QUDA_GITVERSION, QUDA_PRECISION, QUDA_RECONSTRUCT

cupy = lazy_import("cupy")


class QudaLib(Lib):
    "Adds additional enviromental control required by QUDA"

    __slots__ = [
        "_initialized",
        "_device_id",
        "_comm",
        "MPI",
    ]

    def __init__(self, *args, **kwargs):
        self._initialized = False
        self._device_id = 0
        self._comm = None
        if not self.tune_dir:
            self.tune_dir = user_data_dir("quda", "lyncs") + "/" + QUDA_GITVERSION
        super().__init__(*args, **kwargs)

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
        if self._device_id == -1:
            self.init_quda()
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
            raise TypeError(f"Unsupported type for device ID: {type(value)}")
        self._device_id = value

    def get_current_device(self):
        return cupy.cuda.runtime.getDevice()

    def get_device_count(self):
        return cupy.cuda.runtime.getDeviceCount()

    def init_quda(self, dev=None):
        # ASSUME: self.comm has been set in case of non-trivial MPI job
        if self.initialized:
            raise RuntimeError("Quda already initialized")
        # At first, we set initialized to True to avoid recursion
        #  btwn __getattr__ and init_quda() [setMP..., lyncs_quda_comm...]
        self._initialized = True
        if not self.loaded:
            self.load()
        if self.tune_dir:
            Path(self.tune_dir).mkdir(parents=True, exist_ok=True)
        if QUDA_MPI and self.comm is None:
            self.set_comm()
        if QUDA_MPI:
            comm = get_comm(self.comm)
            comm_ptr = self._comm_ptr(comm)

            self.setMPICommHandleQuda(comm_ptr)
            # self.initQUDA(0,comm_ptr)
            dims = array("i", self.comm.dims)
            # self.initQUDA(1,4, dims, self._comms_map, comm_ptr)
            self.initCommsGridQuda(4, dims, self._comms_map, comm_ptr)

        if dev is None:
            dev = self._device_id
        # self.initQUDA(2,dev)
        self.initQuda(dev)
        self._device_id = self.get_current_device()
        atexit.register(self.end_quda)

    # for profiling
    def initQUDA(self, val, *args):
        if val == 0:
            self.setMPICommHandleQuda(*args)
        elif val == 1:
            self.initCommsGridQuda(*args)
        elif val == 2:
            self.initQuda(*args)

    def set_comm(self, comm=None, procs=None):
        # NOTE: comm==None && procs==None taken as indication of single rank MPI job
        # NOTE: set_comm should be called explicitly when using multiple ranks
        #        before using any of lyncs_quda objects (e.g., creating such objects)
        # TODO:
        #  set DEFAULT_COMM
        #  provide getter and setter for it
        #  allow dynamic management of the lib's comm

        if comm is not None and not QUDA_MPI:
            raise RuntimeError("Quda has not been compiled with MPI")
        if comm is None and not QUDA_MPI:
            return
        if comm is None:
            if procs is None or prod(procs) == 1:
                comm = MPI.COMM_SELF.Create_cart((1, 1, 1, 1))
            else:
                comm = MPI.COMM_WORLD.Create_cart(procs)
        if not isinstance(comm, MPI.Cartcomm):
            raise TypeError("comm expected to be a Cartcomm")
        if (
            self._comm is not None
            and MPI.Comm.Compare(comm, self._comm) <= MPI.CONGRUENT
        ):
            return
        if comm.ndim != 4:
            raise ValueError("comm expected to be a 4D Cartcomm")
        # not supported at the moment
        # if self._comm is not None: #original
        # when ending and starting over QUDA, strange things happen
        # self.end_quda()
        self._comm = comm
        if not self.initialized and self.comm.size > 1:
            self._device_id = -1

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

    @property
    def copy_strings(self):
        try:
            return self.lyncs_quda_copy_strings
        except AttributeError:
            cppdef(
            """
            void lyncs_quda_copy_strings(QudaMultigridParam &mp, std::vector<std::string> in, bool is_infile){
            //void lyncs_quda_copy_strings(char (&out)[][256], std::vector<std::string> in){
              int n = in.size();
              for (int i=0; i<n; i++){
                std::string s = in[i];
                if (is_infile) strcpy(mp.vec_infile[i], s.c_str());
                else strcpy(mp.vec_outfile[i], s.c_str()); 
                //strcpy(out[i], s.c_str());
              }                                     
            }
            """
            )
        return self.lyncs_quda_copy_strings
    
    def save_tuning(self):
        if self.tune_enabled:
            self.saveTuneCache()

    def end_quda(self):
        if not self.initialized:
            raise RuntimeError("Quda has not been initialized")
        self.save_tuning()
        self.endQuda()
        self._comm = None
        self._initialized = False
        atexit.unregister(self.end_quda)

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(f"QudaLib does not have attribute '{key}'")
        if not self.initialized:
            self.init_quda()
        if self._device_id >= 0 and self.get_current_device() != self.device_id:
            super().__getattr__("cudaSetDevice")(self.device_id)
        return super().__getattr__(key)

    def __del__(self):
        if self.initialized:
            self.end_quda()


libs = []
if QUDA_MPI:
    from lyncs_mpi import lib as libmpi, get_comm, MPI

    libs.append(libmpi)
else:
    MPI = None

PATHS = list(__path__)

headers = [
    "quda.h",
    "comm_quda.h",
    "gauge_field.h",
    "gauge_tools.h",
    "gauge_path_quda.h",
    "gauge_update_quda.h",
    "clover_field.h",
    "dirac_quda.h",
    "invert_quda.h",
    "blas_quda.h",
    "multigrid.h",
    "evenodd.h",
    "array.h",
    "momentum.h",
    "tune_quda.h",
]


lib = QudaLib(
    path=PATHS,
    header=headers,
    library=["libquda.so"] + libs,
    namespace=["quda", "lyncs_quda"],
    defined={"QUDA_PRECISION": QUDA_PRECISION, "QUDA_RECONSTRUCT": QUDA_RECONSTRUCT},
)
lib.MPI = MPI

# used?
try:
    from pytest import fixture

    @fixture(scope="session")
    def fixlib():
        "A fixture to guarantee that in pytest lib is finalized at the end"
        if QUDA_MPI and MPI.COMM_WORLD.Get_size() > 1:
            pass
        elif not lib.initialized:
            lib.init_quda()
        yield lib
        if lib.initialized:
            lib.end_quda()

except ImportError:
    pass
