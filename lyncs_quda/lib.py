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
from lyncs_cppyy import Lib
from . import __path__
from .config import QUDA_MPI, GITVERSION, CUDA_INCLUDE


class QudaLib(Lib):
    "Adds additional enviromental control required by QUDA"

    __slots__ = [
        "_initialized",
        "_device_id",
    ]

    def __init__(self, *args, **kwargs):
        self._initialized = False
        self._device_id = 0
        if not self.tune_dir:
            self.tune_dir = user_data_dir("quda", "lyncs") + "/" + GITVERSION
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
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        if value == self.device_id:
            return
        if self.initialized:
            raise RuntimeError(
                f"device_id cannot be changed: current={self.device_id}, given={value}"
            )
        self._device_id = value

    def get_current_device(self):
        dev = array("i", [0])
        super().__getattr__("cudaGetDevice")(dev)
        return dev[0]

    def initQuda(self, dev=None):
        if self.initialized:
            raise RuntimeError("Quda already initialized")
        if self.tune_dir:
            Path(self.tune_dir).mkdir(parents=True, exist_ok=True)
        if dev is None:
            if QUDA_MPI:
                dev = -1
            else:
                dev = self.device_id
        super().__getattr__("initQuda")(dev)
        self.device_id = self.get_current_device()
        self._initialized = True

    def endQuda(self):
        if not self.initialized:
            raise RuntimeError("Quda has not been initialized")
        super().__getattr__("endQuda")()
        self._initialized = False

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(f"QudaLib does not have attribute '{key}'")
        if not self.initialized:
            self.initQuda()
        if self.get_current_device() != self.device_id:
            super().__getattr__("cudaSetDevice")(self.device_id)
        return super().__getattr__(key)

    def __del__(self):
        if self.initialized:
            self.endQuda()


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
]


lib = QudaLib(
    path=PATHS,
    header=headers,
    library=["libquda.so"] + libs,
    check="initQuda",
    include=CUDA_INCLUDE.split(";"),
    namespace="quda",
)
