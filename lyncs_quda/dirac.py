"""
Interface to quda::Dirac (dirac_quda.h)
"""

__all__ = [
    "Dirac",
]

from functools import wraps
from dataclasses import dataclass
from lyncs_cppyy import make_shared
from .gauge_field import gauge, GaugeField
from .spinor_field import spinor
from .lib import lib
from .enums import get_precision


@dataclass
class Dirac:
    gauge: GaugeField
    kappa: float = 1
    m5: float = 0
    Ls: int = 0
    csw: float = 0
    mu: float = 0
    epsilon: float = 0

    @property
    def type(self):
        "Type of the operator"
        if self.csw == 0:
            if self.mu == 0:
                return "WILSON"
            return "TWISTED_MASS"
        if self.mu == 0:
            return "CLOVER_WILSON"
        return "TWISTED_CLOVER"

    @property
    def quda_type(self):
        "Quda enum for quda dslash type"
        return getattr(lib, f"QUDA_{self.type}_DIRAC")

    @property
    def precision(self):
        return self.gauge.precision

    @property
    def dagger(self):
        "If the operator is daggered"
        return "NO"

    @property
    def quda_dagger(self):
        "Quda enum for if the operator is dagger"
        return getattr(lib, f"QUDA_DAG_{self.dagger}")

    @property
    def quda_params(self):
        params = lib.DiracParam()
        params.type = self.quda_type
        params.kappa = self.kappa
        params.m5 = self.m5
        params.Ls = self.Ls
        params.mu = self.mu
        params.epsilon = self.epsilon
        params.dagger = self.quda_dagger

        # Needs to prevent the gauge field to get destroyed
        self.quda_gauge = self.gauge.quda_field
        params.gauge = self.quda_gauge

        return params

    @property
    def quda_dirac(self):
        return make_shared(lib.Dirac.create(self.quda_params))

    def get_matrix(self, key="M"):
        "Returns the respective quda matrix."
        return DiracMatrix(self, key)

    def __call__(self, spinor_in, spinor_out=None, key="M"):
        return self.get_matrix(key)(spinor_in, spinor_out)

    @property
    def M(self):
        "Returns the matrix M"
        return self.get_matrix("M")

    @property
    def MdagM(self):
        "Returns the matrix MdagM"
        return self.get_matrix("MdagM")

    @property
    def MdagMLocal(self):
        "Returns the matrix MdagMLocal"
        return self.get_matrix("MdagMLocal")

    @property
    def Mdag(self):
        "Returns the matrix Mdag"
        return self.get_matrix("Mdag")

    @property
    def MMdag(self):
        "Returns the matrix MMdag"
        return self.get_matrix("MMdag")


GaugeField.Dirac = wraps(Dirac)(lambda *args, **kwargs: Dirac(*args, **kwargs))


class DiracMatrix:
    __slots__ = ["_dirac", "_gauge", "_matrix", "_key"]

    def __init__(self, dirac, key="M"):
        self._dirac = dirac.quda_dirac
        self._matrix = getattr(lib, "Dirac" + key)(self._dirac)
        self._gauge = dirac.quda_gauge
        self._key = key
        del dirac.quda_gauge

    def __call__(self, rhs, out=None):
        rhs = spinor(rhs)
        out = rhs.prepare(out)
        self.quda(out.quda_field, rhs.quda_field)
        return out

    @property
    def key(self):
        "The name of the natrix"
        return self._key

    @property
    def name(self):
        "The name of the operator"
        return self.quda.Type()

    @property
    def shift(self):
        "Shift to be added to the result"
        return self.quda.shift

    @shift.setter
    def shift(self, value):
        self.quda.shift = value

    @property
    def precision(self):
        "The precision of the operator (same as the gauge field)"
        return get_precision(self._gauge.Precision())

    @property
    def flops(self):
        "The flops of the operator"
        return self.quda.flops()

    @property
    def hermitian(self):
        "Whether is an hermitian operator"
        return bool(self.quda.hermitian())

    @property
    def is_wilson(self):
        "Whether is a Wilson-like operator"
        return bool(self.quda.isWilsonType())

    @property
    def is_staggered(self):
        "Whether is a staggered operator"
        return bool(self.quda.isStaggered())

    @property
    def is_dwf(self):
        "Whether is a domain wall fermions operator"
        return bool(self.quda.isDwf())

    @property
    def is_coarse(self):
        "Whether is a coarse operator"
        return bool(self.quda.isCoarse())

    @property
    def quda(self):
        return self._matrix
