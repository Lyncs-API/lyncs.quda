"""
Interface to quda::Dirac (dirac_quda.h)
"""

__all__ = [
    "Dirac",
]

from functools import wraps
from dataclasses import dataclass, field
from numpy import sqrt
from dataclasses import dataclass
from typing import Union, List
from lyncs_cppyy import make_shared, nullptr
from .gauge_field import gauge, GaugeField
from .clover_field import CloverField
from .spinor_field import spinor
from .lib import lib
from .enums import QudaPrecision

@dataclass(frozen=True)
class Dirac:

    # For QUDA DiracParam class:
    gauge: GaugeField  # TODO: check acceptable geometry of gauge as an argument to QUDA Dirac class
    clover: Union[CloverField, GaugeField] = None
    coarse_clover_inv: GaugeField = None
    coarse_precond: GaugeField = None
    kappa: float = 1
    m5: float = 0
    Ls: int = 1
    csw: float = 0
    mu: float = 0
    epsilon: float = 0
    matPCtype: str = "INVALID"
    # For QUDA CloverField class:
    rho: float = 0
    computeTrLog: bool = False

    _quda: ... = field(init=False, repr=False, default=None)

    # TODO: Support more Dirac types
    #   Unsupported: DomainWall(4D/PC), Mobius(PC/Eofa), (Improved)Staggered(KD/PC), GaugeLaplace(PC), GaugeCovDev
    @property
    def type(self):
        "Type of the operator"
        PC = "PC" if self.matPCtype != "INVALID" else ""
        if self.gauge.is_coarse:
            return "COARSE" + PC
        if self.csw == 0:
            if self.mu == 0:
                return "WILSON" + PC
            return "TWISTED_MASS" + PC
        if self.mu == 0:
            return "CLOVER" + PC
        return "TWISTED_CLOVER" + PC

    @property
    def quda_type(self):
        "Quda enum for quda dslash type"
        return getattr(lib, f"QUDA_{self.type}_DIRAC")

    @property
    def quda_matPCtype(self):
        if self.matPCtype is None:
            self.matPCtype = "INVALID"
        return getattr(lib, f"QUDA_MATPC_{self.matPCtype}")
    
    @property
    def is_coarse(self):
        "Whether is a coarse operator"
        return self.type == "COARSE"

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
        params.matpcType = self.quda_matPCtype
        
        # Needs to prevent the gauge field to get destroyed
        #  now we store QUDA gauge object in _quda, but it
        #  still can be a problem if we create DiracMatrix object
        #  using a Dirac object, and if this Dirac object and
        #  GaugeField object inside of it are both gone.
        # Possible solution:
        #  add an attribute for Dirac object in DiracMatrix object
        # self.quda_gauge = self.gauge.quda_field as well as _quda in Dirac
        params.gauge = self.gauge.quda_field

        if self.csw != 0.0 and not self.gauge.is_coarse:
            if self.clover is None:
                object.__setattr__(
                    self,
                    "clover",
                    CloverField(
                        self.gauge,
                        coeff=self.kappa * self.csw,
                        twisted=(self.mu != 0),
                        tf=("SINGLET" if "TWISTED" in self.type else "NO"),
                        mu2=self.mu**2,
                        eps2=self.epsilon**2,
                        rho=self.rho,
                        computeTrLog=self.computeTrLog,
                    ),
                )  # well, this is a hack, but this dataclass is now frozen. so...
                self.clover.clover_field
            params.clover = self.clover.quda_field

        return params

    @property
    def quda_dirac(self):
        if not self.is_coarse:
            if self._quda is None:
                object.__setattr__(self, "_quda", make_shared(lib.Dirac.create(self.quda_params)))
            return self._quda
            
        #  This constcutor seems to rely on initializeLazy when performing M, MdagM, etc.
        #  initializeLazy seems to assume that if gauge is allocated on LOCATION,
        #  then coarse_* is alredy allocated on LOCATION too if we use the follwing constructor
        # Note
        #  * clover_inv is necessary when doing clover inversion or applying prec coarse op
        #  * coarse_precond is necessary when applying prec coarse op

        assert self.clover is not None and isinstance(self.clover, GaugeField)
        if self._quda is None:
            object.__setattr__(
                self,"_quda",
                make_shared(lib.DiracCoarse(
                    self.quda_params,
                    self.gauge.cpu_field,
                    self.clover.cpu_field,
                    self.coarse_clover_inv.cpu_field
                    if self.coarse_clover_inv is not None
                    else nullptr,
                    self.coarse_precond.cpu_field
                    if self.coarse_precond is not None
                    else nullptr,
                    self.gauge.gpu_field,
                    self.clover.gpu_field,
                    self.coarse_clover_inv.gpu_field
                    if self.coarse_clover_inv is not None
                    else nullptr,
                    self.coarse_precond.gpu_field
                    if self.coarse_precond is not None
                    else nullptr,
                )))
        return self._quda

    def Dslash(self, spinor_in, spinor_out=None, parity="EVEN"):
        # change below to use Even and Odd
        self._quda.Dslash(spinor_out.quda_field, spinor_in.quda_field,
                          getattr(lib, f"QUDA_{parity}_PARITY"))
        
    def get_matrix(self, key="M"):
        "Returns the respective quda matrix."
        return DiracMatrix(self, key)

    def __call__(self, spinor_in, spinor_out=None, key="M"):
        return self.get_matrix(key)(spinor_in, spinor_out)

    # TODO: Support more functors: Dagger, G5M

    #? DiracMatrix simply calls the corresponding method
    #  of Dirac with the same name..., e.g., DiracM() -> Dirac.M()
    #  Methos below calls Dirac.M* via DiracMatrix, which
    #  seems unnecessary detour
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
    __slots__ = ["dirac", "_prec", "_matrix", "_key"]

    def __init__(self, dirac, key="M"):
        self.dirac = dirac 
        self._matrix = getattr(lib, "Dirac" + key)(self.dirac.quda_dirac)
        self._key = key
        self._prec = dirac.precision

    def __call__(self, rhs, out=None):
        rhs = spinor(rhs)
        out = rhs.prepare(out)
        self.quda(out.quda_field, rhs.quda_field)
        return out


    @property
    def key(self):
        "The name of the matrix"
        return self._key

    @property
    def name(self):
        "The type of the operator"
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
        return QudaPrecision[self._prec]

    @property
    def mat_PC(self):
        return QudaMatPCType[self.quda.getMatPCType()]
    
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
