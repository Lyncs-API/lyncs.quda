"""
Interface to quda::Dirac (dirac_quda.h)
"""

__all__ = [
    "Dirac",
]

from functools import wraps
from dataclasses import dataclass, field, InitVar
from numpy import sqrt
from dataclasses import dataclass
from lyncs_cppyy import make_shared, nullptr
from .gauge_field import gauge, GaugeField
from .clover_field import CloverField
from .spinor_field import spinor
from .lib import lib
from .enums import QudaPrecision


@dataclass
class Dirac:

    # For QUDA DiracParam class:
    gauge: GaugeField  # TODO: check acceptable geometry of gauge as an argument to QUDA Dirac class
    _gauge: GaugeField = field(
        init=False, repr=False
    )  # Do not define __getattribute__ to make this work, unless designed wisely
    coarse_clover: GaugeField = None
    coarse_clover_inv: GaugeField = None
    coarse_precond: GaugeField = None
    kappa: float = 1
    m5: float = 0
    Ls: int = 1
    mu: float = 0
    epsilon: float = 0

    # For LYNCS CloverField class
    coeff: float = 0
    rho: float = 0
    computeTrLog: bool = False

    clover: InitVar[CloverField] = None
    _clover: CloverField = field(
        init=False, repr=False
    )  # Do not define __getattribute__ to make this work, unless designed wisely

    def __post_init__(
        self, clover
    ):  # this is to overcome default overwritten as a property
        if clover is self.__dataclass_fields__["clover"].default:
            clover = None

        self.clover = clover

    @property
    def gauge(self):
        return self._gauge

    @gauge.setter
    def gauge(self, gauge: GaugeField):
        self._gauge = gauge
        self.clover = None

    @property
    def clover(self):
        return self._clover

    @clover.setter
    def clover(self, clover: CloverField):
        if clover is not None and not isinstance(clover, CloverField):
            raise TypeError(
                "This class expects its clover argument to be an instance of CloverField"
            )

        self._clover = clover

        if clover is not None:
            self.coeff = clover.coeff
            self.mu = sqrt(clover.mu2)
            self.rho = clover.rho
            self.computeTrLog = clover.computeTrLog

    # ? do we want to support more methods for Dirac?

    # TODO: Support more Dirac types
    #   Unsupported: PC for the below types, Hasenbusch for clover types
    #                DomainWall(4D/PC), Mobius(PC/Eofa), (Improved)Staggered(KD/PC), CoarsePC, GaugeLaplace(PC), GaugeCovDev
    @property
    def type(self):
        "Type of the operator"
        if self.gauge.is_coarse:
            return "COARSE"
        if self.coeff == 0:
            if self.mu == 0:
                return "WILSON"
            return "TWISTED_MASS"
        if self.mu == 0:
            return "CLOVER"
        return "TWISTED_CLOVER"

    @property
    def quda_type(self):
        "Quda enum for quda dslash type"
        return getattr(lib, f"QUDA_{self.type}_DIRAC")

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

        # Needs to prevent the gauge field to get destroyed
        self.quda_gauge = self.gauge.quda_field
        params.gauge = self.quda_gauge

        if self.coeff != 0.0 and not self.gauge.is_coarse:
            if self.clover is None:
                self.clover = CloverField(
                    self.gauge,
                    coeff=self.coeff,
                    twisted=(self.mu != 0),
                    tf =("SINGLET" if "TWISTED" in self.type else "NO"),
                    mu2=self.mu**2,
                    rho=self.rho,
                    computeTrLog=self.computeTrLog,
                )
                self.clover.clover_field
            params.clover = self.clover.quda_field

        return params

    @property
    def quda_dirac(self):
        if not self.is_coarse:
            return make_shared(lib.Dirac.create(self.quda_params))
        # Creating coarse operator
        #  This constcutor seems to rely on initializeLazy when performing M, MdagM, etc.
        #  initializeLazy seems to assume that if gauge is allocated on LOCATION,
        #  then coarse_* is alredy allocated on LOCATION too if we use the follwing constructor
        return lib.DiracCoarse(
            self.quda_params,
            self.gauge.cpu_field,
            self.coarse_clover.cpu_field, #if self.coarse_clover else nullptr,  # if we give nullptr, (ok if gauge's location=CUDA) then cpuClass is internally allocated if an operand is located on cpu, but this is not accessible from Python side.  no accessor in DiracCoarse
            self.coarse_clover_inv.cpu_field if self.coarse_clover_inv else nullptr,
            self.coarse_precond.cpu_field if self.coarse_precond else nullptr,
            self.gauge.gpu_field,
            self.coarse_clover.gpu_field if self.coarse_clover else nullptr,
            self.coarse_clover_inv.gpu_field if self.coarse_clover_inv else nullptr,
            self.coarse_precond.gpu_field if self.coarse_precond else nullptr,
        )

    def get_matrix(self, key="M"):
        "Returns the respective quda matrix."
        return DiracMatrix(self, key)

    def __call__(self, spinor_in, spinor_out=None, key="M"):
        return self.get_matrix(key)(spinor_in, spinor_out)

    # TODO: Support more functors: Dagger, G5M

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
        self._dirac = dirac.quda_dirac  # necessary?
        self._matrix = getattr(lib, "Dirac" + key)(self._dirac)
        self._gauge = dirac.quda_gauge  # necessary?
        self._key = key
        del (
            dirac.quda_gauge
        )  # ? why?  if an instance of Dirac tries to construct DiracMatrix next time, it will fail

    def __call__(self, rhs, out=None):
        rhs = spinor(rhs)
        out = rhs.prepare(out)
        self.quda(out.quda_field, rhs.quda_field)
        return out

    # TODO: Support int getStencilSteps(); QudaMatPCType getMatPCType();

    @property
    def key(self):
        "The name of the matrix"
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

    # why is it here?
    @property
    def precision(self):
        "The precision of the operator (same as the gauge field)"
        return QudaPrecision[self._gauge.Precision()]

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
