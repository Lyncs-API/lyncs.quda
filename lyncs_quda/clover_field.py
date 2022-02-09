"""
Interface to clover_field.h
"""

__all__ = [
    "CloverField",
]

from functools import reduce
from time import time
from math import sqrt
import numpy
import cupy
import ctypes
from lyncs_cppyy import make_shared, lib as tmp, to_pointer, array_to_pointers
from lyncs_utils import prod
from .lib import lib, cupy
from .lattice_field import LatticeField
from .gauge_field import GaugeField
from .time_profile import default_profiler
from .enums import QudaParity

# TODO list
# We want dimension of (cu/num)py array to reflect parity and order
# For float64, native order is (2,72,-1,2) where the (left/right)-most 2 is parity/real-imag
# For flaot32, native order is (2,36,-1,4) where the (left/right)-most 2 is parity/real-imag+half_of_color_spin


class CloverField(LatticeField):
    """
    Mimics the quda::CloverField object
     This is designed as an intermediary to QUDA CloverField class
     so that it should have 1-to-1 correspondence to an QUDA instance.
    Note:
     * This class stores the corresponding gauge field in its "field" attribute
     * direct & inverse fields are both allocated upon initialization
     * Only rho is mutable.  To change other params, a new instance should be created
     * QUDA convention for clover field := 1+i ( kappa csw )/4 sigma_mu,nu F_mu,nu (<-sigma_mu,nu: spinor tensor)
    """

    def __init__(
        self,
        field,
        *args,
        csw=0,
        twisted=False,
        mu2=0,
        rho=0,
        computeTrLog=False,
        **kwards,
    ):
        if not isinstance(field, GaugeField):
            field = GaugeField(field)
        if field.geometry is "VECTOR":
            self._fmunu = field.compute_fmunu()
        elif field.geometry is "TENSOR":
            self._fmunu = field
        else:
            raise TypeError(
                "The input GaugeField instabce needs to be of geometry VECTOR or TENSOR"
            )
        super().__init__(self._fmunu.field, comm=self._fmunu.comm)

        # QUDA clover field inherently works with real's not with complex's (c.f., include/clover_field_order.h)
        idof = 72  # int((self._fmunu.ncol*self._fmunu.ndims)**2/2)
        prec = self._fmunu.precision

        self._direct = (
            False  # Here, it is a flag to indicate whether the field has been computed
        )
        self._inverse = (
            False  # Here, it is a flag to indicate whether the field has been computed
        )

        new = self._fmunu.new
        self._clover = new(dofs=(idof,), dtype=prec)
        self._cloverInv = new(dofs=(idof,), dtype=prec)
        # norm used only when self.quda_precision = HALF or QUARTER?
        # Apparently, stride for clover field needs to be QUDA_FULL_SITE_SUBSET for clover fields,
        #  suggested by CloverFieldParam::CloverFieldParam(const CloverField &a) in clover_field.cpp
        self._norm = new(dofs=(2,), dtype=numpy.float32)  # 2 for chirality
        self._normInv = new(dofs=(2,), dtype=numpy.float32)  # 2 for chirality

        self._csw = csw
        self._twisted = twisted
        self._mu2 = mu2
        self._rho = rho
        self.computeTrLog = computeTrLog

        self._quda_clover = None

    # shape, dofs, dtype, iscomlex, isreal are overwriten to report their values for the clover field, instead of _fmunu

    @property
    def shape(self):
        "Shape of the clover field"
        return self._clover.shape

    @property
    def dofs(self):
        "Shape of the per-site degrees of freedom"
        return self._clover.dofs

    @property
    def dtype(self):
        "Clover field data type"
        return self._clover.dtype

    @property
    def iscomplex(self):
        "Whether the clover field dtype is complex"
        return self._clover.iscomplex

    @property
    def isreal(self):
        "Whether the clover field dtype is real"
        return self._clover.isreal

    @property
    def csw(self):
        return self._csw

    @property
    def twisted(self):
        return self._twisted

    @property
    def mu2(self):
        return self._mu2

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, val):
        if not isinstance(val, float):
            if isinstance(val, int):
                val = float(val)
            else:
                raise TypeError("rho value should be a real number")
        if self._quda_clover is not None:
            self._quda_clover.setRho(val)
        self._rho = val

    @property
    def order(self):
        "Data order of the field"
        if self.precision is "double":
            return "FLOAT2"
        else:
            return "FLOAT4"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_CLOVER_ORDER")

    @property
    def quda_params(self):
        "Returns an instance of quda::CloverFieldParams"
        params = lib.CloverFieldParam()
        lib.copy_struct(params, super().quda_params)
        params.direct = True
        params.inverse = True
        params.clover = to_pointer(self._clover.ptr)
        params.norm = to_pointer(self._norm.ptr)
        params.cloverInv = to_pointer(self._cloverInv.ptr)
        params.normInv = to_pointer(self._normInv.ptr)
        params.csw = self.csw
        params.twisted = self.twisted
        params.mu2 = self.mu2
        params.rho = self.rho
        params.order = self.quda_order
        params.create = lib.QUDA_REFERENCE_FIELD_CREATE
        params.location = self.quda_location
        return params

    @property
    def quda_field(self):
        "Returns an instance of quda::(cpu/cuda)CloverField for QUDA_(CPU/CUDA)_FIELD_LOCATION"
        if self._quda_clover is None:
            self.activate()
            self._quda_clover = make_shared(lib.CloverField.Create(self.quda_params))
        return self._quda_clover

    def is_native(self):
        "Whether the field is native for Quda"
        return lib.clover.isNative(self.quda_order, self.quda_precision)

    @property
    def ncol(self):
        # The value is hard-coded to be 3 in the constructor found in clover_field.cpp & include/kernel/clover_invert.cuh
        return self.quda_field.Ncolor()

    @property
    def nspin(self):
        # The value is hard-coded to be 4 in the constructor found in clover_field.cpp & include/kernel/clover_invert.cuh
        return self.quda_field.Nspin()

    @property
    def clover_field(self):
        # Note: This is a kind reminder that QUDA internally applies a normalization factor of 1/2 in clover field.
        if not self._direct:
            lib.computeClover(self.quda_field, self._fmunu.quda_field, self.csw)
            self._direct = True
        return self._clover.field

    @property
    def clover_norm(self):
        self.clover_field
        return self._norm.field

    @property
    def inverse_field(self):
        if not self._inverse:
            self.clover_field
            lib.cloverInvert(self.quda_field, self.computeTrLog)
            self._inverse = True
        return self._cloverInv.field

    @property
    def inverse_norm(self):
        self.inverse_field
        return self._normInv.field

    @property
    def trLog(self):
        if self._inverse:
            # separation into the following two lines is necessary
            arr = self.quda_field.TrLog()
            arr.reshape((2,))
            return numpy.frombuffer(arr, dtype=self.dtype, count=2)
        return numpy.zeros((2,), dtype="double")

    @property
    def Bytes(self):
        return self.quda_field.Bytes()

    @property
    def Norm_bytes(self):
        return self.quda_field.NormBytes()

    """
    Note for norm(1,2) & abs_(max,min)
     When computing these quantities, QUDA internally first cast CloverField into
      basically a pointer to comlex<FLOAT>.  
     This will ignore representation of internal d.o.f.s of clover term explained in
      include/clover_field_order.h.  
     Also, when compuing them, QUDA removes the factor of 1/2, present in fields.
    """

    def norm1(self, inverse=False):
        "Computes the L1 norm of the field"
        return self.quda_field.norm1(inverse)

    def norm2(self, inverse=False):
        "Computes the L2 norm of the field"
        return self.quda_field.norm2(inverse)

    def abs_max(self, inverse=False):
        "Computes the absolute maximum of the field (Linfinity norm)"
        return self.quda_field.abs_max(inverse)

    def abs_min(self, inverse=False):
        "Computes the absolute minimum of the field"
        return self.quda_field.abs_min(inverse)

    def computeCloverForce(self, coeff):
        """
        Compute the force contribution from the solver solution fields
        """
        # should be placed in GaugeField and use self.quda_field?
        # should take arrays of SpinorFields and put them in std::vector<ColorSpinorField*>
        # turn an array of doubles to std::vector<double>
        pass

    def computeCloverSigmaOprod(self):
        # should be in SpinorField?
        # should take arrays of SpinorFields and put them in std::vector<ColorSpinorField*>
        # turn an array of doubles to std::vector<double>
        pass

    def computeCloverSigmaTrace(self, coeff=1.0):
        """
        Compute the matrix tensor field necessary for the force calculation from
        the clover trace action.  This computes a tensor field [mu,nu].

        @param coeff  Scalar coefficient multiplying the result (e.g., stepsize)
        """
        out = self._fmunu.new()
        lib.computeCloverSigmaTrace(out.quda_field, self.quda_field, coeff)
        return out

    def cloverDerivative(self, oprod, parity: QudaParity, coeff=1.0):
        """
        Compute the derivative of the clover matrix in the direction
        mu,nu and compute the resulting force given the outer-product field

        @param coeff Multiplicative coefficient (e.g., clover coefficient)
        @param parity The field parity we are working on
        """
        # should be placed in GaugeField and use self.quda_field ?
        pass
