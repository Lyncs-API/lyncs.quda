"""
Interface to clover_field.h
"""

__all__ = [
    "CloverField",
]

import numpy

from lyncs_cppyy import make_shared, to_pointer
from .lib import lib, cupy
from .lattice_field import LatticeField
from .gauge_field import GaugeField
from .enums import QudaParity

# TODO list
# We want dimension of (cu/num)py array to reflect parity and order
# For float64, native order is (2,72,-1)   where the (left/right)-most index is for parity/real-imag
# For flaot32, native order is (2,36,-1,4) where the (left/right)-most index is for parity/real-imag+half_of_color_spin
# make compatible with QUDA_CLOVER_DYNAMIC=ON


class CloverField(LatticeField):
    """
    Mimics the quda::CloverField object
     This is designed as an intermediary to QUDA CloverField class
     so that it should have 1-to-1 correspondence to an QUDA instance.
    Note:
     * This class stores the corresponding gauge field in its "field" attribute
        to make except-clause of copy() work
     * direct & inverse fields are both allocated upon initialization
     * Only rho is mutable.  To change other params, a new instance should be created
     * QUDA convention for clover field := 1+i ( kappa csw )/4 sigma_mu,nu F_mu,nu (<-sigma_mu,nu: spinor tensor)
    """

    def __init__(
        self,
        fmunu,
        coeff=0.0,
        twisted=False,
        mu2=0,
        tf="NO",
        eps2=0,
        rho=0,
        computeTrLog=False,
    ):
        # ? better to store fmunu.quda_field to _fmunu -> import gauge_tensor to be used in some methods
        # ? better to put clover into self.field -> need walk-arond to make copy() work
        if not isinstance(fmunu, GaugeField):
            fmunu = GaugeField(fmunu)

        if fmunu.geometry == "VECTOR":
            self._fmunu = fmunu.compute_fmunu()
        elif fmunu.geometry == "TENSOR":
            self._fmunu = fmunu
        else:
            raise TypeError(
                "The input GaugeField instabce needs to be of geometry VECTOR or TENSOR"
            )
        super().__init__(self._fmunu.field, comm=self._fmunu.comm)

        # QUDA clover field inherently works with real's not with complex's (c.f., include/clover_field_order.h)

        idof = int((self._fmunu.ncol * self._fmunu.ndims) ** 2 / 2)
        prec = self._fmunu.precision
        self._direct = (
            False  # Here, it is a flag to indicate whether the field has been computed
        )
        self._inverse = (
            False  # Here, it is a flag to indicate whether the field has been computed
        )

        new = lambda idof: LatticeField.create(
            self._fmunu.lattice,
            dofs=(idof,),
            dtype=prec,
            device=self._fmunu.device,
            empty=True,
        )
        self._clover = new(idof)
        self._cloverInv = new(idof)
        self.coeff = coeff
        self._twisted = twisted
        self._twist_flavor = tf
        self._mu2 = mu2
        self._eps2 = eps2
        self._rho = rho
        self.computeTrLog = computeTrLog

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

    # naming suggestion: native_view? default_* saved for dofs+lattice?
    def default_view(self):
        N = 1 if self.order == "FLOAT2" else 4
        shape = (2,)  # even-odd
        shape += (self.dofs[0] // N, -1, N)

        return self.field.view().reshape(shape)

    @property
    def twisted(self):
        return self._twisted

    @property
    def twist_flavor(self):
        return self._twist_flavor

    @property
    def mu2(self):
        return self._mu2

    @property
    def eps2(self):
        return self._eps2

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
        if self._quda is not None:
            self._quda.setRho(val)
        self._rho = val

    @property
    def order(self):
        "Data order of the field"
        if self.precision == "double":
            return "FLOAT2"
        return "FLOAT4"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_CLOVER_ORDER")

    @property
    def quda_params(self):
        "Returns an instance of quda::CloverFieldParams"
        """
        Remarks:
         computeClover assumes param.reconstruct True 
          if compiled with QUDA_CLOVER_RECONSTRUCT=ON
          else False, which is default
         So to use this function to construct clover field, we need to
          stick with the default value
         When compiled with QUDA_CLOVER_DYNAMIC=ON, clover field is used as 
          an alias to inverse.  not really sure what this is, but does 
          not work properly when reconstruct==True
        """
        params = lib.CloverFieldParam()
        lib.copy_struct(params, super().quda_params)
        params.inverse = True
        params.clover = to_pointer(self._clover.ptr)
        params.cloverInv = to_pointer(self._cloverInv.ptr)
        params.coeff = self.coeff
        params.twisted = self.twisted
        params.twist_flavor = getattr(lib, f"QUDA_TWIST_{self.twist_flavor}")
        params.mu2 = self.mu2
        params.epsilon2 = self.eps2
        params.rho = self.rho
        params.order = self.quda_order
        params.create = lib.QUDA_REFERENCE_FIELD_CREATE
        params.location = self.quda_location
        return params

    @property
    def quda_field(self):
        "Returns an instance of quda::(cpu/cuda)CloverField for QUDA_(CPU/CUDA)_FIELD_LOCATION"
        self.activate()
        if self._quda is None:
            self._quda = make_shared(lib.CloverField.Create(self.quda_params))
        return self._quda

    @property
    def clover_field(self):
        # Note: This is a kind reminder that QUDA internally applies a normalization factor of 1/2 in clover field.
        if not self._direct:
            lib.computeClover(self.quda_field, self._fmunu.quda_field, self.coeff)
            self._direct = True
        return self._clover.field

    @property
    def inverse_field(self):
        if not self._inverse:
            self.clover_field
            lib.cloverInvert(self.quda_field, self.computeTrLog)
            self._inverse = True
        return self._cloverInv.field

    @property
    def trLog(self):
        if self._inverse and self.computeTrLog:
            # separation into the following two lines is necessary
            arr = self.quda_field.TrLog().data  # can simply use tuple?
            arr.reshape((2,))
            return numpy.frombuffer(arr, dtype="double", count=2)
        return None

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
    def reconstruct(self):
        return self.quda_field.Reconstruct()

    @property
    def csw(self):
        return self.quda_field.Csw()

    # used when reconstructing the clover field
    @property
    def diagonal(self):
        return self.quda_field.Diagonal()

    @diagonal.setter
    def diagonal(self, val: float):
        self.quda_field.Diagonal(val)

    @property
    def Bytes(self):
        return self.quda_field.Bytes()

    @property
    def total_bytes(self):
        return self.quda_field.TotalBytes()

    @property
    def compressed_block_size(self):
        return self.quda_field.compressed_block_size()

    def max_elem(self, inverse=False):  # ? don't know what this is
        return self.quda_field.max_element(inverse)

    """
    Note for norm(1,2) & abs_(max,min)
     For norm1, norm2,   
       QUDA turns CloverField into basically a pointer to complex numbers 
        so that 12 entries of .5 becomes 6 entries of 0.5*(1+I)  
     For abs_max, abs_min,
       QUDA treats fields as fields of reals and finds L-inf norm of the sequence 
     Norm factor
       If not QUDA_PACKED_CLOVER_ORDER, norm factor of 2 is applied to the sequence
       If QUDA_PACKED_CLOVER_ORDER, the factor == 1 
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

    # ? may not be necessary
    def backup(self):
        "Back up clover field (& its inverse if computed) onto CPU"
        self.quda_field.backup()

    # ? may not be necessary
    def restore(self):
        "Restore clover field (& its inverse if computed) from CPU to GPU"
        self.quda_field.restore()

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
