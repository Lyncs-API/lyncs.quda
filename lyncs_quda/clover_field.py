"""
Interface to clover_field.h
"""

__all__ = [
    "CloverField",
]

import numpy
from cppyy.gbl.std import vector

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
     *  so that sigma_mu,nu = i[g_mu, g_nu], F_mu,nu = (Q_mu,nu - Q_nu,mu)/8 (1/2 is missing from sigma_mu,nu)
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
        # ASSUME: coeff = kappa*csw wihout a normalization factor of 1/4 or 1/32 (suggested in interface_quda.cpp)
        # ? better to store fmunu.quda_field to _fmunu -> import gauge_tensor to be used in some methods
        # ? better to put clover into self.field -> need walk-around to make copy() work
        if not isinstance(fmunu, GaugeField):
            fmunu = GaugeField(fmunu)
        self._fmunu = fmunu.compute_fmunu()
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

    def computeCloverForce(self, gauge, force, D, vxs, vps, mult=2, coeffs=None):
        # contribution from TrLn A and U dA/dU are similar except for the factor in the middle of the expression
        # i.e., they have the form: GL A GR where A is different
        # so computeCloverSigmaTrace and computeCloverSigmaOprod computes A for each and add them together.
        # cloverDerivative computes and multiplies GL and GR without csw*kappa*normalizarion_factor

        # D: Python Dirac class
        # mult: Number fermions this bilinear reresents
        # coeff: Array of residues for each contribution (multiplied by stepsize)
        #  interface_quda.cpp suggests that it is an overall coeffs to the entire force term except for sigmaTr

        # TODO
        # does not work when D.type == CLOVERPC
        
        ck = D.kappa*D.csw/8.0
        k2 = D.kappa*D.kappa
        n = len(vxs)
        
        if not D.full and not D.even:
            # The obstacle is computeCloverSigmaTrace
            # This needs to be able to work on both A_o and A_e
            raise NotImplementedError("QUDA implements only for EVEN case")
        
        # First compute the contribution from Tr ln A
        oprod = force.new(reconstruct="NO", empty=False, is_momentum=False, dofs=(6,18))
        if not D.full and D.even:
            # check!: we need only TrLn A_o for EVEN and TrLn A_e for ODD.
            D.clover.inverse_field
            lib.computeCloverSigmaTrace(oprod.quda_field, D.clover.quda_field, 2.0*ck*mult)
            
        # Now the U dA/dU terms (for the moment, we assume either full or even)
        ferm_epsilon = lib.std.vector([lib.std.vector([2.0*ck*coeffs[i], -k2 * 2.0*ck*coeffs[i]] if not D.full
                                                      else [2.0*ck*coeffs[i], 2.0*ck*coeffs[i]]
                                                      ) for i in range(n)])
        lib.computeCloverSigmaOprod(oprod.quda_field, vxs, vps, ferm_epsilon)
        
        R = [2 if d == 0 else 1 for d in range(4)]
        oprodEx = oprod.extended_field(sites = R)
        u = gauge.extended_field(sites = R)
        if gauge.precision == "double":
            u = gauge.cast(reconstruct="NO").extended_field(sites = R)
        lib.cloverDerivative(force.quda_field, u, oprodEx, 1.0, getattr(lib, "QUDA_ODD_PARITY"))
        lib.cloverDerivative(force.quda_field, u, oprodEx, 1.0, getattr(lib, "QUDA_EVEN_PARITY"))

        return force
