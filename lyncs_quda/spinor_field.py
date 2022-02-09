"""
Interface to gauge_field.h
"""

__all__ = [
    "spinor",
    "spinor_coarse",
    "SpinorField",
]

from functools import reduce
from time import time
from lyncs_cppyy import make_shared
from lyncs_cppyy.ll import to_pointer
from .lib import lib
from .lattice_field import LatticeField


def spinor(lattice, **kwargs):
    "Constructs a new spinor field"
    return SpinorField.create(lattice, dofs=(4, 3), **kwargs)


def spinor_coarse(lattice, dofs=24, **kwargs):
    "Constructs a new coarse spinor field (dofs=nColor)"
    kwargs.setdefault("gamma_basis", "DEGRAND_ROSSI")
    return SpinorField.create(lattice, dofs=(2, dofs), **kwargs)


class SpinorField(LatticeField):
    "Mimics the quda::ColorSpinorField object"

    gammas = ["DEGRAND_ROSSI", "UKQCD", "CHIRAL"]

    @classmethod
    def get_dtype(cls, dtype):
        dtype = super().get_dtype(dtype)
        if dtype in ["float64", "complex128"]:
            return "complex128"
        if dtype in ["float32", "complex64"]:
            return "complex64"
        if dtype in ["float16", "complex32"]:
            return "complex32"
        raise TypeError("Unsupported dtype for spinor")

    def __init__(self, *args, gamma_basis=None, site_order="EO", **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_basis = gamma_basis
        self.site_order = site_order

    @property
    def ncolor(self):
        "Number of colors of the field"
        return self.dofs[-1]

    @property
    def nspin(self):
        "Number of spin component. 1 for staggered, 2 for coarse Dslash, 4 for 4d spinor"
        return self.dofs[-2]

    @property
    def nvec(self):
        "Number of packed vectors"
        if len(self.dofs) == 3:
            return self.dofs[0]
        return 1

    @property
    def gamma_basis(self):
        "Gamma basis in use"
        return self._gamma_basis

    @gamma_basis.setter
    def gamma_basis(self, value):
        if value is None:
            value = "UKQCD"
        values = f"Possible values are {SpinorField.gammas}"
        if not isinstance(value, str):
            raise TypeError("Expected a string. " + values)
        if not value.upper() in values:
            raise ValueError("Invalid gamma. " + values)
        self._gamma_basis = value.upper()

    @property
    def quda_gamma_basis(self):
        "Quda enum for gamma basis in use"
        return getattr(lib, f"QUDA_{self.gamma_basis}_GAMMA_BASIS")

    @property
    def order(self):
        "Data order of the field"
        return "FLOAT2"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_FIELD_ORDER")

    @property
    def twist_flavor(self):
        "Twist flavor of the field"
        return "SINGLET"

    @property
    def quda_twist_flavor(self):
        "Quda enum for twist flavor of the field"
        return getattr(lib, f"QUDA_TWIST_{self.twist_flavor}")

    @property
    def site_order(self):
        "Site order in use"
        return self._site_order

    @site_order.setter
    def site_order(self, value):
        if value is None:
            value = "NONE"
        values = f"Possible values are NONE, EVEN_ODD, ODD_EVEN"
        if not isinstance(value, str):
            raise TypeError("Expected a string. " + values)
        value = value.upper()
        if value in ["NONE", "LEX", "LEXICOGRAPHIC"]:
            value = "LEXICOGRAPHIC"
        elif value in ["EO", "EVEN_ODD"]:
            value = "EVEN_ODD"
        elif value in ["OE", "ODD_EVEN"]:
            value = "ODD_EVEN"
        else:
            raise ValueError("Invalid site_order. " + values)
        self._site_order = value

    @property
    def quda_site_order(self):
        "Quda enum for site order in use"
        return getattr(lib, f"QUDA_{self.site_order}_SITE_ORDER")

    @property
    def quda_pc_type(self):
        "Select checkerboard preconditioning method"
        return getattr(lib, f"QUDA_{self.ndims}D_PC")

    @property
    def quda_params(self):
        "Returns and instance of quda::ColorSpinorParams"
        params = lib.ColorSpinorParam()
        lib.copy_struct(params, super().quda_params)
        params.nColor = self.ncolor
        params.nSpin = self.nspin
        params.nVec = self.nvec
        params.gammaBasis = self.quda_gamma_basis
        params.pc_type = self.quda_pc_type
        params.twistFlavor = self.quda_twist_flavor

        params.v = to_pointer(self.ptr)
        params.create = lib.QUDA_REFERENCE_FIELD_CREATE
        params.location = self.quda_location
        params.fieldOrder = self.quda_order
        params.siteOrder = self.quda_site_order
        return params

    @property
    def quda_field(self):
        "Returns and instance of quda::ColorSpinorField"
        self.activate()
        return make_shared(lib.ColorSpinorField.Create(self.quda_params))

    def is_native(self):
        "Whether the field is native for Quda"
        return lib.colorspinor.isNative(
            self.quda_order, self.quda_precision, self.nspin, self.ncolor
        )

    def zero(self):
        "Set all field elements to zero"
        self.quda_field.zero()

    def gaussian(self, seed=None):
        "Generates a random gaussian noise spinor"
        seed = seed or int(time() * 1e9)
        lib.spinorNoise(self.quda_field, seed, lib.QUDA_NOISE_GAUSS)

    def uniform(self, seed=None):
        "Generates a random uniform noise spinor"
        seed = seed or int(time() * 1e9)
        lib.spinorNoise(self.quda_field, seed, lib.QUDA_NOISE_UNIFORM)

    def gamma5(self, out=None):
        "Returns the vector transformed by gamma5"
        out = self.prepare(out)
        lib.gamma5(out.quda_field, self.quda_field)
        return out

    def apply_gamma5(self):
        "Applies gamma5 to the field itself"
        return self.gamma5(self)

    def norm1(self):
        "L1 norm of the field"
        return lib.blas.norm1(self.quda_field)

    def norm2(self):
        "L2 norm of the field"
        return lib.blas.norm2(self.quda_field)

    norm = norm2
