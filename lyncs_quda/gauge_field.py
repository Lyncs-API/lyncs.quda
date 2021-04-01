"""
Interface to gauge_field.h
"""
# pylint: disable=C0303
__all__ = [
    "gauge",
    "GaugeField",
]

from functools import reduce
from time import time
import numpy
import cupy
from lyncs_cppyy import make_shared
from lyncs_cppyy.ll import to_pointer, array_to_pointers
from .lib import lib
from .lattice_field import LatticeField
from .time_profile import default_profiler


def gauge(lattice, dtype=None, device=True, **kwargs):
    "Constructs a new gauge field"
    shape = (4, 18) + tuple(lattice)
    field_kwargs = dict(dtype=dtype)

    if device is False or device is None:
        return GaugeField(numpy.empty(shape, **field_kwargs), **kwargs)

    if device is True:
        device = lib.device_id
    elif isinstance(device, int):
        lib.device_id = device
    else:
        raise TypeError(f"Unsupported type for device: {type(device)}")

    with cupy.cuda.Device(device):
        return GaugeField(cupy.empty(shape, **field_kwargs), **kwargs)


class GaugeField(LatticeField):
    "Mimics the quda::LatticeField object"

    @LatticeField.field.setter
    def field(self, field):
        LatticeField.field.fset(self, field)
        if not str(self.dtype).startswith("float"):
            raise TypeError("GaugeField support only float type")
        if self.reconstruct == "INVALID":
            raise TypeError(f"Unrecognized field dofs {self.dofs}")

    @staticmethod
    def get_reconstruct(dofs):
        "Returns the reconstruct type of dofs"
        dofs = reduce((lambda x, y: x * y), dofs)
        if dofs == 18:
            return "NO"
        if dofs == 12:
            return "12"
        if dofs == 8:
            return "8"
        if dofs == 10:
            return "10"
        return "INVALID"

    @property
    def reconstruct(self):
        "Reconstruct type of the field"
        geo = self.geometry
        if geo == "INVALID":
            return "INVALID"
        if geo == "SCALAR" and self.dofs[0] == 1:
            return self.get_reconstruct(self.dofs[1:])
        if geo != "SCALAR":
            return self.get_reconstruct(self.dofs[1:])
        return self.get_reconstruct(self.dofs)

    @property
    def quda_reconstruct(self):
        "Quda enum for reconstruct type of the field"
        return getattr(lib, f"QUDA_RECONSTRUCT_{self.reconstruct}")

    @property
    def geometry(self):
        """
        Geometry of the field 
            VECTOR = all links
            SCALAR = one link
            TENSOR = Fmunu antisymmetric (upper triangle)
        """
        if self.dofs[0] == self.ndims:
            return "VECTOR"
        if self.dofs[0] == 1:
            return "SCALAR"
        if self.dofs[0] == self.ndims * (self.ndims - 1) / 2:
            return "TENSOR"
        if self.get_reconstruct(self.dofs) != "INVALID":
            return "SCALAR"
        return "INVALID"

    @property
    def quda_geometry(self):
        "Quda enum for geometry of the field"
        return getattr(lib, f"QUDA_{self.geometry}_GEOMETRY")

    @property
    def ghost_exchange(self):
        "Ghost exchange"
        return "NO"

    @property
    def quda_ghost_exchange(self):
        "Quda enum for ghost exchange"
        return getattr(lib, f"QUDA_GHOST_EXCHANGE_{self.ghost_exchange}")

    @property
    def order(self):
        "Data order of the field"
        return "FLOAT2"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_GAUGE_ORDER")

    @property
    def t_boundary(self):
        "Boundary conditions in time"
        return "PERIODIC"

    @property
    def quda_t_boundary(self):
        "Quda enum for boundary conditions in time"
        return getattr(lib, f"QUDA_{self.t_boundary}_T")

    @property
    def quda_params(self):
        "Returns and instance of quda::GaugeFieldParams"
        params = lib.GaugeFieldParam(
            self.quda_dims,
            self.quda_precision,
            self.quda_reconstruct,
            self.pad,
            self.quda_geometry,
            self.quda_ghost_exchange,
        )
        params.link_type = lib.QUDA_SU3_LINKS
        params.gauge = to_pointer(self.ptr)
        params.create = lib.QUDA_REFERENCE_FIELD_CREATE
        params.location = self.quda_location
        params.t_boundary = self.quda_t_boundary
        params.order = self.quda_order
        return params

    @property
    def quda_field(self):
        "Returns and instance of quda::GaugeField"
        self.activate()
        return make_shared(lib.GaugeField.Create(self.quda_params))

    def extended_field(self, sites=1):
        if sites in (None, 0) or self.comm is None:
            return self.quda_field

        if isinstance(sites, int):
            sites = [sites] * self.ndims

        # self.check_shape(sites)

        return make_shared(
            lib.createExtendedGauge(
                self.quda_field,
                numpy.array(sites, dtype="int32"),
                default_profiler().quda,
            )
        )

    def new(self):
        "Returns a new empy field based on the current"
        return gauge(self.lattice, dtype=self.dtype, device=self.device)

    def zero(self):
        "Set all field elements to zero"
        self.quda_field.zero()

    def unity(self):
        "Set all field elements to unity"
        if self.reconstruct != "NO":
            raise NotImplementedError
        tmp = self.field.reshape((2, 4, 9, -1, 2))
        tmp[:] = 0
        tmp[:, :, [0, 4, 8], :, 0] = 1
        self.field = tmp.reshape(self.shape)

    def project(self, tol=None):
        """
        Project the gauge field onto the SU(3) group.  This
        is a destructive operation.  The number of link failures is
        reported so appropriate action can be taken.
        """
        if tol is None:
            tol = numpy.finfo(self.dtype).eps
        fails = cupy.zeros((1,), dtype="int32")
        lib.projectSU3(self.quda_field, tol, to_pointer(fails.data.ptr, "int *"))
        return fails[0]

    def gaussian(self, epsilon=1, seed=None):
        """
        Generates Gaussian distributed su(N) or SU(N) fields.  
        If U is a momentum field, then generates a random Gaussian distributed
        field in the Lie algebra using the anti-Hermitation convention.
        If U is in the group then we create a Gaussian distributed su(n)
        field and exponentiate it, e.g., U = exp(sigma * H), where H is
        the distributed su(n) field and sigma is the width of the
        distribution (sigma = 0 results in a free field, and sigma = 1 has
        maximum disorder).
        """
        seed = seed or int(time() * 1e9)
        lib.gaugeGauss(self.quda_field, seed, epsilon)

    def plaquette(self):
        """
        Computes the plaquette of the gauge field 
        
        Returns
        -------
        tuple(total, spatial, temporal) plaquette site averaged and
            normalized such that each plaquette is in the range [0,1]
        """
        plaq = lib.plaquette(self.extended_field(1))
        return plaq.x, plaq.y, plaq.z

    def topological_charge(self):
        """
        Computes the topological charge

        Returns
        -------
        charge, (total, spatial, temporal): The total topological charge
            and total, spatial, and temporal field energy
        """
        out = numpy.zeros(4, dtype="double")
        lib.computeQCharge(out[:3], out[3:], self.quda_field)
        return out[3], tuple(out[:3])

    def topological_charge_density(self):
        "Computes the topological charge density"
        out1 = numpy.zeros(4, dtype="double")
        if out is None:
            out = numpy.zeros(4, dtype="double")
        return lib.computeQCharge(out[:3], out[3:], self.quda_field)

    def norm1(self, link_dir=-1, local=False):
        "Computes the L1 norm of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.reduce(self.quda_field.norm1(link_dir), local=local)

    def norm2(self, link_dir=-1, local=False):
        "Computes the L2 norm of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.reduce(self.quda_field.norm2(link_dir), local=local)

    def abs_max(self, link_dir=-1, local=False):
        "Computes the absolute maximum of the field (Linfinity norm)"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.reduce(self.quda_field.abs_max(link_dir), local=local, opr="MAX")

    def abs_min(self, link_dir=-1, local=False):
        "Computes the absolute minimum of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.reduce(self.quda_field.abs_min(link_dir), local=local, opr="MIN")

    def compute_paths(self, paths, coeffs=None, add_to=None, add_coeff=1):
        """
        Computes the gauge paths on the lattice.
        
        The same paths are computed for every direction.

        - The paths are given with respect to direction "1" and 
          this must be the first number of every path list.
        - Directions go from 1 to self.ndims
        - Negative value (-1,...) means backward movement in the direction
        - Paths are then rotated for every direction.
        """

        if coeffs is None:
            coeffs = [1] * len(paths)
        elif isinstance(coeffs, (int, float)):
            coeffs = [coeffs] * len(paths)
        if not len(paths) == len(coeffs):
            raise ValueError("Paths and coeffs must have the same length")

        num_paths = len(paths)
        coeffs = numpy.array(coeffs, dtype="float64")
        lengths = (
            numpy.array(list(map(len, paths)), dtype="int32") - 1
        )  # -1 because the first step is always 1 (the direction itself)
        max_length = int(lengths.max())
        paths_array = numpy.zeros((self.ndims, num_paths, max_length), dtype="int32")

        for i, path in enumerate(paths):
            if min(path) < -self.ndims:
                raise ValueError(
                    f"Path {i} = {path} has direction smaller than {-self.ndims}"
                )
            if max(path) > self.ndims:
                raise ValueError(
                    f"Path {i} = {path} has direction larger than {self.ndims}"
                )
            if path[0] != 1:
                raise ValueError(f"Path {i} = {path} does not start with 1")
            if 0 in path:
                raise ValueError(f"Path {i} = {path} has zeros")

            for dim in range(self.ndims):
                for j, step in enumerate(path[1:]):
                    if step > 0:
                        paths_array[dim, i, j] = (step - 1 + dim) % self.ndims
                    else:
                        paths_array[dim, i, j] = 7 - (-step - 1 + dim) % self.ndims

        if add_to is None:
            add_to = self.new()
            add_to.zero()

        quda_paths_array = array_to_pointers(paths_array)
        in_quda_field = lib.createExtendedGauge(
            self.quda_field, numpy.ones(4, dtype="int32"), default_profiler().quda
        )
        lib.gaugePath(
            add_to.quda_field,
            self.quda_field,
            add_coeff,
            quda_paths_array.view,
            lengths,
            coeffs,
            num_paths,
            max_length,
        )
        return add_to

    def plaquette_field(self):
        "Computes the plaquette field"
        return self.compute_paths(
            [[1, 2, -1, -2], [1, 3, -1, -3], [1, 4, -1, -4]], coeffs=1 / 3
        )

    def rectangle_field(self):
        "Computes the rectangle field"
        return self.compute_paths(
            [
                [1, 2, 2, -1, -2, -2],
                [1, 3, 3, -1, -3, -3],
                [1, 4, 4, -1, -4, -4],
                [1, 1, 2, -1, -1, -2],
                [1, 1, 3, -1, -1, -3],
                [1, 1, 4, -1, -1, -4],
            ],
            coeffs=1 / 6,
        )

    def exponentiate(self, coeff=1, mul_to=None, out=None, conj=False, exact=False):
        """
        Exponentiates a momentum field
        """
        if out is None:
            out = self.new()
        if mul_to is None:
            mul_to = self.new()
            mul_to.unity()

        lib.updateGaugeField(
            out.quda_field, coeff, mul_to.quda_field, self.quda_field, conj, exact
        )
        return out
