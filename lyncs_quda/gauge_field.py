"""
Interface to gauge_field.h
"""

__all__ = [
    "gauge",
    "GaugeField",
]

from functools import reduce
from time import time
import numpy
from lyncs_cppyy import make_shared, lib as tmp
from lyncs_cppyy.ll import to_pointer
from lyncs_cppyy.numpy import array_to_pointers
from .lib import lib, cupy
from .lattice_field import LatticeField
from .time_profile import default_profiler


def gauge(lattice, dofs=(4, 18), **kwargs):
    "Constructs a new gauge field"
    # TODO add option to select field type -> dofs
    # TODO reshape/shuffle to native order
    return GaugeField.create(lattice, dofs, **kwargs)


class GaugeField(LatticeField):
    "Mimics the quda::LatticeField object"

    @LatticeField.field.setter
    def field(self, field):
        LatticeField.field.fset(self, field)
        if self.reconstruct == "INVALID":
            raise TypeError(f"Unrecognized field dofs {self.dofs}")

    def get_reconstruct(self, dofs):
        "Returns the reconstruct type of dofs"
        dofs = reduce((lambda x, y: x * y), dofs)
        if self.iscomplex:
            dofs *= 2
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
    def order(self):
        "Data order of the field"
        return "FLOAT2"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_GAUGE_ORDER")

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
    def t_boundary(self):
        "Boundary conditions in time"
        return "PERIODIC"

    @property
    def quda_t_boundary(self):
        "Quda enum for boundary conditions in time"
        return getattr(lib, f"QUDA_{self.t_boundary}_T")

    @property
    def link_type(self):
        "Type of the links"
        return "SU3"

    @property
    def quda_link_type(self):
        "Quda enum for link type"
        return getattr(lib, f"QUDA_{self.link_type}_LINKS")

    @property
    def quda_params(self):
        "Returns an instance of quda::GaugeFieldParams"
        params = lib.GaugeFieldParam()
        lib.copy_struct(params, super().quda_params)
        params.reconstruct = self.quda_reconstruct
        params.geometry = self.quda_geometry
        params.link_type = self.quda_link_type
        params.gauge = to_pointer(self.ptr)
        params.create = lib.QUDA_REFERENCE_FIELD_CREATE
        params.location = self.quda_location
        params.t_boundary = self.quda_t_boundary
        params.order = self.quda_order
        return params

    @property
    def quda_field(self):
        "Returns an instance of quda::GaugeField"
        self.activate()
        return make_shared(lib.GaugeField.Create(self.quda_params))

    def is_native(self):
        "Whether the field is native for Quda"
        return lib.gauge.isNative(
            self.quda_order, self.quda_precision, self.quda_reconstruct
        )

    def extended_field(self, sites=1):
        if sites in (None, 0) or self.comm is None:
            return self.quda_field

        if isinstance(sites, int):
            sites = [sites] * self.ndims

        # self.check_shape(sites)
        sites = [site if dim > 1 else 0 for site, dim in zip(sites, self.comm.dims)]
        if sites == [0, 0, 0, 0]:
            return self.quda_field

        return make_shared(
            lib.createExtendedGauge(
                self.quda_field,
                numpy.array(sites, dtype="int32"),
                default_profiler().quda,
            )
        )

    def zero(self):
        "Sets all field elements to zero"
        self.quda_field.zero()

    def unity(self):
        "Set all field elements to unity"
        if self.reconstruct != "NO":
            raise NotImplementedError
        tmp = self.field.reshape((2, 4, 9, -1, 2))
        tmp[:] = 0
        tmp[:, :, [0, 4, 8], :, 0] = 1
        self.field = tmp.reshape(self.shape)

    def trace(self):
        "Returns the trace in color of the field"
        if self.reconstruct != "NO":
            raise NotImplementedError
        field = self.field
        if self.dtype == "float64":
            field = field.view("complex128")
        elif self.dtype == "float32":
            field = field.view("complex64")
        else:
            assert self.iscomplex
        return field.reshape((2, 4, 3, 3, -1)).trace(axis1=2, axis2=3)

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
        lib.computeQCharge(out[:3], out[3:], self.extended_field(0))  # should be 1
        return out[3], tuple(out[:3])

    def topological_charge_density(self):
        "Computes the topological charge density"
        out1 = numpy.zeros(4, dtype="double")
        if out is None:
            out = numpy.zeros(4, dtype="double")
        return lib.computeQCharge(out[:3], out[3:], self.extended_field(1))

    def norm1(self, link_dir=-1):
        "Computes the L1 norm of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.norm1(link_dir)

    def norm2(self, link_dir=-1):
        "Computes the L2 norm of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.norm2(link_dir)

    def abs_max(self, link_dir=-1):
        "Computes the absolute maximum of the field (Linfinity norm)"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.abs_max(link_dir)

    def abs_min(self, link_dir=-1):
        "Computes the absolute minimum of the field"
        if not -1 <= link_dir < self.ndims:
            raise ValueError(
                f"link_dir can be either -1 (all) or must be between 0 and {self.ndims}"
            )
        return self.quda_field.abs_min(link_dir)

    def compute_paths(self, paths, coeffs=None, out=None, add_coeff=1, force=False):
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

        quda_paths_array = array_to_pointers(paths_array)

        kwargs = dict(empty=False)
        fnc = lib.gaugePath
        if force:
            fnc = lib.gaugeForce
            if self.iscomplex:
                kwargs["dofs"] = (4, 5)
            else:
                kwargs["dofs"] = (4, 10)

        out = self.prepare(out, **kwargs)

        fnc(
            out.quda_field,
            self.extended_field(1),
            add_coeff,
            quda_paths_array,
            lengths,
            coeffs,
            num_paths,
            max_length,
        )
        return out

    def plaquette_field(self, force=False):
        "Computes the plaquette field"
        return self.compute_paths(
            [[1, 2, -1, -2], [1, 3, -1, -3], [1, 4, -1, -4]], coeffs=1 / 3, force=force
        )

    def rectangle_field(self, force=False):
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
            force=force,
        )

    def rectangles(self):
        "Returns the average over rectangles"
        # Suboptimal implementation based on rectangle_field
        local = self.rectangle_field().trace().mean().real / 3
        # TODO: global reduction
        return float(local)

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

    def gauge_action(self, plaq_coeff=0, rect_coeff=0):
        """
        Returns the gauge action.

        The coefficients are use as follows

            (1-8*c1) ((1+c0) P_time + (1-c0) P_space) + c1*R

        where P is the sum over plaquette, R over the rectangles,
        c0 is the plaq_coeff (see volume_plaquette) and c1 the rect_coeff
        """
        plaq, time, space = self.plaquette()
        if plaq_coeff != 0:
            plaq = (1 + plaq_coeff) * time + (1 - plaq_coeff) * space

        if rect_coeff == 0:
            return plaq

        rect = self.rectangles()
        return (1 - 8 * rect_coeff) * plaq + rect_coeff * rect

    def symanzik_gauge_action(self, plaq_coeff=0):
        "Returns the tree-level Symanzik improved gauge action"
        return self.gauge_action(plaq_coeff, -1 / 12)

    def iwasaki_gauge_action(self, plaq_coeff=0):
        "Returns the Iwasaki gauge action"
        return self.gauge_action(plaq_coeff, -0.331)
