"""
Interface to gauge_field.h
"""

__all__ = [
    "gauge",
    "gauge_field",
    "gauge_scalar",
    "gauge_links",
    "gauge_tensor",
    "gauge_coarse",
    "momentum",
    "GaugeField",
]

from time import time
from math import sqrt
from collections import defaultdict
import numpy
from lyncs_cppyy import make_shared, lib as tmp, to_pointer, array_to_pointers
from lyncs_utils import prod, isiterable
from .lib import lib, cupy
from .lattice_field import LatticeField, backend
from .time_profile import default_profiler

# TODO: Make array dims consistent with gauge order


def gauge_field(lattice, dofs=(4, 18), **kwargs):
    "Constructs a new gauge field"
    # TODO add option to select field type -> dofs
    # TODO reshape/shuffle to native order
    return GaugeField.create(lattice, dofs=dofs, **kwargs)


def gauge_scalar(lattice, dofs=18, **kwargs):
    "Constructs a new scalar gauge field"
    return gauge_field(lattice, dofs=(1, dofs), **kwargs)


def gauge_links(lattice, dofs=18, **kwargs):
    "Constructs a new gauge field of links"
    return gauge_field(lattice, dofs=(4, dofs), **kwargs)


gauge = gauge_links


def gauge_tensor(lattice, dofs=18, **kwargs):
    "Constructs a new gauge field with tensor structure"
    return gauge_field(lattice, dofs=(6, dofs), **kwargs)


def gauge_coarse(lattice, dofs=2 * 48**2, **kwargs):
    "Constructs a new coarse gauge field"
    return gauge_field(lattice, dofs=(8, dofs), **kwargs)


def momentum(lattice, **kwargs):
    return gauge_field(lattice, dofs=(4, 10), **kwargs)


class GaugeField(LatticeField):
    "Mimics the quda::GaugeField object"

    @LatticeField.field.setter
    def field(self, field):
        LatticeField.field.fset(self, field)
        if self.reconstruct == "INVALID":
            raise TypeError(f"Unrecognized field dofs {self.dofs}")

    def new(self, reconstruct=None, **kwargs):
        "Returns a new empty field based on the current"
        if reconstruct is None:
            pass
        elif reconstruct == self.reconstruct:
            pass
        elif reconstruct == "NO":  # ? what if geometry == COARSE?
            size = self.ncol**2
            kwargs["dofs"] = (self.geometry_size, size if self.iscomplex else size * 2)
        else:
            try:
                val = int(reconstruct)
                kwargs["dofs"] = (
                    self.geometry_size,
                    val // 2 if self.iscomplex else val,
                )
            except ValueError:
                raise ValueError(f"Invalid reconstruct {reconstruct}")
        out = super().new(**kwargs)
        is_momentum = kwargs.get("is_momentum", self.is_momentum)
        out.is_momentum = is_momentum
        return out

    def equivalent(self, other, **kwargs):
        "Whether a field is equivalent to the current"
        if not super().equivalent(other, **kwargs):
            return False
        reconstruct = kwargs.get("reconstruct", self.reconstruct)
        if other.reconstruct != str(reconstruct):
            return False
        return True

    def copy(self, other=None, out=None, **kwargs):
        "Returns out, a copy+=kwargs, of other if given, else of self"
        #Note for Future Developers:
        # This will turn every term in the expression into the one
        #  with link_type = MOM if there is one such term in the expression
        # As for other link types except SU3 and MOM, such mixture simply
        #  results in QUDA errors

        src = self if other is None else other
        dst = self if out   is None else out
        if src.is_momentum != dst.is_momentum:
            kwargs.update({"is_momentum": True})
        
        return super().copy(other, out, **kwargs) 
        
    def __array_finalize__(self, obj):
        "Support for __array_finalize__ standard"
        # need to reset QUDA object when meta data of its Python wrapper is changed
        self._quda = None

    def _prepare(self, *fields, **kwargs):
        "Prepares the fields by creating new one if None given else casting them to type(self) then checking them if compatible with self and/or copying them"
        
        fields = super()._prepare(*fields, **kwargs)
        #? we don't need to do the following except for the base case?
        for field in fields if isinstance(fields, tuple) else (fields,):
            is_momentum = kwargs.get("is_momentum", self.is_momentum)
            field.is_momentum = is_momentum
            
        return fields

    @property
    def dofs_per_link(self):
        if self.geometry == "SCALAR":
            dofs = prod(self.dofs)
        else:
            dofs = prod(self.dofs[1:])
        if self.iscomplex:
            return dofs * 2
        return dofs

    @property
    def reconstruct(self):
        "Reconstruct type of the field"
        dofs = self.dofs_per_link
        if dofs == 12:
            return "12"
        if dofs == 8:
            return "8"
        if dofs == 10:
            return "10"
        if sqrt(dofs / 2).is_integer():
            return "NO"
        return "INVALID"

    @property
    def quda_reconstruct(self):
        "Quda enum for reconstruct type of the field"
        return getattr(lib, f"QUDA_RECONSTRUCT_{self.reconstruct}")

    @property
    def ncol(self):
        "Number of colors"
        if self.reconstruct == "NO":
            dofs = self.dofs_per_link
            ncol = sqrt(dofs / 2)
            assert ncol.is_integer()
            return int(ncol)
        return 3

    @property
    def order(self):
        "Data order of the field"
        dofs = self.dofs_per_link
        if self.precision != "double" and (
            dofs == 8 or dofs == 12
        ):  # if FLOAT8 defined, if prec=half/quarter and recon=8, FLOAT8
            return "FLOAT4"
        return "FLOAT2"

    @property
    def quda_order(self):
        "Quda enum for data order of the field"
        return getattr(lib, f"QUDA_{self.order}_GAUGE_ORDER")

    @property
    def _geometry_values(self):
        return (
            ("SCALAR", "VECTOR", "TENSOR", "COARSE"),
            (1, self.ndims, self.ndims * (self.ndims - 1) / 2, self.ndims * 2),
        )

    @property
    def geometry(self):
        """
        Geometry of the field
            VECTOR = all links
            SCALAR = one link
            TENSOR = Fmunu antisymmetric (upper triangle)
            COARSE = all links, both directions
        """
        keys, vals = self._geometry_values
        if self.dofs[0] in vals:
            return keys[vals.index(self.dofs[0])]
        return "SCALAR"

    @property
    def geometry_size(self):
        "Size of the geometry index"
        keys, vals = self._geometry_values
        if self.dofs[0] in vals:
            return self.dofs[0]
        return 1

    @property
    def quda_geometry(self):
        "Quda enum for geometry of the field"
        return getattr(lib, f"QUDA_{self.geometry}_GEOMETRY")

    @property
    def is_coarse(self):
        "Whether is a coarse gauge field"
        return self.geometry == "COARSE" or (
            self.geometry == "SCALAR" and self.ncol != 3
        )

    @property
    def is_momentum(self):
        "Whether is a momentum field"
        return self.reconstruct == "10" or getattr(self, "_is_momentum", False)

    @is_momentum.setter
    def is_momentum(self, value):
        self._is_momentum = value
        if self._quda is not None:
            self._quda.link_type = self.quda_link_type

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
        if self.is_coarse:
            return "COARSE"
        if self.is_momentum:
            return "MOMENTUM"
        return "SU3"

    @property
    def quda_link_type(self):
        "Quda enum for link type"
        return getattr(lib, f"QUDA_{self.link_type}_LINKS")

    @property
    def quda_params(self):
        "Returns an instance of quda::GaugeFieldParams"
        # TODO: Support MILC gauge order (site_offset, site_size)
        # TODO: Support Staggered phase (staggeredPhaseType, staggeredPhaseApplied)
        # TODO: Allow control on QudaGaugeFixed, i_mu, nFace, anisotropy, tadpole, compute_fat_link_max,
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
        params.nColor = self.ncol
        return params

    @property
    def quda_field(self):
        "Returns an instance of quda::(cpu/cuda)GaugeField for QUDA_(CPU/CUDA)_FIELD_LOCATION"
        self.activate()
        if self._quda is None:
            self._quda = make_shared(lib.GaugeField.Create(self.quda_params))
        return self._quda

    def is_native(self):
        "Whether the field is native for Quda"
        return lib.gauge.isNative(
            self.quda_order, self.quda_precision, self.quda_reconstruct
        )

    def extended_field(self, sites=1):
        "Extends the gauge field in each direction by sites (i.e., width of the halo shell) on each MPI rank"
        # TODO: Check the acceptable geometries of the gauge field
        if sites in (None, 0) or self.comm is None:
            return self.quda_field

        if isinstance(sites, int):
            sites = [sites] * self.ndims

        sites = [site if dim > 1 else 0 for site, dim in zip(sites, self.comm.dims)]
        if sites == [0, 0, 0, 0]:
            return self.quda_field

        if self.location == "CPU":
            "Returns cpuGaugeField"
            """
              Remark:
                * createExtendedGauge takes (void **) as its first argument
                * However, it is used to set (void *gauge) in GaugeFieldParam
                * (void **gauge) is defined as a private member in cpuGaugeField
                * This private member does not seem relevant in createExtendedGauge as it sets
                   'create' to QUDA_ZERO_FIELD_CREATE for the returned cpuGaugeField,
                   which is copied from the input gauge with halo via  copyExtendedGauge,
                   which in turn takes GaugeField
                * So in copyExtendedGauge for cpuGaugeField, (void *gauge) is copied into
                   (void * gauge) in the returned cpuGaugeField with halo, and this (void *gauge)
                   is what we want in the end so that upcasting is justified
            """
            return make_shared(
                lib.createExtendedGauge(
                    self.ptr, self.quda_params, numpy.array(sites, dtype="int32")
                )
            )
        elif self.location == "CUDA":
            "Returns cudaGaugeField"
            return make_shared(
                lib.createExtendedGauge(
                    self.quda_field,  # quda_field returns an instance of cudaGaugeField in this case
                    numpy.array(sites, dtype="int32"),
                    default_profiler().quda,
                )
            )
        else:
            raise ValueError("Something is wrong!")  # just for debugging

    def zero(self):
        "Sets all field elements to zero"
        self.quda_field.zero()

    def full(self):
        "Returns a full matrix version of the field (with reconstruct=NO)"
        out = self if self.reconstruct == "NO" else self.copy(reconstruct="NO")
        out.is_momentum = False
        return out

    def to_momentum(self):
        "Returns a momentum version of the field (with reconstruct=10)"
        return self if self.reconstruct == "10" else self.copy(reconstruct="10")

    def default_view(self, split_col=True):
        "Returns the default view of the field including reshaping"
        # ? if we take into account FLAOT4 order, unity, etc shoud not depend on this; tr,dag might need reshuffle
        shape = (2,)  # even-odd
        # geometry
        if len(self.dofs) == 1:
            shape += (1,)
        else:
            shape += (self.dofs[0],)
        # matrix
        if self.reconstruct == "NO" and split_col:
            shape += (self.ncol, self.ncol)
        else:
            shape += (self.dofs_per_link // 2,)
        # lattice
        shape += (-1,)
        if self.reconstruct == "10":
            shape += (2,)
            return super().float_view().reshape(shape)
        return super().complex_view().reshape(shape)

    def unity(self):
        "Set all field elements to unity"
        if self.reconstruct != "NO":
            raise NotImplementedError

        field = self.default_view(split_col=False)

        field[:] = 0
        diag = [i * self.ncol + i for i in range(self.ncol)]
        field[:, :, diag, ...] = 1

    def trace(self, only_real=False):
        "Returns the trace in color of the field"
        return (
            self.full()
            .default_view()
            .trace(axis1=2, axis2=3, dtype="float64" if only_real else "complex128")
        )

    def dagger(self, out=None):
        "Returns the complex conjugate transpose of the field"
        out = self.prepare_out(out)
        self.backend.conj(
            self.default_view().transpose((0, 1, 3, 2, 4)), out=out.default_view()
        )
        return out

    def reduce(self, local=False, only_real=True, mean=True):
        "Reduction of a gauge field (real of mean of trace)"
        # ASSUME: cupy array
        out = self.trace(only_real=only_real)
        if mean:
            out = out.mean() / self.ncol
        else:
            out = out.sum()
        if not local:
            return super().reduce(out.get())
        return out

    def dot(self, other, out=None):
        "Matrix product between two gauge fields"
        if not isinstance(other, GaugeField):
            raise ValueError
        self = self.full()
        other = other.full()
        out = self.prepare_out(out)
        self.backend.matmul(
            self.default_view(),
            other.default_view(),
            out=out.default_view(),
            axes=[(2, 3)] * 3,
        )
        return out

    def project(self, tol=None):
        """
        Project the gauge field onto the SU(3) group.  This
        is a destructive operation.  The number of link failures is
        reported so appropriate action can be taken.
        """
        if self.location == "CPU":
            raise NotImplementedError(
                "This method currently works only when running on GPUs"
            )

        if tol is None:
            tol = numpy.finfo(self.dtype).eps

        assert self.device == cupy.cuda.runtime.getDevice()
        fails = cupy.zeros((1,), dtype="int32")
        lib.projectSU3(self.quda_field, tol, to_pointer(fails.data.ptr, "int *"))
        # return fails.get()[0]  # shouldn't we reduce?
        return super().reduce(fails.get()[0])

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
        # TODO: Check the acceptable geometries of the gauge field
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
        if self.location == "CPU":  # I don't think double3 is defined without CUDA
            raise NotImplementedError(
                "The underlying QUDA function will not work without GPU"
            )

        if self.geometry != "VECTOR":
            raise TypeError("This gauge object needs to have VECTOR geometry")
        plaq = lib.plaquette(self.extended_field(1))
        return plaq.x, plaq.y, plaq.z

    def compute_fmunu(self, out=None):
        if self.geometry == "TENSOR":
            return self
        if self.geometry != "VECTOR":
            raise TypeError("This gauge object needs to have VECTOR geometry")

        out = self.prepare_out(out, dofs=(6, 9 if self.iscomplex else 18))
        lib.computeFmunu(out.quda_field, self.extended_field(1))
        return out

    def topological_charge(self):
        """
        Computes the topological charge

        Returns
        -------
        charge, (total, spatial, temporal): The total topological charge
            and total, spatial, and temporal field energy
        """
        if self.geometry != "TENSOR":
            self = self.compute_fmunu()

        out = numpy.zeros(4, dtype="double")
        lib.computeQCharge(out[:3], out[3:], self.quda_field)
        return out[3], tuple(out[:3])

    def topological_charge_density(self, density=None):
        """
        Computes the topological charge and density

        Returns
        -------
        charge, (total, spatial, temporal), density:
          The total topological charge, (total, spatial, temporal) field energy, and
          topological charge density
        """
        if self.geometry != "TENSOR":
            self = self.compute_fmunu()
        charge = numpy.zeros(4, dtype="double")
        if density is None:
            density = self.new(dofs=(1,), dtype=self.precision)
        lib.computeQChargeDensity(charge[:3], charge[3:], density.ptr, self.quda_field)
        return charge[3], tuple(charge[:3]), density

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
        if self.reconstruct == "10":
            # TODO: patch quda, reconstruct 10 not supported
            # ? assume cupy?
            norm2 = (self.default_view() ** 2).sum(axis=(0, 1, 3, 4))
            norm2 = norm2.sum() - norm2[-1] / 2 - norm2[-2] / 2
            return 4 * super().reduce(norm2.get())
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

    def _check_paths(self, paths):
        "Check if all paths are valid"
        if not isiterable(paths):
            raise TypeError(f"Paths ({paths}) are not iterable")
        for i, path in enumerate(paths):
            if not isiterable(path):
                raise TypeError(f"Path {i} = {path} is not iterable")
            if path[0] < 0:
                raise ValueError(f"Path {i} = {path} nevative first movement")
            if min(path) < -self.ndims:
                raise ValueError(
                    f"Path {i} = {path} has direction smaller than {-self.ndims}"
                )
            if max(path) > self.ndims:
                raise ValueError(
                    f"Path {i} = {path} has direction larger than {self.ndims}"
                )
            if 0 in path:
                raise ValueError(f"Path {i} = {path} has zeros")

    def _paths_to_array(self, paths):
        "Returns array of paths and their length"

        lengths = numpy.array(list(map(len, paths)), dtype="int32")
        max_length = lengths.max()
        paths_array = numpy.empty((1, len(paths), max_length), dtype="int32")

        convert = lambda step: (step - 1) if step > 0 else (8 + step)

        for i, path in enumerate(paths):
            for j, step in enumerate(path):
                paths_array[0, i, j] = convert(step)

        return paths_array, lengths

    def _paths_for_force(self, paths, coeffs):
        "Create all paths needed for force"
        out = defaultdict(int)
        shift = lambda path, i: path if i in (0, len(path)) else (path[i:] + path[:i])
        for path, coeff in zip(paths, coeffs):
            for i in range(len(path)):
                if path[i] > 0:
                    tmp = shift(path, i)
                else:
                    tmp = tuple(-_ for _ in reversed(shift(path, i + 1)))
                out[tmp] -= coeff
        return tuple(out.keys()), tuple(out.values())

    def compute_paths(
        self,
        paths,
        coeffs=None,
        out=None,
        add_coeff=1,
        force=False,
        grad=None,
        left_grad=False,
        keep_paths=False,
    ):
        """
        Computes the gauge paths on the lattice.

        The same paths are computed for every direction.

        - The paths are given with respect to direction "1" and
          this must be the first number of every path list.
        - Directions go from 1 to self.ndims
        - Negative value (-1,...) means backward movement in the direction
        - Paths are then rotated for every direction.
        """
        if self.geometry != "VECTOR":
            raise TypeError("This gauge object needs to have VECTOR geometry")

        # Checking paths for error
        self._check_paths(paths)
        paths = tuple(tuple(path) for path in paths)

        # Preparing coeffs
        if coeffs is None:
            coeffs = self.ndims / len(paths)
        if isinstance(coeffs, (int, float)):
            coeffs = [coeffs] * len(paths)
        if not len(paths) == len(coeffs):
            raise ValueError("Paths and coeffs must have the same length")

        # Preparing grad and fnc
        if grad is not None:
            grad = self.prepare_in(grad, reconstruct=10)
            fnc = lambda out, u, *args: self._gaugeForceGradient(
                out,
                u,
                grad.quda_field,
                *args,
                left=left_grad,
            )
        elif force:
            fnc = self._gaugeForce
        else:
            fnc = self._gaugePath

        # Preparing paths
        if force and not keep_paths:
            paths, coeffs = self._paths_for_force(paths, coeffs)
            self._check_paths(paths)
        paths, lengths = self._paths_to_array(paths)

        # Calling Quda function
        num_paths = paths.shape[1]
        max_length = paths.shape[2]
        quda_paths_array = array_to_pointers(paths)
        coeffs = numpy.array(coeffs, dtype="float64")
        out = self.prepare_out(out, empty=False, reconstruct=10 if force else None)

        fnc(
            out.quda_field,
            self.extended_field(1),  # TODO: compute correct extension (max distance)
            add_coeff,
            quda_paths_array.get(),
            lengths,
            coeffs,
            num_paths,
            max_length,
            False,
        )
        return out

    # for profiling
    def _gaugeForceGradient(self, *args, **kwargs):
        return lib.gaugeForceGradient(*args, **kwargs)

    def _gaugeForce(self, *args, **kwargs):
        return lib.gaugeForce(*args, **kwargs)

    def _gaugePath(self, *args, **kwargs):
        return lib.gaugePath(*args, **kwargs)

    @property
    def plaquette_paths(self):
        "List of plaquette paths"
        return tuple(
            (mu, nu, -mu, -nu)
            for mu in range(1, self.ndims + 1)
            for nu in range(mu + 1, self.ndims + 1)
        )

    def plaquette_field(self, **kwargs):
        "Computes the plaquette field"
        return self.compute_paths(self.plaquette_paths, **kwargs)

    def plaquettes(self, **kwargs):
        "Returns the average over plaquettes (Note: plaquette should performs better)"
        return self.plaquette_field().reduce(**kwargs)

    @property
    def rectangle_paths(self):
        "List of rectangle paths"
        return tuple(
            (mu, nu, nu, -mu, -nu, -nu)
            for mu in range(1, self.ndims + 1)
            for nu in range(mu + 1, self.ndims + 1)
        )

    def rectangle_field(self, **kwargs):
        "Computes the rectangle field"
        return self.compute_paths(self.rectangle_paths, **kwargs)

    def rectangles(self, **kwargs):
        "Returns the average over rectangles"
        return self.rectangle_field().reduce(**kwargs)

    def exponentiate(self, coeff=1.0, mul_to=None, out=None, conj=False, exact=False):
        """
        Exponentiates a momentum field
        """
        # TODO: Check the acceptable geometries of the gauge field
        if out is None:
            out = (
                mul_to.new()
                if mul_to is not None
                else self.new(reconstruct="NO", is_momentum=False)
            )  # the result of exponentiation should be SU3?
        if mul_to is None:
            mul_to = out.new()
            mul_to.unity()

        lib.updateGaugeField(
            out.quda_field, coeff, mul_to.quda_field, self.quda_field, conj, exact
        )
        return out

    def update_gauge(self, mom, coeff=1, out=None, conj=False, exact=False):
        """
        Updates a gauge field with momentum field
        """
        mom = self.prepare_in(mom, reconstruct="10")
        return mom.exponentiate(
            coeff=coeff, mul_to=self, out=out, conj=conj, exact=exact
        )

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

    def S_F(self, phi, **params):
        "Retruns pseudo-fermionic action given the pseudo-fermion field"
        solver = self.Dirac(**params).Solver()
        s_params = {k: v for k, v in params.items() if k in solver.default_params}
        out = solver(phi, **s_params)
        return out.norm2()

    def fermionic_force(self):
        pass
