from lyncs_quda import gauge, momentum, spinor
import numpy as np
import cupy as cp
from pytest import skip
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    device_loop,
    dtype_loop,
    epsilon_loop,
)
from lyncs_cppyy.ll import addressof
from math import isclose


@lattice_loop
def test_default(lattice):
    gf = gauge(lattice)
    assert gf.location == "CUDA"
    assert gf.reconstruct == "NO"
    assert gf.geometry == "VECTOR"


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_params(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    params = gf.quda_params
    assert gf.is_native()
    assert params.nColor == 3
    assert params.nFace == 0
    assert params.reconstruct == gf.quda_reconstruct
    assert params.location == gf.quda_location
    assert params.order == gf.quda_order
    assert params.t_boundary == gf.quda_t_boundary
    assert params.link_type == gf.quda_link_type
    assert params.geometry == gf.quda_geometry
    assert addressof(params.gauge) == gf.ptr
    assert params.Precision() == gf.quda_precision
    assert params.nDim == gf.ndims
    assert tuple(params.x)[: gf.ndims] == gf.dims
    assert params.pad == gf.pad
    assert params.ghostExchange == gf.quda_ghost_exchange


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_zero(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.zero()
    assert gf == 0
    assert gf.plaquette() == (0, 0, 0)
    assert gf.topological_charge() == (0, (0, 0, 0))
    assert gf.norm1() == 0
    assert gf.norm2() == 0
    assert gf.abs_max() == 0
    assert gf.abs_min() == 0

    assert gf.project() == 4 * np.prod(lattice)
    gf.zero()

    gf2 = gf.new()
    gf2.gaussian()
    assert gf.dot(gf2) == 0

    # Testing operators
    assert not gf
    assert gf + 0 == gf
    assert gf + 1 != gf
    assert gf - 0 == gf
    assert gf * 1 == gf
    assert gf / 1 == gf

    assert isinstance(gf + 0, type(gf))

    gf3 = momentum(lattice, dtype=dtype, device=device)
    gf3.zero()
    assert gf + gf3 == 0
    assert gf3 + gf == 0


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_unity(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.unity()
    assert gf.plaquette() == (1, 1, 1)
    topo = gf.topological_charge()
    assert np.isclose(topo[0], 0)
    assert topo[1] == (0, 0, 0)
    assert gf.norm1() == 3 * 4 * np.prod(lattice)
    assert gf.norm2() == 3 * 4 * np.prod(lattice)
    assert gf.abs_max() == 1
    assert gf.abs_min() == 0
    assert gf.project() == 0
    assert np.isclose(gf.plaquettes(), 1)
    assert np.allclose(gf.plaquette_field().trace().mean(axis=1), gf.ncol)
    assert np.allclose(gf.plaquette_field(force=True), 0)
    assert np.isclose(gf.rectangles(), 1)
    assert np.allclose(gf.rectangle_field().trace().mean(axis=1), gf.ncol)
    assert np.allclose(gf.rectangle_field(force=True), 0)
    assert np.isclose(gf.gauge_action(), 1)
    assert np.isclose(gf.symanzik_gauge_action(), 1 + 7 / 12)
    assert np.isclose(gf.iwasaki_gauge_action(), 1 + 7 * 0.331)

    gf2 = gf.new()
    gf2.gaussian()
    assert gf.dot(gf2) == gf2


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_random(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    plaq = gf.plaquette()
    total = gf.plaquettes()
    assert np.isclose(plaq[0], total)

    gf2 = gf.copy()
    assert gf == gf2
    assert isclose(gf.norm2(), (gf.field**2).sum(), rel_tol=1e-6)


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_exponential(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    mom = momentum(lattice, dtype=dtype, device=device)
    mom.zero()

    gf.unity()
    mom.copy(out=gf)
    assert np.allclose(gf.field, 0)
    # gf.is_momentum = False
    assert gf == 0

    gf.unity()
    gf2 = mom.exponentiate()
    assert gf2 == gf

    gf.unity()
    gf2 = mom.exponentiate(exact=True)
    assert gf2 == gf

    mom.gaussian(epsilon=0)
    gf2 = mom.exponentiate()
    assert np.allclose(gf.field, gf2.field)
    assert gf2 == gf

    gf.gaussian()
    gf2 = mom.exponentiate(mul_to=gf)
    assert gf2 == gf

    gf2 = gf.update_gauge(mom)
    assert gf2 == gf


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_mom_to_full(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    mom = momentum(lattice, dtype=dtype, device=device)
    mom.zero()
    mom.copy(out=gf)

    assert gf == 0
    assert (gf.trace() == 0).all()

    mom.gaussian()
    mom.copy(out=gf)

    assert gf.dagger() == -gf
    assert np.allclose(gf.trace().real, 0, atol=1e-9)

    gf2 = mom.full()
    assert gf2 == gf

    mom2 = gf.copy(out=mom.new())
    assert mom2 == mom

    mom2 = gf.to_momentum()
    assert mom2 == mom

    norm2 = 2 * (-gf.dot(gf)).reduce(mean=False)
    assert isclose(mom.norm2(), norm2, rel_tol=1e-6)


# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@epsilon_loop  # enables epsilon
def test_force(lib, lattice, device, epsilon):
    dtype = "float64"
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    mom = momentum(lattice, dtype=dtype, device=device)
    mom.gaussian(epsilon=epsilon)

    gf2 = mom.exponentiate(mul_to=gf)

    for path in "plaquette", "rectangle":
        action = getattr(gf, path + "s")()
        action2 = getattr(gf2, path + "s")()
        rel_tol = epsilon * np.prod(lattice)
        assert isclose(action, action2, rel_tol=rel_tol)

        daction = (
            getattr(gf, path + "_field")(force=True).full().dot(mom.full()).reduce()
        )
        daction2 = action2 - action
        assert isclose(daction, daction2, rel_tol=rel_tol)

        zeros = getattr(gf, path + "_field")(coeffs=0, force=True)
        assert zeros == 0


# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@epsilon_loop  # enables epsilon
def test_fermionic_force(lib, lattice, device, epsilon):
    dtype = "float64"
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    mom = momentum(lattice, dtype=dtype, device=device)
    mom.gaussian(epsilon=epsilon)

    gf2 = mom.exponentiate(mul_to=gf)

    R = spinor(lattice, dtype=dtype)
    R.gaussian()

    params = {"kappa": 0.01, "csw": 1, "computeTrLog": True}

    # U'- U ~ eps*mom where U' = exp(eps*mom)*U
    for parity in [None, "EVEN"]:
        params.update(
            {
                "full": True if parity is None else False,
                "symm": False if params["csw"] != 0 else True,
            }
        )
        D = gf.Dirac(**params)
        D2 = gf2.Dirac(**params)
        phi = D.Mdag(R)

        action = D.action(phi)
        action2 = D2.action(phi)
        rel_tol = epsilon * np.prod(lattice) * 4
        print(parity, action, action2)
        assert isclose(action, action2, rel_tol=rel_tol / 4)

        daction = D.force(phi).full().dot(mom.full()).reduce(mean=False)
        daction2 = action2 - action
        print(parity, daction, daction2, daction / daction2)
        assert isclose(daction, daction2, rel_tol=rel_tol)


# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@epsilon_loop  # enables epsilon
def _test_force_gradient(lib, lattice, device, epsilon):
    dtype = "float64"
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()

    mom1 = momentum(lattice, dtype=dtype, device=device)
    mom1.gaussian(epsilon=epsilon)

    mom2 = momentum(lattice, dtype=dtype, device=device)
    mom2.gaussian(epsilon=epsilon)

    gf1 = mom1.exponentiate(mul_to=gf)
    gf2 = mom2.exponentiate(mul_to=gf)
    gf12 = mom1.exponentiate(mul_to=gf2)
    gf21 = mom2.exponentiate(mul_to=gf1)

    rel_tol = epsilon * np.prod(lattice)
    for path in "plaquette", "rectangle":
        action = getattr(gf, path + "s")()
        action1 = getattr(gf1, path + "s")()
        action2 = getattr(gf2, path + "s")()
        action21 = getattr(gf21, path + "s")()
        action12 = getattr(gf12, path + "s")()

        ddaction21 = action21 + action - action1 - action2
        ddaction12 = action12 + action - action1 - action2

        ddaction = (
            getattr(gf, path + "_field")(force=True, grad=mom1)
            .full()
            .dot(mom2.full())
            .reduce()
        )
        assert isclose(ddaction, ddaction21, rel_tol=rel_tol)

        ddaction = (
            getattr(gf, path + "_field")(force=True, grad=mom1, left_grad=True)
            .full()
            .dot(mom2.full())
            .reduce()
        )
        assert isclose(ddaction, ddaction12, rel_tol=rel_tol)

        ddaction = (
            getattr(gf, path + "_field")(force=True, grad=mom2)
            .full()
            .dot(mom1.full())
            .reduce()
        )
        assert isclose(ddaction, ddaction12, rel_tol=rel_tol)

        ddaction = (
            getattr(gf, path + "_field")(force=True, grad=mom2, left_grad=True)
            .full()
            .dot(mom1.full())
            .reduce()
        )
        assert isclose(ddaction, ddaction21, rel_tol=rel_tol)


def roll(self, shift, axis):
    "This is an implementation of shifting via rolling"
    assert 0 <= axis < self.ndims
    lath = list(self.dims)
    lath[-1] //= 2
    out = self.default_view()
    shape = out.shape
    out = out.reshape(*shape[:-1], *lath)
    if axis == 0 and self.ndims == 4:
        if shift // 2 != 0:
            out = cp.roll(out, shift // 2, axis=-1)
    else:
        out = cp.roll(out, shift, axis=self.ndims - axis - 5)
    if shift % 2 != 0:
        # Exchanging even with odd
        out = cp.roll(out, 1, axis=0)
    return out


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_shift(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    out1 = gf.shift((1, 1))
    out2 = gf.shift((1, 0)).shift((0, 1))
    out3 = gf.shift((1, 0)).shift((-1, 0))
    out4 = gf.shift((0, 1)).shift((0, -1))

    assert gf != out1
    assert gf != out2
    assert gf == out3
    assert gf == out4
    assert out1 == out2

    for shift in -1, 1, -2, 2:
        for axis in range(4):
            if axis == 0 and shift % 2 == 1:
                continue
            shifts = [0] * 4
            shifts[axis] = -shift
            out1 = gf.shift(shifts).default_view()
            out2 = roll(gf, shift, axis)
            out1 = out1.reshape(out2.shape)
            assert (out1 == out2).all()
