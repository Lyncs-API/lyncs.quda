from lyncs_quda import gauge, momentum
import numpy as np
import cupy as cp
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    device_loop,
    dtype_loop,
    epsilon_loop,
)
from lyncs_cppyy.ll import addressof
from math import prod, isclose


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
    assert (gf.field == 0).all()
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
    assert (gf.dot(gf2).field == 0).all()


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
    assert gf.norm1() == 3 * 4 * np.prod(lattice)
    assert gf.norm2() == 3 * 4 * np.prod(lattice)
    assert gf.abs_max() == 1
    assert gf.abs_min() == 0
    assert gf.project() == 0
    assert np.allclose(gf.plaquette_field().trace(), 1)
    assert np.allclose(gf.plaquette_field(force=True), 0)
    assert np.allclose(gf.rectangle_field().trace(), 1)
    assert np.allclose(gf.rectangle_field(force=True), 0)
    assert np.isclose(gf.rectangles(), 1)
    assert np.isclose(gf.gauge_action(), 1)
    assert np.isclose(gf.symanzik_gauge_action(), 1 + 7 / 12)
    assert np.isclose(gf.iwasaki_gauge_action(), 1 + 7 * 0.331)

    gf2 = gf.new()
    gf2.gaussian()
    assert (gf.dot(gf2).field == gf2.field).all()


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_random(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    plaq = gf.plaquette()
    total = gf.plaquettes()
    assert np.isclose(plaq[0], total)
    split = gf.plaquettes(split=True)
    assert np.isclose(plaq[0], split[0])
    assert np.isclose(plaq[1], split[1])
    assert np.isclose(plaq[2], split[2])

    gf2 = gf.copy()
    assert (gf.field == gf2.field).all()


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_exponential(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    mom = momentum(lattice, dtype=dtype, device=device)
    mom.zero()

    mom.copy(out=gf)
    assert (gf.field == 0).all()

    gf.unity()
    gf2 = mom.exponentiate()
    assert (gf2.field == gf.field).all()

    mom.gaussian(epsilon=0)
    gf2 = mom.exponentiate()
    assert (gf2.field == gf.field).all()

    gf.gaussian()
    gf2 = mom.exponentiate(mul_to=gf)
    assert (gf2.field == gf.field).all()

    gf2 = gf.update_gauge(mom)
    assert (gf2.field == gf.field).all()


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_mom_to_full(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    mom = momentum(lattice, dtype=dtype, device=device)
    mom.zero()
    mom.copy(out=gf)

    assert (gf.field == 0).all()
    assert (gf.trace() == 0).all()

    mom.gaussian()
    mom.copy(out=gf)

    assert (gf.dagger().field == -gf.field).all()
    assert np.allclose(gf.trace().real, 0, atol=1e-9)


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@epsilon_loop  # enables epsilon
def test_force(lib, lattice, device, dtype, epsilon):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    mom = momentum(lattice, dtype=dtype, device=device)
    mom.gaussian(epsilon=epsilon)

    gf2 = mom.exponentiate(mul_to=gf)

    plaq = gf.plaquette()[0]
    plaq2 = gf2.plaquette()[0]
    rel_tol = epsilon * prod(lattice)
    assert isclose(plaq, plaq2, rel_tol=rel_tol)

    dplaq = gf.plaquette_field().dot(mom).reduce()
    dplaq2 = plaq2 - plaq
    assert isclose(dplaq, dplaq2, rel_tol=rel_tol)
