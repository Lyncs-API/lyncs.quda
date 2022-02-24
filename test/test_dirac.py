from random import random
import numpy as np
from lyncs_quda import gauge, spinor, gauge_coarse, gauge_scalar, spinor_coarse, clover_coarse
from lyncs_quda.lattice_field import get_precision
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    device_loop,
    dtype_loop,
    mtype_loop,
    gamma_loop,
)


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_params(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    dirac = gf.Dirac()
    params = dirac.quda_params
    assert dirac.precision == gf.precision
    assert params.type == dirac.quda_type
    assert params.kappa == dirac.kappa
    assert params.m5 == dirac.m5
    assert params.Ls == dirac.Ls
    assert params.mu == dirac.mu
    assert params.epsilon == dirac.epsilon


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_matrix(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    dirac = gf.Dirac()
    matrix = dirac.M
    assert matrix.key == "M"
    assert "Wilson" in matrix.name
    assert matrix.shift == 0
    assert matrix.precision == get_precision(dtype)
    assert matrix.flops == 0
    assert matrix.hermitian == False
    assert matrix.is_wilson == True
    assert matrix.is_staggered == False
    assert matrix.is_dwf == False
    assert matrix.is_coarse == False


@mtype_loop # enables matrix type
# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@gamma_loop  # enables gamma
def test_zero(lib, lattice, device, gamma, mtype, dtype=None):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.zero()
    sf = spinor(lattice, dtype=dtype, device=device, gamma_basis=gamma)
    sf.uniform()
    kappa = random()
    dirac = gf.Dirac(kappa=kappa)
    assert (dirac.M(sf).field == sf.field).all()
    assert (dirac.Mdag(sf).field == sf.field).all()
    assert (dirac.MdagM(sf).field == sf.field).all()
    assert (dirac.MMdag(sf).field == sf.field).all()

    mu = random()
    dirac = gf.Dirac(kappa=kappa, mu=mu)
    sfmu = (2 * kappa * mu) * 1j * sf.gamma5().field
    assert np.allclose(dirac.M(sf).field, sf.field + sfmu)
    assert np.allclose(dirac.Mdag(sf).field, sf.field - sfmu)
    assert np.allclose(dirac.MdagM(sf).field, (1 + (2 * kappa * mu) ** 2) * sf.field)
    assert np.allclose(dirac.MMdag(sf).field, (1 + (2 * kappa * mu) ** 2) * sf.field)

    csw = random()
    mu = mtype*mu
    dirac = gf.Dirac(kappa=kappa, mu=mu, csw=csw)
    sfmu = (2 * kappa * mu) * 1j * sf.gamma5().field
    assert np.allclose(dirac.M(sf).field, sf.field + sfmu)
    assert np.allclose(dirac.Mdag(sf).field, sf.field - sfmu)
    assert np.allclose(dirac.MdagM(sf).field, (1 + (2 * kappa * mu) ** 2) * sf.field)
    assert np.allclose(dirac.MMdag(sf).field, (1 + (2 * kappa * mu) ** 2) * sf.field)


# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_coarse_zero(lib, lattice, device, dtype=None):
    dtype = "float32"
    gf = gauge_coarse(lattice, dtype=dtype, device=device)
    gf.zero()
    gf2 = gauge_scalar(lattice, dtype=dtype, dofs=2 * 48**2, device=device)
    gf2.unity()
    gf3 = gauge_scalar(lattice, dtype=dtype, dofs=2 * 48**2, device=device)
    gf3.unity()
    gf4 = gauge_scalar(lattice, dtype=dtype, dofs=2 * 48**2, device=device)
    gf4.unity()
    sf = spinor_coarse(lattice, dtype=dtype, device=device)
    sf.uniform()
    dirac = gf.Dirac(coarse_clover=gf2)#,coarse_clover_inv=gf3,coarse_precond=gf4)

    assert (dirac.M(sf).field == sf.field).all()
    assert (dirac.Mdag(sf).field == sf.field).all()
    assert (dirac.MdagM(sf).field == sf.field).all()
    assert (dirac.MMdag(sf).field == sf.field).all()

