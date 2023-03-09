from random import random
import numpy as np
from lyncs_utils import isclose
from lyncs_quda import (
    gauge,
    momentum,
    spinor,
    gauge_coarse,
    gauge_scalar,
    spinor_coarse,
)
from lyncs_quda.lattice_field import get_precision
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    device_loop,
    dtype_loop,
    mu_loop,
    gamma_loop,
    epsilon_loop,
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


@mu_loop  # enables mu
# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@gamma_loop  # enables gamma
def test_zero(lib, lattice, device, gamma, mu, dtype=None):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.zero()
    sf = spinor(lattice, dtype=dtype, device=device, gamma_basis=gamma)
    sf.uniform()
    kappa = random()
    dirac = gf.Dirac(kappa=kappa)
    assert (dirac.M(sf) == sf).all()
    assert (dirac.Mdag(sf) == sf).all()
    assert (dirac.MdagM(sf) == sf).all()
    assert (dirac.MMdag(sf) == sf).all()

    dirac = gf.Dirac(kappa=kappa, mu=mu)
    sfmu = (2 * kappa * mu) * 1j * sf.gamma5()
    assert np.allclose(dirac.M(sf), sf + sfmu)
    assert np.allclose(dirac.Mdag(sf), sf - sfmu)
    assert np.allclose(dirac.MdagM(sf), (1 + (2 * kappa * mu) ** 2) * sf)
    assert np.allclose(dirac.MMdag(sf), (1 + (2 * kappa * mu) ** 2) * sf)

    csw = random()
    dirac = gf.Dirac(kappa=kappa, mu=mu, csw=csw)
    sfmu = (2 * kappa * mu) * 1j * sf.gamma5()
    assert np.allclose(dirac.M(sf), sf + sfmu)
    assert np.allclose(dirac.Mdag(sf), sf - sfmu)
    assert np.allclose(dirac.MdagM(sf), (1 + (2 * kappa * mu) ** 2) * sf)
    assert np.allclose(dirac.MMdag(sf), (1 + (2 * kappa * mu) ** 2) * sf)


# @dtype_loop  # enables dtype #Double precision multigrid has not been enabled
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_coarse_zero(lib, lattice, device, dtype=None):
    dtype = "float32"
    gf = gauge_coarse(lattice, dtype=dtype, device=device)
    gf.zero()
    gf2 = gauge_scalar(lattice, dtype=dtype, dofs=2 * 48**2, device=device)
    gf2.unity()
    sf = spinor_coarse(lattice, dtype=dtype, device=device)
    sf.uniform()

    dirac = gf.Dirac(clover=gf2)
    assert (dirac.M(sf) == sf).all()
    assert (dirac.Mdag(sf) == sf).all()
    assert (dirac.MdagM(sf) == sf).all()
    assert (dirac.MMdag(sf) == sf).all()


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
