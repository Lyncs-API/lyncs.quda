from random import random
import numpy as np
from lyncs_quda import gauge, spinor
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    device_loop,
    dtype_loop,
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
    assert dirac.precision == dirac.M.precision
    assert params.type == dirac.quda_type
    assert params.kappa == dirac.kappa
    assert params.m5 == dirac.m5
    assert params.Ls == dirac.Ls
    assert params.mu == dirac.mu
    assert params.epsilon == dirac.epsilon


# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@gamma_loop  # enables gamma
def test_zero(lib, lattice, device, gamma, dtype=None):
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
