from lyncs_quda import gauge, spinor
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    device_loop,
    dtype_loop,
    gamma_loop,
)


@device_loop  # enables device
@lattice_loop  # enables lattice
@gamma_loop  # enables gamma
def test_solve_random(lib, lattice, device, gamma, dtype=None):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    dirac = gf.Dirac(kappa=0.01)
    rhs = spinor(lattice, dtype=dtype, device=device, gamma_basis=gamma)
    rhs.uniform()
    mat = dirac.M
    out = mat.solve(rhs, delta=10.0)  # this value allowed convergence for all cases
    res = mat(out)
    res.field -= rhs.field
    res = res.norm() / rhs.norm()
    assert res < 1e-9
