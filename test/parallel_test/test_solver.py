from lyncs_quda import gauge, spinor
import pytest
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    parallel_loop,
    device_loop,
    gamma_loop,
    mark_mpi,
    get_cart,
)


comm = None
@mark_mpi
@device_loop  # enables device
@parallel_loop  # enables procs  
@lattice_loop  # enables lattice
@gamma_loop  # enables gamma
def test_solve_random(lib, lattice, procs, device, gamma, dtype=None):
    global comm
    if not lib.initialized:
        comm = get_cart(procs)
        lib.set_comm(comm)
    gf = gauge(lattice, dtype=dtype, device=device, comm=comm)
    gf.gaussian()
    dirac = gf.Dirac(kappa=0.01)
    rhs = spinor(lattice, dtype=dtype, device=device, gamma_basis=gamma, comm=comm)
    rhs.uniform()
    mat = dirac.M
    out = mat.solve(rhs, delta=10.) #magic happends, and solver now converges with chiral with this value
    res = mat(out)
    res.field -= rhs.field
    res = res.norm() / rhs.norm()
    assert res < 1e-9
