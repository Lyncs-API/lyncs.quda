from lyncs_quda import gauge, spinor, MultigridPreconditioner
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
def test_solve_mg_random(lib, lattice, device, gamma, dtype=None):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.gaussian()
    dirac = gf.Dirac(kappa=0.01, csw=1, computeTrLog=True, full=True)
    rhs = spinor(lattice, dtype=dtype, device=device, gamma_basis=gamma)
    rhs.uniform()
    mat = dirac.M
    prec = MultigridPreconditioner(mat.dirac)
    out = mat.solve(rhs, precon=prec, delta=1e-4)  # this value allowed convergence for all cases
    res = mat(out)
    res -= rhs
    res = res.norm() / rhs.norm()
    prec.destroyMG_solver()
    assert res < 1e-9

    if gamma == "UKQCD": # precond Dirac op works only with 
        dirac = gf.Dirac(kappa=0.01, csw=1, computeTrLog=True, full=False)
        mat = dirac.M
        prec = MultigridPreconditioner(mat.dirac)
        print(prec.inv_param.solution_type)
        pout = mat.solve(rhs, precon=prec, solution_typ=prec.inv_param.solution_type, delta=1e-4)
        res = out-pout
        res = res.norm() / pout.norm()
        prec.destroyMG_solver()
        assert res < 1e-9
    
