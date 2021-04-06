from lyncs_quda import gauge, spinor
from lyncs_quda.testing import fixlib as lib, lattice_loop, device_loop, dtype_loop


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_params(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    dirac = gf.Dirac()
    params = dirac.quda_params
    assert params.type == dirac.quda_type
    assert params.kappa == dirac.kappa
    assert params.m5 == dirac.m5
    assert params.Ls == dirac.Ls
    assert params.mu == dirac.mu
    assert params.epsilon == dirac.epsilon


# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_zero(lib, lattice, device, dtype=None):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.zero()
    sf = spinor(lattice, dtype=dtype, device=device)
    sf.uniform()
    dirac = gf.Dirac(kappa=1)
    Dsf = dirac.M(sf)
    assert (dirac.M(sf).field == sf.field).all()
    assert (dirac.Mdag(sf).field == sf.field).all()
    assert (dirac.MdagM(sf).field == sf.field).all()
    assert (dirac.MMdag(sf).field == sf.field).all()
