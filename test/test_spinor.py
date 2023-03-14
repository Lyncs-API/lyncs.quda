from lyncs_quda import spinor
import numpy as np
import cupy as cp
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    device_loop,
    dtype_loop,
    gamma_loop,
)
from lyncs_cppyy.ll import addressof


@lattice_loop
def test_default(lattice):
    sf = spinor(lattice)
    assert sf.location == "CUDA"
    assert sf.ncolor == 3
    assert sf.nspin == 4
    assert sf.nvec == 1


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_params(lib, lattice, device, dtype):
    sf = spinor(lattice, dtype=dtype, device=device)
    params = sf.quda_params
    if dtype == "float64":  # single wants order float4
        assert sf.is_native()
    assert params.nColor == sf.ncolor
    assert params.nSpin == sf.nspin
    assert params.nVec == sf.nvec
    assert params.gammaBasis == sf.gamma_basis
    assert params.pc_type == sf.pc_type

    assert params.location == sf.location
    assert params.fieldOrder == sf.order
    assert params.siteOrder == sf.site_order
    assert addressof(params.v) == sf.ptr
    assert params.Precision() == sf.precision
    assert params.nDim == sf.ndims
    assert tuple(params.x)[: sf.ndims] == sf.dims
    assert params.pad == sf.pad
    assert params.ghostExchange == sf.ghost_exchange


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_init(lib, lattice, device, dtype):
    sf = spinor(lattice, dtype=dtype, device=device)
    sf.zero()
    assert (sf == 0).all()
    assert sf.norm1() == 0
    assert sf.norm2() == 0
    sf.uniform()
    assert np.isclose(sf.mean(), 0.5 + 0.5j, atol=0.1)
    sf.gaussian()
    assert np.isclose(sf.mean(), 0.0, atol=0.1)


# @dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
@gamma_loop  # enables gamma
def test_gamma5(lib, lattice, device, gamma, dtype=None):
    sf = spinor(lattice, dtype=dtype, device=device, gamma_basis=gamma)
    sf.uniform()
    sf2 = sf.gamma5().apply_gamma5()
    assert (sf == sf2).all()
