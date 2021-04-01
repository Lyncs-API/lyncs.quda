from lyncs_quda import gauge
import numpy as np
import cupy as cp
from lyncs_quda.testing import fixlib as lib, lattice_loop, device_loop, dtype_loop


@lattice_loop
def test_default(lattice):
    gf = gauge(lattice)
    assert gf.location == "CUDA"
    assert gf.reconstruct == "NO"
    assert gf.geometry == "VECTOR"


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


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_unity(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.unity()
    assert gf.plaquette() == (1, 1, 1)
    topo = gf.topological_charge()
    assert np.isclose(topo[0], -3 / 4 / np.pi ** 2 * np.prod(lattice))
    assert topo[1] == (0, 0, 0)
    assert gf.norm1() == 3 * 4 * np.prod(lattice)
    assert gf.norm2() == 3 * 4 * np.prod(lattice)
    assert gf.norm1(local=True) == 3 * 4 * np.prod(lattice)
    assert gf.norm2(local=True) == 3 * 4 * np.prod(lattice)
    assert gf.abs_max() == 1
    assert gf.abs_min() == 0
    assert gf.project() == 0
    # assert (gf.plaquette_field() == 1).all()
    # assert (gf.rectangle_field() == 1).all()
