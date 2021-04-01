from lyncs_quda import gauge
import numpy as np
import cupy as cp
import pytest
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
    parallel_loop,
    device_loop,
    dtype_loop,
    get_cart,
)


@dtype_loop  # enables dtype
@device_loop  # enables device
@parallel_loop  # enables procs
@lattice_loop  # enables lattice
@pytest.mark.mpi(min_size=1)
def test_unity(lib, lattice, procs, device, dtype):
    comm = get_cart(procs)
    gf = gauge(lattice, dtype=dtype, device=device, comm=comm)
    gf.unity()
    assert gf.norm1() == 3 * 4 * np.prod(lattice) * np.prod(procs)
    assert gf.norm2() == 3 * 4 * np.prod(lattice) * np.prod(procs)
    assert gf.norm1(local=True) == 3 * 4 * np.prod(lattice)
    assert gf.norm2(local=True) == 3 * 4 * np.prod(lattice)
    assert gf.abs_max() == 1
    assert gf.abs_min() == 0
    assert gf.project() == 0
    return
    assert gf.plaquette() == (1, 1, 1)
    topo = gf.topological_charge()
    assert np.isclose(topo[0], -3 / 4 / np.pi ** 2 * np.prod(lattice))
    assert topo[1] == (0, 0, 0)
