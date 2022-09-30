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
    mark_mpi,
    get_cart,
)

comm = None


@mark_mpi
@dtype_loop  # enables dtype
@device_loop  # enables device
@parallel_loop  # enables procs
@lattice_loop  # enables lattice
def test_unity(lib, lattice, procs, device, dtype):
    global comm
    if not lib.initialized:
        comm = get_cart(procs)
    lib.set_comm(comm)

    gf = gauge(lattice, dtype=dtype, device=device)
    gf.unity()
    assert gf.norm1() == 3 * 4 * np.prod(lattice)
    assert gf.norm2() == 3 * 4 * np.prod(lattice)
    assert gf.abs_max() == 1
    assert gf.abs_min() == 0
    assert gf.project() == 0
    assert gf.plaquette() == (1, 1, 1)
    topo = gf.topological_charge()
    assert np.isclose(topo[0], 0)
    assert topo[1] == (0, 0, 0)
