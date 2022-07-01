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

comm=None
@mark_mpi
@dtype_loop  # enables dtype
@device_loop  # enables device
@parallel_loop  # enables procs
@lattice_loop  # enables lattice
def test_unity(lib, lattice, procs, device,dtype):
    #dtype= dtype = "float32"
    global comm
    if not lib.initialized:
        comm = get_cart(procs)
        lib.set_comm(comm)
        lib.init_quda()
    gf = gauge(lattice, dtype=dtype, device=device, comm=comm)
    gf.unity()
    print("after",gf.device,lib.device_id,cp.cuda.runtime.getDevice(),lib.comm_gpuid())
    assert gf.norm1() == 3 * 4 * np.prod(lattice) * np.prod(procs)
    print("tes 2")
    assert gf.norm2() == 3 * 4 * np.prod(lattice) * np.prod(procs)
    print("tes3")
    assert gf.abs_max() == 1
    assert gf.abs_min() == 0
    print("tes4")
    assert gf.project() == 0
    print("tes5")
    assert gf.plaquette() == (1, 1, 1)
    print("tes6")
    print("after2",gf.device,lib.device_id,cp.cuda.runtime.getDevice(),lib.comm_gpuid())
    topo = gf.topological_charge()
    print("tes7")
    assert np.isclose(topo[0], 0) #-3 / 4 / np.pi**2 * np.prod(lattice) * np.prod(procs))
    assert topo[1] == (0, 0, 0)
    #lib.end_quda()
