import pytest
from lyncs_quda.testing import (
    fixlib as lib,
    parallel_loop,
    mark_mpi,
    get_cart,
)


@mark_mpi
@parallel_loop  # enables procs
def test_comm(lib, procs):
    comm = get_cart(procs)
    lib.set_comm(comm)
    lib.init_quda()
    for i, (dim, coord) in enumerate(zip(comm.dims, comm.coords)):
        assert lib.commDimPartitioned(i) == (dim > 1)
        assert lib.commDim(i) == dim
        assert lib.commCoords(i) == coord
