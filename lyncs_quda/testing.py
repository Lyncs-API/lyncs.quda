"""
Import this file only if in a testing environment
"""

__all__ = [
    "fixlib",
    "lattice_loop",
    "device_loop",
    "parallel_loop",
    "mark_mpi",
]

from itertools import product
from pytest import fixture, mark
from lyncs_utils import factors, prod
from .lib import lib, QUDA_MPI
from .spinor_field import SpinorField

@fixture(scope="session")
def fixlib():
    "A fixture to guarantee that in pytest lib is finalized at the end"
    if not lib.initialized:
        lib.init_quda()
    yield lib
    if lib.initialized:
        lib.end_quda()


lattice_loop = mark.parametrize(
    "lattice",
    [
        # (2, 2, 2, 2),
        # (3, 3, 3, 3),
        (4, 4, 4, 4),
        (8, 8, 8, 8),
    ],
)

device_loop = mark.parametrize(
    "device",
    [
        True,
        # False,
    ],
)

dtype_loop = mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
        # "float16",
    ],
)

gamma_loop = mark.parametrize(
    "gamma", SpinorField.gammas)


def get_procs_list(comm_size=None, max_size=None):
    if comm_size is None:
        if not QUDA_MPI:
            return [
                None,
            ]
        comm_size = lib.MPI.COMM_WORLD.size
    facts = {1} | set(factors(comm_size))
    procs = list(
        set(procs for procs in product(facts, repeat=4) if prod(procs) == comm_size)
    )
    if not max_size:
        return procs
    return procs[:max_size]


def get_cart(procs=None, comm=None):
    if not QUDA_MPI or procs is None:
        return None
    if comm is None:
        comm = lib.MPI.COMM_WORLD
    return comm.Create_cart(procs)


parallel_loop = mark.parametrize("procs", get_procs_list(max_size=1))

mark_mpi = mark.mpi(min_size=1)
