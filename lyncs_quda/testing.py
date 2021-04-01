"""
Import this file only if in a testing environment
"""

__all__ = [
    "fixlib",
    "lattice_loop",
    "device_loop",
    "parallel_loop",
]

from itertools import product
from pytest import fixture, mark
from lyncs_utils import factors, prod
from .lib import lib, QUDA_MPI


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
