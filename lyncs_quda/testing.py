__all__ = [
    "fixlib",
    "lattice_loop",
]

from pytest import fixture, mark
from .lib import lib


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
