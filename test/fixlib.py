from pytest import fixture
from lyncs_quda import lib as _lib

# Use the fixture lib everywhere the quda library is used directly or indirectly


@fixture(scope="session")
def lib():
    "A fixture to guarantee that lib is finalized at the end"
    if not _lib.initialized:
        _lib.initQuda()
    yield _lib
    if _lib.initialized:
        _lib.endQuda()
