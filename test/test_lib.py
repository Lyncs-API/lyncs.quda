from fixlib import lib


def test_init(lib):
    assert lib.initialized
    lib.endQuda()
    assert not lib.initialized
    lib.initQuda()
    assert lib.initialized
