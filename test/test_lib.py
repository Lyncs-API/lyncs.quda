from lyncs_quda.lib import fixlib as lib


def test_init(lib):
    assert lib.initialized
    lib.end_quda()
    assert not lib.initialized
    lib.init_quda()
    assert lib.initialized
