from lyncs_quda.lib import fixlib as lib


def test_device_count(lib):
    assert lib.get_device_count() > 0
    assert lib.get_current_device() == 0


def test_init(lib):
    assert lib.initialized
    lib.end_quda()
    assert not lib.initialized
    lib.init_quda()
    assert lib.initialized


def test_excep(lib):
    lib.test_exception()
