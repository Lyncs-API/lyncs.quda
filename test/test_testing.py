from lyncs_quda.testing import get_procs_list


def test_get_procs_list():
    assert get_procs_list(comm_size=1) == [(1, 1, 1, 1)]
    assert len(get_procs_list(comm_size=2)) == 4
    assert len(get_procs_list(comm_size=3)) == 4
