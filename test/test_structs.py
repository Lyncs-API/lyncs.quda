from lyncs_quda import structs
from lyncs_quda.testing import fixlib as lib


def test_assign_zero(lib):
    for struct in dir(structs):
        if struct.startswith("_") or struct == "Struct":
            continue

        params = getattr(structs, struct)()

        for key in params.keys():
            setattr(params, key, 0)
