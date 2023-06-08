from lyncs_quda import structs  # This is also importing Enum
from lyncs_quda.enum import Enum
from lyncs_quda.testing import fixlib as lib

def test_assign_zero(lib):
    for struct in dir(structs):

        if struct.startswith("_") or struct == "Struct" or struct == "Enum" or issubclass(getattr(structs, struct), Enum):
            continue
            
        params = getattr(structs, struct)()
        for key in params.keys():
            typ = getattr(structs, struct)._types[key]
            obj = getattr(structs, typ) if typ in dir(structs) else None
            val = 0
            if obj != Enum and issubclass(obj, Enum):
                val = list(obj.values())[0]
            elif "*" in typ: # cannot set a pointer field to nullptr via cppyy
                continue
            print("tst",struct,key,typ, obj,val)

            setattr(params, key, val)

def test_assign_something(lib):
    mp = structs.QudaMultigridParam()
    ip = structs.QudaInvertParam()
    ep = structs.QudaEigParam()

    # ptr to strct class works
    mp.n_level = 3 # This is supposed to be set explicitly
    mp.invert_param = ip.quda
    print(ep.quda)
    lib.set_mg_eig_param["QudaEigParam", lib.QUDA_MAX_MG_LEVEL](mp.eig_param, ep.quda, 0)
    ip.split_grid = list(range(lib.QUDA_MAX_DIM))
    ip.madwf_param_infile = "hi I'm here!"
    mp.geo_block_size = [[i+j+1 for j in range(lib.QUDA_MAX_DIM)] for i in range(lib.QUDA_MAX_MG_LEVEL)]
    mp.vec_infile = ["infile" + str(i) for i in range(lib.QUDA_MAX_MG_LEVEL)]
    mp.printf()
    print(ip.madwf_param_infile)
