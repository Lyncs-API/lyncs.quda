from lyncs_quda import structs  # This is also importing Enum
from lyncs_quda.enum import Enum
from lyncs_cppyy import nullptr, array_to_pointers, to_pointer
from lyncs_quda.testing import fixlib as lib
#from lyncs_cppyy.ll import addressof
import cppyy.ll
import numpy as np
from array import array
import cppyy as cp

import ctypes as C
from ctypes.util import find_library

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
    mp.n_level = lib.QUDA_MAX_MG_LEVEL #needs to be supplied manually
    ip = structs.QudaInvertParam()
    ep = structs.QudaEigParam()

    # ptr to strct class works
    print("pre inv set", mp.n_level, getattr(lib.newQudaMultigridParam(),"n_level"), lib.QUDA_MAX_MG_LEVEL,hasattr(mp.invert_param,"shape"))
    mp.invert_param = cppyy.ll.addressof(ip.quda)
    ip.split_grid = list(range(lib.QUDA_MAX_DIM))
    mp.printf()

    # array of complex conversion works
    for i in range(lib.QUDA_MAX_DWF_LS):
        print("pre b_5",ip.b_5[i])
        ip.b_5[i] = complex(1,i)
        print("post b_5",ip.b_5[i])
    ip.b_5 = [complex(2,i) for i in range(lib.QUDA_MAX_DWF_LS)]
    print([(ip.b_5[i],ip.b_5[i]) for i in range(lib.QUDA_MAX_DWF_LS)])
    # array of basic types probably works
    ip.split_grid = list(range(lib.QUDA_MAX_DIM))
    # char[] needs to be set one by one
    for i, s in enumerate("hi I'm here!"):
        ip.madwf_param_infile[i] = s #"hi I'm here!"
        print(ip.madwf_param_infile[i])
    print(ip.madwf_param_infile, hasattr(ip.madwf_param_infile,"shape"))
    ip.printf()
    
    # the following works
    print(mp.geo_block_size.shape)
    for i in range(lib.QUDA_MAX_MG_LEVEL):
        for j in range(lib.QUDA_MAX_DIM):
            print("pre geo",i,j,":",mp.geo_block_size[i][j])
            mp.geo_block_size[i][j] = i+j
            print("post geo",i,j,":", mp.geo_block_size[i][j])
    mp.printf()
    # setitems does not work properly as it works wiht subviews, I suppose: it sets entries to values different from the expected: 
    l = [[i+j+1 for j in range(lib.QUDA_MAX_DIM)] for i in range(lib.QUDA_MAX_MG_LEVEL)]
    #mp.geo_block_size = array('i', sum(l,[]))# [[i+j+1 for j in range(lib.QUDA_MAX_DIM)] for i in range(lib.QUDA_MAX_MG_LEVEL)] #np.ones((lib.QUDA_MAX_MG_LEVEL,lib.QUDA_MAX_DIM), dtype="i")
    mp.geo_block_size = l #np.asarray(l,dtype="i")
    mp.printf()
    print("FRom BUUGGREFFFER")
    """
    Assume: arr is a lowlvelview obj of int carr[5][5]
    Observation: obj.shape == (5,5)
    in the following, possibly after reshaping, np.frombuffer(arr, dtype=DTYPE, count=-1) is used
    if obj is reshaped to (25,), print(v, dtype=np.int32) prints only the first 10 elems of carr but correctly
                                 print(v, dtype=np.int64) prints only the first 5 elems of carr incorectly
    if obj is not reshaped or reshaped to (5,5) redundantly, v2=v[0] and v2[0] = 10 results to segfault
    """
    arr = mp.geo_block_size
    print(arr.shape)
    #arr.reshape(arr.shape) # fails: segfault <- if you reshape it to 
    #v = np.frombuffer(arr, dtype=np.int32, count=-1)
    data_pointer = C.cast(cppyy.ll.addressof(arr),C.POINTER(C.c_int))
    v = np.ctypeslib.as_array(data_pointer,shape=arr.shape)
    print(v)
    v2 = v[0]
    for i in range(len(v2)):
        v2[i] = 100+i
    print(v)
    mp.printf()
    # the following attempt sets the entries only up to a certain point...
    a1 = np.ones((np.prod(arr.shape),),dtype=np.int32)
    cp.gbl.memcpy(to_pointer(cp.ll.addressof(mp.geo_block_size)), to_pointer(a1.__array_interface__["data"][0]), int(np.prod(arr.shape))*4)
    mp.printf()
    
    #print(list(mp.geo_block_size.reshape((lib.QUDA_MAX_MG_LEVEL,lib.QUDA_MAX_DIM))))#[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM]
    print("infile",mp.vec_infile,mp.vec_infile.shape)
    for i in range(lib.QUDA_MAX_MG_LEVEL):
        pass
        #mp.vec_infile[i] = "hi"+str(i) #error
        #d=mp.vec_infile.reshape((lib.QUDA_MAX_MG_LEVEL, ))#256))# ValueError: cannot reshape array of size 261 into shape (5,)
    #mp.vec_infile = ["hi"+str(i) for i in range(lib.QUDA_MAX_MG_LEVEL)]
    #f = array_to_pointers(mp.vec_infile)
    #mp.vec_infile.reshape(mp.vec_infile.shape)
    print(mp.vec_infile)
    #v = np.frombuffer(mp.vec_infile, dtype=np.dtype('b') , count=-1)#lib.QUDA_MAX_MG_LEVEL)
    #print(v)
    #print(np.sum([1 for i in v]), len(v),len(mp.vec_infile[0]))
    #v[0]=1 #bytes("h", 'ascii')
    sa = ["hi"+str(i) for i in range(lib.QUDA_MAX_MG_LEVEL)]
    #lib.copy_strings(mp._quda_params, lib.std.vector["std::string"](sa))
    
    #print(lib.std.vector["std::string"]([mp.vec_infile[i] for i in range(5)]))
    #lib.copy_strings(mp.vec_infile, lib.std.vector["std::string"](sa))
    sa2 = np.chararray((lib.QUDA_MAX_MG_LEVEL,256))
    for i,s in enumerate(sa):
        sa2[i] = [s[j] if j<len(s) else b"\0" for j in range(256)]
    print(sa2)
    
    mp.printf()
    lib.memcpy(to_pointer(cp.ll.addressof(mp.vec_infile)), to_pointer(sa2.__array_interface__["data"][0]),5*256)
    #mp.vec_infile = np.empty(mp.vec_infile.shape, dtype=np.dtype("b")) #np.asarray(["hi"+str(i) for i in range(lib.QUDA_MAX_MG_LEVEL)], dtype=np.dtype(str)) #[lib.std.string("hi"+str(i)) for i in range(lib.QUDA_MAX_MG_LEVEL)]
    #mp.vec_infile[0] = "hi"
    mp.printf()
    print(mp)
