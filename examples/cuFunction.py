from numba import cuda, int32
from numba.types import CPointer
import cupy


def get_function_name(ptx):
    lines = ptx.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith(".visible .entry"):
            return line.split()[2].strip("(")
    raise RuntimeError


def add(x, y, i):
    x[i] = x[i] + y


def add_host(x, y, n):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    i = tx + ty * bw
    if i < n:  # Check array boundaries
        add(x, y, i)


ptx = cuda.compile_ptx(add, (CPointer(int32), int32, int32))
print(ptx)
ptx = ptx[0]
open("foo.ptx", "w").write(ptx)

name = get_function_name(ptx)
print(name)
module = cupy.cuda.Module()
module.load(bytes(ptx, "utf-8"))
function = module.get_function(name)
print(function.ptr)
