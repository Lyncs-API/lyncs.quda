from lyncs_quda import LatticeField
import numpy as np
import cupy as cp

shape = (4, 3, 4, 4, 4, 4)


def test_precision():
    for dtype, prec in (
        (None, "DOUBLE"),
        ("float", "DOUBLE"),
        ("float64", "DOUBLE"),
        ("float32", "SINGLE"),
        ("float16", "HALF"),
        ("int", "INVALID"),
    ):
        field = LatticeField(np.zeros(shape, dtype=dtype))
        assert field.precision == prec


def test_numpy():
    field = LatticeField(np.zeros(shape))
    assert field.location == "CPU"
    assert field.dims == shape[-4:]
    assert field.dofs == shape[:-4]
    assert field.precision == "DOUBLE"


def test_cupy():
    field = LatticeField(cp.zeros(shape))
    assert field.location == "CUDA"
    assert field.dims == shape[-4:]
    assert field.dofs == shape[:-4]
    assert field.precision == "DOUBLE"