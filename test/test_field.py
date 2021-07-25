from lyncs_quda import LatticeField
from pytest import raises
import numpy as np
import cupy as cp

shape = (4, 3, 4, 4, 4, 4)


def test_precision():
    for dtype, prec in (
        (None, "double"),
        ("float", "double"),
        ("float64", "double"),
        ("float32", "single"),
        ("float16", "half"),
    ):
        field = LatticeField(np.zeros(shape, dtype=dtype))
        assert field.precision == prec
    with raises(ValueError):
        LatticeField(np.zeros(shape, dtype=int)).precision


def test_numpy():
    field = LatticeField(np.zeros(shape))
    assert field.location == "CPU"
    assert field.dims == shape[-4:]
    assert field.dofs == shape[:-4]
    assert field.precision == "double"


def test_cupy():
    field = LatticeField(cp.zeros(shape))
    assert field.location == "CUDA"
    assert field.dims == shape[-4:]
    assert field.dofs == shape[:-4]
    assert field.precision == "double"
