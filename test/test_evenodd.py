import pytest
from lyncs_quda import evenodd, continous
import numpy as np


@pytest.fixture(params=[(4,), (4, 8), (4, 4, 8), (4, 6, 8, 2)])
def shape(request):
    return request.param


@pytest.fixture(params=[(1,), (4,), (3, 3), (4, 4, 4)])
def inner(request):
    return request.param


@pytest.fixture(params=[(1,), (4,), (3, 3), (4, 4, 4)])
def outer(request):
    return request.param


def test_evenodd(shape, inner, outer):
    tile = np.array([1, -1])
    for i in range(1, len(shape)):
        tile = np.array([tile, tile * -1])

    arr = np.tile(tile, shape)
    shape = arr.shape
    out = evenodd(arr)

    assert (continous(out) == arr).all()

    out = out.flatten()
    n = out.shape[0] // 2

    assert (out[:n] > 0).all()
    assert (out[n:] < 0).all()

    arr = np.outer(arr, np.ones(inner, dtype="int")).reshape(shape + inner)
    axes = np.arange(len(shape))
    out = evenodd(arr, axes=axes)

    assert (continous(out, axes=axes) == arr).all()

    out = out.flatten()
    n = out.shape[0] // 2

    assert (out[:n] > 0).all()
    assert (out[n:] < 0).all()

    arr = np.outer(np.ones(outer, dtype="int"), arr).reshape(outer + shape + inner)
    axes += len(outer)
    out = evenodd(arr, axes=axes)

    assert (continous(out, axes=axes) == arr).all()

    out = out.reshape((np.prod(outer), -1))
    n = out.shape[1] // 2

    assert (out[:, :n] > 0).all()
    assert (out[:, n:] < 0).all()

    arr = np.random.rand(*(outer + shape + inner))
    out = evenodd(arr, axes=axes)
    assert (continous(out, axes=axes) == arr).all()
