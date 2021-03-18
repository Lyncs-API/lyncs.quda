from lyncs_quda import gauge
import numpy as np
import cupy as cp

lattice = (4, 4, 4, 4)


def test_default():
    gf = gauge(lattice)
    assert gf.location == "CUDA"
    assert gf.reconstruct == "NO"
    assert gf.geometry == "VECTOR"


def test_zero():
    gf = gauge(lattice)
    gf.zero()
    assert (gf.field == 0).all()
    assert gf.plaquette() == (0, 0, 0)
    assert gf.topological_charge() == (0, (0, 0, 0))
    assert gf.norm1() == 0
    assert gf.norm2() == 0
    assert gf.abs_max() == 0
    assert gf.abs_min() == 0

    assert gf.project() == 4 * np.prod(lattice)
    gf.zero()


def test_unity():
    gf = gauge(lattice)
    gf.unity()
    assert gf.plaquette() == (1, 1, 1)
    topo = gf.topological_charge()
    assert np.isclose(topo[0], -3 / 4 / np.pi ** 2 * np.prod(lattice))
    assert topo[1] == (0, 0, 0)
    assert gf.norm1() == 3 * 4 * np.prod(lattice)
    assert gf.norm2() == 3 * 4 * np.prod(lattice)
    assert gf.abs_max() == 1
    assert gf.abs_min() == 0
    assert gf.project() == 0
    # assert (gf.plaquette_field() == 1).all()
    # assert (gf.rectangle_field() == 1).all()
