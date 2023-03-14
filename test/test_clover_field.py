from lyncs_quda import gauge, gauge_tensor, CloverField
from lyncs_utils import prod
import numpy as np
import cupy as cp
from lyncs_quda.testing import fixlib as lib, lattice_loop, device_loop, dtype_loop
from lyncs_cppyy.ll import addressof
from lyncs_quda.lattice_field import get_ptr

# TODO
# When compiled with QUDA_CLOVER_RECONSTRUCT=ON, data field check should be modified
#  In addition if QUDA_CLOVER_DYNAMIC=ON, inversio does not work as reconstruct is set True


@lattice_loop
def test_default(lattice):
    gf = gauge(lattice)
    clv = CloverField(gf)
    assert clv.location == "CUDA"
    assert clv.reconstruct == False
    assert clv.coeff == 0
    assert clv.twisted == False
    assert clv.twist_flavor == "NO"
    assert clv.mu2 == 0
    assert clv.eps2 == 0
    assert clv.rho == 0
    assert clv.computeTrLog == False


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_params(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    clv = CloverField(gf, computeTrLog=True, coeff=0)
    params = clv.quda_params

    assert clv.is_native()
    assert params.inverse == True
    assert addressof(params.clover) == get_ptr(clv.clover_field)
    assert addressof(params.cloverInv) == get_ptr(clv.inverse_field)
    assert params.coeff == clv.coeff
    assert params.twisted == clv.twisted
    assert params.twist_flavor == lib.QUDA_TWIST_NO
    assert params.mu2 == clv.mu2
    assert params.epsilon2 == clv.eps2
    assert params.rho == clv.rho
    assert params.order == clv.order
    assert params.create == lib.QUDA_REFERENCE_FIELD_CREATE
    assert params.location == clv.location
    assert params.Precision() == clv.precision
    assert params.nDim == clv.ndims
    assert tuple(params.x)[: clv.ndims] == clv.dims
    assert params.pad == clv.pad
    assert params.ghostExchange == clv.ghost_exchange


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_zero(lib, lattice, device, dtype):
    d = 0.5
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.zero()
    clv = CloverField(gf, coeff=1.0, computeTrLog=True)
    clv.diagonal = 0

    idof = int(((clv.ncol * clv.nspin) ** 2 / 2))
    if dtype == "float64":
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6, :] = 0.5
    else:
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0, :, :] = 0.5
        tmp[:, :, 1, :, 0:2] = 0.5
    assert np.allclose(clv.clover_field.flatten(), tmp.flatten())
    if dtype == "float64":
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6, :] = d
    else:
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0, :, :] = d
        tmp[:, :, 1, :, 0:2] = d
    # Here, create a lattice of the same dims, specified in the argument in this function  on each device.  So comparison with the numpy array of dims=lattice works
    assert np.allclose(clv.inverse_field.flatten(), tmp.flatten())
    # For norm1, norm2,
    #  QUDA turns CloverField into basically a pointer to complex numbers so that 12 entries of .5 becomes 6 entries of 0.5*(1+I)
    # For abs_max, abs_min
    #  QUDA treats fields as fields of reals and finds L-inf norm of the sequence
    # Norm factor
    #  If not QUDA_PACKED_CLOVER_ORDER, norm factor of 2 is applied to the sequence
    #  If QUDA_PACKED_CLOVER_ORDER, the factor == 1
    assert np.isclose(clv.norm1(), 2 * np.sqrt(0.5) * prod(lattice) * 6)
    assert np.isclose(clv.norm2(), 4 * 0.5**2 * 2 * prod(lattice) * 6)
    assert np.isclose(clv.norm1(True), 2 * np.sqrt(2 * d**2) * prod(lattice) * 6)
    assert np.isclose(clv.norm2(True), 4 * 2 * d**2 * prod(lattice) * 6)
    assert np.isclose(clv.abs_max(), 2 * 0.5)
    assert np.isclose(clv.abs_min(), 0.0)
    assert np.isclose(clv.abs_max(True), 2 * d)
    assert np.isclose(clv.abs_min(True), 0.0)
    if clv.computeTrLog:
        assert (clv.trLog == 0).all()
    else:
        assert clv.trLog is None


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_unit(lib, lattice, device, dtype):
    mu2 = 2.0
    d = 1 / (1 + mu2) / 2

    gf = gauge(lattice, dtype=dtype, device=device)
    gf.unity()
    clv = CloverField(
        gf, coeff=1.0, tf="SINGLET", twisted=True, mu2=mu2, computeTrLog=True
    )

    clv.fill(0)
    assert clv == 0
    assert clv + 0 == 0
    idof = int(((clv.ncol * clv.nspin) ** 2 / 2))
    if dtype == "float64":
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6, :] = 0.5
    else:
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0, :, :] = 0.5
        tmp[:, :, 1, :, 0:2] = 0.5
    assert np.allclose(clv.clover_field.flatten(), tmp.flatten())
    if dtype == "float64":
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6, :] = d
    else:
        tmp = np.zeros((idof,) + lattice, dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0, :, :] = d
        tmp[:, :, 1, :, 0:2] = d
    assert np.allclose(clv.inverse_field.flatten(), tmp.flatten())
    assert np.isclose(clv.norm1(), np.sqrt(2) * prod(lattice) * 6)
    assert np.isclose(clv.norm2(), 2 * prod(lattice) * 6)
    assert np.isclose(clv.norm1(True), 2 * np.sqrt(2 * d**2) * prod(lattice) * 6)
    assert np.isclose(clv.norm2(True), 4 * 2 * d**2 * prod(lattice) * 6)
    assert np.isclose(clv.abs_max(), 2 * 0.5)
    assert np.isclose(clv.abs_min(), 0.0)
    assert np.isclose(clv.abs_max(True), 2 * d)
    assert np.isclose(clv.abs_min(True), 0.0)
    if clv.computeTrLog:
        assert np.allclose(
            clv.trLog, np.log(1 / (2 * d)) * prod(lattice) * 6 * np.ones(2)
        )
    else:
        assert clv.trLog == None
