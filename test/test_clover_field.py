from lyncs_quda import gauge, CloverField
from lyncs_utils import prod
import numpy as np
import cupy as cp
from lyncs_quda.testing import fixlib as lib, lattice_loop, device_loop, dtype_loop
from lyncs_quda.lattice_field import get_ptr
from lyncs_cppyy.ll import addressof


@lattice_loop
def test_default(lattice):
    gf = gauge(lattice)
    clv = CloverField(gf)
    assert clv.location == "CUDA"
    assert clv.csw == 0
    assert clv.twisted == False
    assert clv.mu2 == 0
    assert clv.rho == 0
    assert clv.computeTrLog == False


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_params(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device) 
    clv = CloverField(gf)
    params = clv.quda_params
    print(cp.cuda.runtime.getDeviceCount(),clv.quda_precision, clv.Bytes, clv.Norm_bytes) #norm_bytes=0
    #assert False
    assert clv.is_native()
    assert params.direct == True
    assert params.inverse == True
    assert addressof(params.clover) == get_ptr(clv.clover_field)
    assert addressof(params.norm) == get_ptr(clv.clover_norm)
    assert addressof(params.cloverInv) == get_ptr(clv.inverse_field)
    assert addressof(params.normInv) == get_ptr(clv.inverse_norm)
    assert params.csw == clv.csw
    assert params.twisted == clv.twisted
    assert params.mu2 == clv.mu2
    assert params.rho == clv.rho
    assert params.order == clv.quda_order
    assert params.create == lib.QUDA_REFERENCE_FIELD_CREATE
    assert params.location == clv.quda_location
    assert params.Precision() == clv.quda_precision
    assert params.nDim == clv.ndims
    assert tuple(params.x)[: clv.ndims] == clv.dims
    assert params.pad == clv.pad
    assert params.ghostExchange == clv.quda_ghost_exchange


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_zero(lib, lattice, device, dtype):
    # TODO: fix the issue when computeTrLog=True
    mu2=2.
    d = 1/(1+mu2)/2
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.zero()
    clv = CloverField(gf, csw=1., twisted=True, mu2=mu2, computeTrLog=False)
    idof = int(((clv.ncol*clv.nspin)**2/2))
    if dtype is 'float64':
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6,:] = 0.5
    else:
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0,:,:] = 0.5
        tmp[:, :, 1,:,0:2] = 0.5
    assert np.allclose(clv.clover_field.flatten(), tmp.flatten()) 
    assert (clv.clover_norm == 0).all()
    if dtype is 'float64':
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6,:] = d
    else:
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0,:,:] = d
        tmp[:, :, 1,:,0:2] = d
    # Here, create a lattice of the same dims, specified in the argument in this function  on each device.  So comparison with the numpy array of dims=lattice works
    assert np.allclose(clv.inverse_field.flatten(), tmp.flatten())
    assert (clv.clover_norm == 0).all()
    # QUDA turns CloverField into basically a pointer to complex numbers
    #  When computing norm's and abs', QUDA removes the internal factor of 1/2, present in the above
    assert np.isclose(clv.norm1(), np.sqrt(2)*prod(lattice)*6)
    assert np.isclose(clv.norm2(), prod(lattice)*6)
    assert np.isclose(clv.norm1(True), 2*np.sqrt(2*d**2)*prod(lattice)*6)
    assert np.isclose(clv.norm2(True), 2*2*d**2*prod(lattice)*6)
    assert np.isclose(clv.abs_max(), np.sqrt(2))
    assert np.isclose(clv.abs_min(), 0.)
    assert np.isclose(clv.abs_max(True), 2*np.sqrt(2*d**2))
    assert np.isclose(clv.abs_min(True), 0.)
    if clv.computeTrLog:
        assert (clv.trLog == 0).all() 
    else:
        assert clv.trLog == None


@dtype_loop  # enables dtype
@device_loop  # enables device
@lattice_loop  # enables lattice
def test_unit(lib, lattice, device, dtype):
    gf = gauge(lattice, dtype=dtype, device=device)
    gf.unity()
    clv = CloverField(gf, csw=1., twisted=True, mu2=1.)
    assert (clv.field == 0).all()
    idof = int(((clv.ncol*clv.nspin)**2/2))
    if dtype is 'float64':
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6,:] = 0.5
    else:
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0,:,:] = 0.5
        tmp[:, :, 1,:,0:2] = 0.5
    assert np.allclose(clv.clover_field.flatten(), tmp.flatten())
    assert (clv.clover_norm == 0).all()
    if dtype is 'float64':
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 36, -1))
        tmp[:, :, 0:6,:] = 0.25
    else:
        tmp = np.zeros((idof,)+lattice,dtype=dtype).reshape((2, 2, 9, -1, 4))
        tmp[:, :, 0,:,:] = 0.25
        tmp[:, :, 1,:,0:2] = 0.25
    assert np.allclose(clv.inverse_field.flatten(), tmp.flatten())
    assert (clv.inverse_norm == 0).all()
    assert np.isclose(clv.norm1(), np.sqrt(2)*prod(lattice)*6)
    assert np.isclose(clv.norm2(), prod(lattice)*6)
    assert np.isclose(clv.norm1(True), np.sqrt(1/2)*prod(lattice)*6)
    assert np.isclose(clv.norm2(True), prod(lattice)*6/4)
    assert np.isclose(clv.abs_max(), np.sqrt(2))
    assert np.isclose(clv.abs_min(), 0.)
    assert np.isclose(clv.abs_max(True), np.sqrt(1/2))
    assert np.isclose(clv.abs_min(True), 0.)
    if clv.computeTrLog:
        assert (clv.trLog == 0).all() 
    else:
        assert clv.trLog == None

