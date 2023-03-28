from lyncs_quda import gauge, gauge_scalar

import pytest
from lyncs_quda.testing import (
    fixlib as lib,
    lattice_loop,
)


@lattice_loop
def test_error(lib, lattice):
    gf = gauge(lattice)
    gs = gauge_scalar(lattice)

    gf.quda_field.copy(gs.quda_field)
    with pytest.raises(lib.std.runtime_error):
        gf.quda_field.copy(gs.quda_field)
    gf.zero()
    assert gf == 0
