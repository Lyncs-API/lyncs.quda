"""
Interface to quda::Dirac (dirac_quda.h)
"""

__all__ = [
    "Dirac",
]

from lyncs_cppyy import make_shared
from .gauge_field import gauge, GaugeField
from .spinor_field import spinor
from .lib import lib


class Dirac:
    def __init__(self, gauge, kappa=1, m5=0, Ls=0, csw=0, mu=0, epsilon=0):
        self.gauge = gauge
        self.kappa = kappa
        self.m5 = m5
        self.Ls = Ls
        self.csw = csw
        self.mu = mu
        self.epsilon = epsilon

    @property
    def type(self):
        "Type of the operator"
        if self.csw == 0:
            if self.mu == 0:
                return "WILSON"
            return "TWISTED_MASS"
        if self.mu == 0:
            return "CLOVER_WILSON"
        return "TWISTED_CLOVER"

    @property
    def quda_type(self):
        "Quda enum for quda dslash type"
        return getattr(lib, f"QUDA_{self.type}_DSLASH")

    @property
    def dagger(self):
        "If the operator is daggered"
        return "NO"

    @property
    def quda_dagger(self):
        "Quda enum for if the operator is dagger"
        return getattr(lib, f"QUDA_DAG_{self.dagger}")

    @property
    def quda_params(self):
        params = lib.DiracParam()
        params.type = self.quda_type
        params.kappa = self.kappa
        params.m5 = self.m5
        params.Ls = self.Ls
        params.mu = self.mu
        params.epsilon = self.epsilon
        params.dagger = self.quda_dagger

        # Needs to prevent the gauge field to get destroyed
        self.quda_gauge = self.gauge.quda_field
        params.gauge = self.quda_gauge

        return params

    @property
    def quda_dirac(self):
        return make_shared(lib.Dirac.create(self.quda_params))

    def _apply(self, key, _in, _out=None):
        _in = spinor(_in)
        if _out is None:
            _out = _in.new()
        _out = spinor(_out)
        getattr(self.quda_dirac, key)(_out.quda_field, _in.quda_field)
        return _out

    def M(self, spinor_in, spinor_out=None):
        "Applies M to the spinor field"
        return self._apply("M", spinor_in, _out=spinor_out)

    def MdagM(self, spinor_in, spinor_out=None):
        "Applies MdagM to the spinor field"
        return self._apply("MdagM", spinor_in, _out=spinor_out)

    def MdagMLocal(self, spinor_in, spinor_out=None):
        "Applies MdagMLocal to the spinor field"
        return self._apply("MdagMLocal", spinor_in, _out=spinor_out)

    def Mdag(self, spinor_in, spinor_out=None):
        "Applies Mdag to the spinor field"
        return self._apply("Mdag", spinor_in, _out=spinor_out)

    def MMdag(self, spinor_in, spinor_out=None):
        "Applies MMdag to the spinor field"
        return self._apply("MMdag", spinor_in, _out=spinor_out)


GaugeField.Dirac = lambda self, *args, **kwargs: Dirac(self, *args, **kwargs)
