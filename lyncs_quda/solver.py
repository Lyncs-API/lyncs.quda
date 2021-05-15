"""
Interface to invert_quda.h
"""

__all__ = [
    "solve",
    "Solver",
]

from functools import wraps
from lyncs_cppyy import nullptr, make_shared
from .dirac import Dirac, DiracMatrix
from .enums import (
    get_inverter_type,
    get_inverter_type_quda,
    get_precision,
    get_precision_quda,
    get_residual_type,
    get_residual_type_quda,
)
from .lib import lib
from .spinor_field import spinor
from .time_profile import default_profiler, TimeProfile


def solve(mat, rhs, out=None, **kwargs):
    return Solver(mat)(rhs, out, **kwargs)


class Solver:
    __slots__ = [
        "_mat",
        "_mat_eig",
        "_mat_precon",
        "_mat_sloppy",
        "_params",
        "_precon",
        "_profiler",
        "_solver",
    ]

    default_params = {
        "inv_type": "GCR",
        "preconditioner": None,
        # "precondition_cycle":
        # "tol_precondition":
        # "maxiter_precondition":
        "deflate": False,
        # "deflation_op":
        "use_init_guess": False,
        "return_residual": False,
        "num_src": 1,
        "num_offset": 0,
        "is_preconditioner": False,
        "global_reduction": True,
        "sloppy_converge": False,
        "precision_sloppy": None,
        "precision_precondition": None,
        "precision_eigensolver": None,
        "residual_type": None,
        "delta": 1e-10,
        "use_alternative_reliable": False,
        "use_sloppy_partial_accumulator": False,
        "solution_accumulator_pipeline": 0,
        "max_res_increase": 0,  # consecutive
        "max_res_increase_total": 0,
        "max_hq_res_increase": 0,
        "max_hq_res_restart_total": 0,
        "heavy_quark_check": 0,
        "pipeline": 0,
        "tol": 1e-9,
        "tol_restart": 1e-6,
        "tol_hq": 1e-6,
        "maxiter": 100,
        # "Nsteps",
        # "Nkrylov",
        # "omega",
        # "ca_basis",
        # "ca_lambda_min",
        # "ca_lambda_max":,
        # "schwarz_type":,
    }

    def __init__(self, mat, **kwargs):
        self._params = lib.SolverParam()
        self._solver = None
        self._profiler = None
        self._precon = None
        self.mat = mat

        params = type(self).default_params.copy()
        params.update(kwargs)

        for key, val in params.items():
            setattr(self, key, val)

    @property
    def params(self):
        return {key: getattr(self, key) for key in type(self).default_params}

    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, mat):
        if isinstance(mat, Dirac):
            mat = mat.get_mat()
        if not isinstance(mat, DiracMatrix):
            raise TypeError("mat should be an instance of Dirac or DiracMatrix")
        self._mat = mat
        self._params.precision = get_precision_quda(mat.precision)
        self._mat_sloppy = None
        self._mat_precon = None
        self._mat_eig = None

    @property
    def precision(self):
        return self.mat.precision

    def _get_mat(self, key, precision):
        if self.precision == precision:
            return self.mat
        current = getattr(self, key)
        if current is not None and current.precision == precision:
            return current
        setattr(self, key, self.mat.new(precision=precision))
        return getattr(self, key)

    @property
    def mat_sloppy(self):
        return self._get_mat("_mat_sloppy", self.precision_sloppy)

    @property
    def precision_sloppy(self):
        return get_precision(self._params.precision_sloppy)

    @precision_sloppy.setter
    def precision_sloppy(self, value):
        if value is None:
            value = self.precision
        self._params.precision_sloppy = get_precision_quda(value)

    @property
    def mat_precon(self):
        return self._get_mat("_mat_precon", self.precision_precondition)

    @property
    def precision_precondition(self):
        return get_precision(self._params.precision_precondition)

    @precision_precondition.setter
    def precision_precondition(self, value):
        if value is None:
            value = self.precision
        self._params.precision_precondition = get_precision_quda(value)

    @property
    def mat_eig(self):
        return self._get_mat("_mat_eig", self.precision_eigensolver)

    @property
    def precision_eigensolver(self):
        return get_precision(self._params.precision_eigensolver)

    @precision_eigensolver.setter
    def precision_eigensolver(self, value):
        if value is None:
            value = self.precision
        self._params.precision_eigensolver = get_precision_quda(value)

    @property
    def profiler(self):
        if self._profiler is None:
            return default_profiler()
        return self._profiler

    @profiler.setter
    def profiler(self, value):
        if not isinstance(value, TimeProfile):
            raise TypeError
        self._profiler = value

    @property
    def inv_type(self):
        "Quda enum for inverter type"
        return get_inverter_type(self._params.inv_type)

    @inv_type.setter
    def inv_type(self, value):
        self._params.inv_type = get_inverter_type_quda(value)

    @property
    def inv_type_precondition(self):
        "Quda enum for inverter type"
        return get_inverter_type(self._params.inv_type_precondition)

    @property
    def preconditioner(self):
        return self._precon

    @preconditioner.setter
    def preconditioner(self, value):
        if value is None:
            self._precon = None
            self._params.inv_type_precondition = lib.QUDA_INVALID_INVERTER
            self._params.preconditioner = nullptr
        else:
            raise NotImplementedError

    @property
    def return_residual(self):
        return bool(self._params.return_residual)

    @return_residual.setter
    def return_residual(self, value):
        value = bool(value)
        self._params.return_residual = value
        self._params.compute_true_res = value
        self._params.preserve_source = not value

    @property
    def residual_type(self):
        return get_residual_type(self._params.residual_type)

    @residual_type.setter
    def residual_type(self, value):
        if value is None:
            value = "L2_RELATIVE"
        self._params.residual_type = get_residual_type_quda(value)

    @property
    def quda(self):
        if self._solver is None:
            self._solver = make_shared(
                lib.Solver.create(
                    self._params,
                    self.mat.quda,
                    self.mat_sloppy.quda,
                    self.mat_precon.quda,
                    self.mat_eig.quda,
                    self.profiler.quda,
                )
            )
        return self._solver

    def swap(self, **params):
        "Changes params to the new value and returns old"
        for key, val in params.items():
            cur = getattr(self, key)
            if cur != val:
                setattr(self, key, val)
                params[key] = cur
            else:
                del params[key]
        return params

    def __call__(self, rhs, out=None, warning=False, **kwargs):
        rhs = spinor(rhs)
        out = rhs.prepare(out)
        kwargs = self.swap(**kwargs)
        self.quda(out.quda_field, rhs.quda_field)
        self.swap(**kwargs)
        # TODO check
        print(self.run_info)
        return out

    @property
    def run_info(self):
        return {
            "secs": self.secs,
            "iter": self.iter,
            "true_res": self.true_res,
            "true_res_hq": self.true_res_hq,
            "gflops": self.gflops,
        }

    def __getattr__(self, key):
        return getattr(self._params, key)

    def __setattr__(self, key, value):
        try:
            super().__setattr__(key, value)
        except AttributeError as err:
            if key in type(self).default_params:
                setattr(self._params, key, value)
            else:
                raise err


Dirac.solve = solve
DiracMatrix.solve = solve
Dirac.Solver = wraps(Solver)(lambda *args, **kwargs: Solver(*args, **kwargs))
DiracMatrix.Solver = wraps(Solver)(lambda *args, **kwargs: Solver(*args, **kwargs))
