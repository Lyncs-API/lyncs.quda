"""
Interface to invert_quda.h
"""

__all__ = [
    "solve",
    "Solver",
]

from functools import wraps, cache
from warnings import warn
from lyncs_cppyy import nullptr, make_shared
from .dirac import Dirac, DiracMatrix
from .enums import QudaInverterType, QudaPrecision, QudaResidualType, QudaBoolean
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
        "inv_type": "bicgstab",
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

    @staticmethod
    def _init_params():
        return lib.SolverParam()

    def __init__(self, mat, **kwargs):
        self._params = self._init_params()
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
            mat = mat.get_matrix()
        if not isinstance(mat, DiracMatrix):
            raise TypeError("mat should be an instance of Dirac or DiracMatrix")
        self._mat = mat
        self._params.precision = int(mat.precision)
        # we should not call this method after setting the below fields
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
        setattr(self, key, self.mat.copy(precision=precision))
        return getattr(self, key)

    @property
    def mat_sloppy(self):
        return self._get_mat("_mat_sloppy", self.precision_sloppy)

    precision_sloppy = QudaPrecision(
        None, lpath="_params.precision_sloppy", default=lambda self: self.precision
    )

    @property
    def mat_precon(self):
        return self._get_mat("_mat_precon", self.precision_precondition)

    precision_precondition = QudaPrecision(
        None,
        lpath="_params.precision_precondition",
        default=lambda self: self.precision,
    )

    @property
    def mat_eig(self):
        return self._get_mat("_mat_eig", self.precision_eigensolver)

    precision_eigensolver = QudaPrecision(
        None, lpath="_params.precision_eigensolver", default=lambda self: self.precision
    )

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

    inv_type = QudaInverterType(None, lpath="_params.inv_type")
    inv_type_precondition = QudaInverterType(
        None, lpath="_params.inv_type_precondition"
    )

    @property
    def preconditioner(self):
        return self._precon

    @preconditioner.setter
    def preconditioner(self, value):
        if value is None:
            self._precon = None
            self._params.inv_type_precondition = int(QudaInverterType["INVALID"])
            self._params.preconditioner = nullptr
        else:
            raise NotImplementedError

    def _update_return_residual(self, old, new):
        assert self._params.return_residual == new
        self._params.compute_true_res = new
        self._params.preserve_source = not new

    return_residual = QudaBoolean(
        None, lpath="_params.return_residual", callback=_update_return_residual
    )

    residual_type = QudaResidualType(
        None, lpath="_params.residual_type", default="L2_RELATIVE"
    )

    @property
    def quda(self):
        # self._params.preserve_source=lib.QUDA_PRESERVE_SOURCE_YES #see above
        # self._params.compute_true_res = True #see above
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

    def __call__(self, rhs, out=None, warning=True, **kwargs):
        rhs = spinor(rhs)
        out = rhs.prepare_out(out)
        kwargs = self.swap(**kwargs)
        # ASSUME: QUDA_FULL_SITE_SUBSET
        if self.mat.dirac.full:
            self.quda(out.quda_field, rhs.quda_field)
        elif self.mat.dirac.even:
            self.quda(out.quda_field.Even(), rhs.quda_field.Even())
        else:
            self.quda(out.quda_field.Odd(), rhs.quda_field.Odd())
        self.swap(**kwargs)

        if self.true_res > self.tol:
            msg = f"Solver did not converge. Residual: {self.true_res}"
            if warning:
                warn(msg)
            else:
                raise RuntimeError(msg)

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
