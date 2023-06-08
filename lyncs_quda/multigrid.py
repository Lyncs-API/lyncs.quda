"""
Interface to multigrid_solver
"""

__all__ = ["MultigridPreconditioner"]

from cppyy import bind_object
from lyncs_cppyy import nullptr
from lyncs_utils import isiterable
from .lib import lib
from .enums import QudaInverterType, QudaPrecision, QudaSolveType
from .structs import QudaInvertParam, QudaMultigridParam, QudaEigParam

class MultigridPreconditioner:
    __slots__ = ["_mg_solver", "mg_param", "inv_param"]
    
    def __init__(self, D, inv_options={}, mg_options={}, eig_options={}, is_eig=False):
        self._mg_solver = None
        self.mg_param, self.inv_param  = self.prepareParams(D, inv_options=inv_options, mg_options=mg_options, eig_options=eig_options, is_eig=is_eig)
        self.setMG_solver(self.mg_param)

    @property
    @QudaInverterType
    def inv_type_precondition(self):
        return "MG_INVERTER"

    @property
    def preconditioner(self):
        return self._mg_solver

    def prepareParams(self, D, g_options={}, inv_options={}, mg_options={}, eig_options={}, is_eig=False):
        # INPUT: D is a Dirac instance
        #        is_eig is a list of bools indicating whether eigsolver is used to generate
        #          near null-vectors at each level
        inv_param = QudaInvertParam()
        mg_param = QudaMultigridParam()
        mg_param.invert_param = inv_param.quda

        # set* are defined in set_params.cpp, setting params to vals according to the ones defined globally
        #  <- command_line_params.cpp: contains some default values for those global vars, some set to invalid
        #     <- host_utils.h provides funcs to set global vars to some meaningful vals, according to vals in command_line...
        #  <- misc.h implemented in misc.cpp
        
        # sets fields to default values
        #app = lib.make_app()
        #lib.add_multigrid_option_group(app)
        #app.parse(1,"--solve_type 2")
        # Set internal global vars to their default vals
        dslash_type = D.dslash_type #inv_param.dslash_type.upper()
        solve_type = QudaSolveType["direct"] if D.full else QudaSolveType["direct_pc"] 
        lib.dslash_type = int(dslash_type)
        lib.solve_type = int(solve_type)
        lib.setQudaPrecisions()
        lib.setQudaDefaultMgTestParams()
        lib.setQudaMgSolveTypes()
        
        

        # Set param vals to the default vals and update according to the user's specification
        D.setGaugeParam(gauge_options=g_options)
        lib.setMultigridParam(mg_param.quda)
        if not D.full: inv_param.matpc_type = int(D.matPCtype)
        inv_param.dagger = int(D.dagger)
        inv_param.cpu_prec = int(D.precision) # quda.h says this is supposed to be the prec of input fermion field
        inv_param.cuda_prec = int(D.precision)
        if "clover" in D.type:
            inv_param.compute_clover = False
            inv_param.clover_cpu_prec = int(D.clover.precision)
            inv_param.clover_cuda_prec = int(D.clover.precision)
            inv_param.clover_order = int(D.clover.order)
            inv_param.clover_location = int(D.clover.location)
            inv_param.clover_csw = D.clover.csw
            inv_param.clover_coeff = D.clover.coeff
            inv_param.clover_rho = D.clover.rho
            inv_param.compute_clover = False
            inv_param.compute_clover_inverse = False
            inv_param.return_clover = False
            inv_param.return_clover_inverse = False
        inv_param.update(inv_options)
        mg_param.update(mg_options)
        if "clover" in D.type:
            print("mult init clover")
            D.clover.clover_field
            D.clover.inverse_field
            lib.loadCloverQuda(D.clover.quda_field.V(), D.clover.quda_field.V(True), inv_param.quda)
        mg_param.invert_param = inv_param.quda #not sure if this is necessary?

        # Only these fermions are supported with MG
        print(dslash_type, type(dslash_type))
        if dslash_type != "WILSON" and dslash_type != "CLOVER_WILSON" and dslash_type != "TWISTED_MASS" and dslash_type != "TWISTED_CLOVER":
            raise ValueError(f"dslash_type {dslash_type} not supported for MG")
        # Only these solve types are supported with MG
        if solve_type != "DIRECT" and solve_type != "DIRECT_PC":
            raise ValueError(f"Solve_type {solve_type} not supported with MG. Please use QUDA_DIRECT_SOLVE or QUDA_DIRECT_PC_SOLVE")
        print(type(mg_param))
        if not isiterable(is_eig):
            is_eig = [is_eig]*mg_param.n_level
        for i, eig in enumerate(is_eig):
            eig_param = QudaEigParam()
            if eig:
                lib.setMultigridEigParam(eig_param.quda)
                eig_param.update(eig_options)
                lib.set_mg_eig_param["QudaEigParam", lib.QUDA_MAX_MG_LEVEL](mg_param.eig_param, eig_param.quda, i)
            else:
                print(mg_param.eig_param)
                lib.set_mg_eig_param["QudaEigParam", lib.QUDA_MAX_MG_LEVEL](mg_param.eig_param, eig_param.quda, i, is_null=True)  #?to_pointer + addressof?
                
        return mg_param, inv_param

    def setMG_solver(self, mg_param):
        if self._mg_solver is None:
            self._mg_solver = lib.newMultigridQuda(mg_param.quda)
        else:
            self.updateMG_solver(mg_param)
        
    def updateMG_solver(self, mg_param):
        lib.updateMultigridQuda(self._mg_solver, mg_param.quda)

    def destroyMG_solver(self):
        lib.destroyMultigridQuda(self._mg_solver)
        self._mg_solver = None

    
