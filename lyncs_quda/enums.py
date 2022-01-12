"List of QUDA enumerations"

# NOTE: This file is automathically generated by setup.py
# DO NOT CHANGE MANUALLY but reinstall the package if needed

from .enum import Enum


class qudaError_t(Enum):
    """
    success = 0
    error = 1
    error_uninitialized = 2
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {"success": 0, "error": 1, "error_uninitialized": 2}


class QudaMemoryType(Enum):
    """
    device = 0
    pinned = 1
    mapped = 2
    invalid = -2147483648
    """

    _prefix = "quda_memory_"
    _suffix = ""
    _values = {"device": 0, "pinned": 1, "mapped": 2, "invalid": -2147483648}


class QudaLinkType(Enum):
    """
    su3 = 0
    general = 1
    three = 2
    momentum = 3
    coarse = 4    # used for coarse-gauge field with multigrid
    smeared = 5    # used for loading and saving gaugeSmeared in the interface
    wilson = 0    # used by wilson, clover, twisted mass, and domain wall
    asqtad_fat = 1
    asqtad_long = 2
    asqtad_mom = 3
    asqtad_general = 1
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_links"
    _values = {
        "su3": 0,
        "general": 1,
        "three": 2,
        "momentum": 3,
        "coarse": 4,
        "smeared": 5,
        "wilson": 0,
        "asqtad_fat": 1,
        "asqtad_long": 2,
        "asqtad_mom": 3,
        "asqtad_general": 1,
        "invalid": -2147483648,
    }


class QudaGaugeFieldOrder(Enum):
    """
    float = 1
    float2 = 2    # no reconstruct and double precision
    float4 = 4    # 8 reconstruct single, and 12 reconstruct single, half, quarter
    float8 = 8    # 8 reconstruct half and quarter
    native = 9    # used to denote one of the above types in a trait, not used directly
    qdp = 10    # expect *gauge[mu], even-odd, spacetime, row-column color
    qdpjit = 11    # expect *gauge[mu], even-odd, complex-column-row-spacetime
    cps_wilson = 12    # expect *gauge, even-odd, mu, spacetime, column-row color
    milc = 13    # expect *gauge, even-odd, mu, spacetime, row-column order
    milc_site = 14    # packed into MILC site AoS [even-odd][spacetime] array, and [dir][row][col] inside
    bqcd = 15    # expect *gauge, mu, even-odd, spacetime+halos, column-row order
    tifr = 16    # expect *gauge, mu, even-odd, spacetime, column-row order
    tifr_padded = 17    # expect *gauge, mu, parity, t, z+halo, y, x/2, column-row order
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_gauge_order"
    _values = {
        "float": 1,
        "float2": 2,
        "float4": 4,
        "float8": 8,
        "native": 9,
        "qdp": 10,
        "qdpjit": 11,
        "cps_wilson": 12,
        "milc": 13,
        "milc_site": 14,
        "bqcd": 15,
        "tifr": 16,
        "tifr_padded": 17,
        "invalid": -2147483648,
    }


class QudaTboundary(Enum):
    """
    anti_periodic_t = -1
    periodic_t = 1
    invalid_t_boundary = -2147483648
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {
        "anti_periodic_t": -1,
        "periodic_t": 1,
        "invalid_t_boundary": -2147483648,
    }


class QudaPrecision(Enum):
    """
    quarter = 1
    half = 2
    single = 4
    double = 8
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_precision"
    _values = {
        "quarter": 1,
        "half": 2,
        "single": 4,
        "double": 8,
        "invalid": -2147483648,
    }


class QudaReconstructType(Enum):
    """
    no = 18    # store all 18 real numbers explicitly
    12 = 12    # reconstruct from 12 real numbers
    8 = 8    # reconstruct from 8 real numbers
    9 = 9    # used for storing HISQ long-link variables
    13 = 13    # used for storing HISQ long-link variables
    10 = 10    # 10-number parameterization used for storing the momentum field
    invalid = -2147483648
    """

    _prefix = "quda_reconstruct_"
    _suffix = ""
    _values = {
        "no": 18,
        "12": 12,
        "8": 8,
        "9": 9,
        "13": 13,
        "10": 10,
        "invalid": -2147483648,
    }


class QudaGaugeFixed(Enum):
    """
    no = 0    # no gauge fixing
    yes = 1    # gauge field stored in temporal gauge
    invalid = -2147483648
    """

    _prefix = "quda_gauge_fixed_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaDslashType(Enum):
    """
    wilson = 0
    clover_wilson = 1
    clover_hasenbusch_twist = 2
    domain_wall = 3
    domain_wall_4d = 4
    mobius_dwf = 5
    mobius_dwf_eofa = 6
    staggered = 7
    asqtad = 8
    twisted_mass = 9
    twisted_clover = 10
    laplace = 11
    covdev = 12
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_dslash"
    _values = {
        "wilson": 0,
        "clover_wilson": 1,
        "clover_hasenbusch_twist": 2,
        "domain_wall": 3,
        "domain_wall_4d": 4,
        "mobius_dwf": 5,
        "mobius_dwf_eofa": 6,
        "staggered": 7,
        "asqtad": 8,
        "twisted_mass": 9,
        "twisted_clover": 10,
        "laplace": 11,
        "covdev": 12,
        "invalid": -2147483648,
    }


class QudaInverterType(Enum):
    """
    cg = 0
    bicgstab = 1
    gcr = 2
    mr = 3
    mpbicgstab = 4
    sd = 5
    pcg = 6
    mpcg = 7
    eigcg = 8
    inc_eigcg = 9
    gmresdr = 10
    gmresdr_proj = 11
    gmresdr_sh = 12
    fgmresdr = 13
    mg = 14
    bicgstabl = 15
    cgne = 16
    cgnr = 17
    cg3 = 18
    cg3ne = 19
    cg3nr = 20
    ca_cg = 21
    ca_cgne = 22
    ca_cgnr = 23
    ca_gcr = 24
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_inverter"
    _values = {
        "cg": 0,
        "bicgstab": 1,
        "gcr": 2,
        "mr": 3,
        "mpbicgstab": 4,
        "sd": 5,
        "pcg": 6,
        "mpcg": 7,
        "eigcg": 8,
        "inc_eigcg": 9,
        "gmresdr": 10,
        "gmresdr_proj": 11,
        "gmresdr_sh": 12,
        "fgmresdr": 13,
        "mg": 14,
        "bicgstabl": 15,
        "cgne": 16,
        "cgnr": 17,
        "cg3": 18,
        "cg3ne": 19,
        "cg3nr": 20,
        "ca_cg": 21,
        "ca_cgne": 22,
        "ca_cgnr": 23,
        "ca_gcr": 24,
        "invalid": -2147483648,
    }


class QudaEigType(Enum):
    """
    tr_lanczos = 0    # Thick restarted lanczos solver
    blk_tr_lanczos = 1    # Block Thick restarted lanczos solver
    ir_arnoldi = 2    # Implicitly Restarted Arnoldi solver
    blk_ir_arnoldi = 3    # Block Implicitly Restarted Arnoldi solver
    invalid = -2147483648
    """

    _prefix = "quda_eig_"
    _suffix = ""
    _values = {
        "tr_lanczos": 0,
        "blk_tr_lanczos": 1,
        "ir_arnoldi": 2,
        "blk_ir_arnoldi": 3,
        "invalid": -2147483648,
    }


class QudaEigSpectrumType(Enum):
    """
    lm_eig = 0
    sm_eig = 1
    lr_eig = 2
    sr_eig = 3
    li_eig = 4
    si_eig = 5
    invalid = -2147483648
    """

    _prefix = "quda_spectrum_"
    _suffix = ""
    _values = {
        "lm_eig": 0,
        "sm_eig": 1,
        "lr_eig": 2,
        "sr_eig": 3,
        "li_eig": 4,
        "si_eig": 5,
        "invalid": -2147483648,
    }


class QudaSolutionType(Enum):
    """
    mat = 0
    matdag_mat = 1
    matpc = 2
    matpc_dag = 3
    matpcdag_matpc = 4
    matpcdag_matpc_shift = 5
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_solution"
    _values = {
        "mat": 0,
        "matdag_mat": 1,
        "matpc": 2,
        "matpc_dag": 3,
        "matpcdag_matpc": 4,
        "matpcdag_matpc_shift": 5,
        "invalid": -2147483648,
    }


class QudaSolveType(Enum):
    """
    direct = 0
    normop = 1
    direct_pc = 2
    normop_pc = 3
    normerr = 4
    normerr_pc = 5
    normeq = 1    # deprecated
    normeq_pc = 3    # deprecated
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_solve"
    _values = {
        "direct": 0,
        "normop": 1,
        "direct_pc": 2,
        "normop_pc": 3,
        "normerr": 4,
        "normerr_pc": 5,
        "normeq": 1,
        "normeq_pc": 3,
        "invalid": -2147483648,
    }


class QudaMultigridCycleType(Enum):
    """
    vcycle = 0
    fcycle = 1
    wcycle = 2
    recursive = 3
    invalid = -2147483648
    """

    _prefix = "quda_mg_cycle_"
    _suffix = ""
    _values = {
        "vcycle": 0,
        "fcycle": 1,
        "wcycle": 2,
        "recursive": 3,
        "invalid": -2147483648,
    }


class QudaSchwarzType(Enum):
    """
    additive = 0
    multiplicative = 1
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_schwarz"
    _values = {"additive": 0, "multiplicative": 1, "invalid": -2147483648}


class QudaResidualType(Enum):
    """
    l2_relative = 1    # L2 relative residual (default)
    l2_absolute = 2    # L2 absolute residual
    heavy_quark = 4    # Fermilab heavy quark residual
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_residual"
    _values = {
        "l2_relative": 1,
        "l2_absolute": 2,
        "heavy_quark": 4,
        "invalid": -2147483648,
    }


class QudaCABasis(Enum):
    """
    power = 0
    chebyshev = 1
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_basis"
    _values = {"power": 0, "chebyshev": 1, "invalid": -2147483648}


class QudaMatPCType(Enum):
    """
    even_even = 0
    odd_odd = 1
    even_even_asymmetric = 2
    odd_odd_asymmetric = 3
    invalid = -2147483648
    """

    _prefix = "quda_matpc_"
    _suffix = ""
    _values = {
        "even_even": 0,
        "odd_odd": 1,
        "even_even_asymmetric": 2,
        "odd_odd_asymmetric": 3,
        "invalid": -2147483648,
    }


class QudaDagType(Enum):
    """
    no = 0
    yes = 1
    invalid = -2147483648
    """

    _prefix = "quda_dag_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaMassNormalization(Enum):
    """
    kappa = 0
    mass = 1
    asymmetric_mass = 2
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_normalization"
    _values = {"kappa": 0, "mass": 1, "asymmetric_mass": 2, "invalid": -2147483648}


class QudaSolverNormalization(Enum):
    """
    default = 0    # leave source and solution untouched
    source = 1    # normalize such that || src || = 1
    """

    _prefix = "quda_"
    _suffix = "_normalization"
    _values = {"default": 0, "source": 1}


class QudaPreserveSource(Enum):
    """
    no = 0    # use the source for the residual
    yes = 1    # keep the source intact
    invalid = -2147483648
    """

    _prefix = "quda_preserve_source_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaDiracFieldOrder(Enum):
    """
    internal = 0    # internal dirac order used, varies on precision and dslash type
     = 1    # even-odd, color inside spin
    qdp = 2    # even-odd, spin inside color
    qdpjit = 3    # even-odd, complex-color-spin-spacetime
    cps_wilson = 4    # odd-even, color inside spin
    lex = 5    # lexicographical order, color inside spin
    tifr_padded = 6    # padded z dimension for TIFR RHMC code
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_dirac_order"
    _values = {
        "internal": 0,
        "": 1,
        "qdp": 2,
        "qdpjit": 3,
        "cps_wilson": 4,
        "lex": 5,
        "tifr_padded": 6,
        "invalid": -2147483648,
    }


class QudaCloverFieldOrder(Enum):
    """
    float = 1    # even-odd float ordering
    float2 = 2    # even-odd float2 ordering
    float4 = 4    # even-odd float4 ordering
    float8 = 8    # even-odd float8 ordering
    packed = 9    # even-odd, QDP packed
    qdpjit = 10    # (diagonal / off-diagonal)-chirality-spacetime
    bqcd = 11    # even-odd, super-diagonal packed and reordered
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_clover_order"
    _values = {
        "float": 1,
        "float2": 2,
        "float4": 4,
        "float8": 8,
        "packed": 9,
        "qdpjit": 10,
        "bqcd": 11,
        "invalid": -2147483648,
    }


class QudaVerbosity(Enum):
    """
    silent = 0
    summarize = 1
    verbose = 2
    debug_verbose = 3
    invalid_verbosity = -2147483648
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {
        "silent": 0,
        "summarize": 1,
        "verbose": 2,
        "debug_verbose": 3,
        "invalid_verbosity": -2147483648,
    }


class QudaTune(Enum):
    """
    no = 0
    yes = 1
    invalid = -2147483648
    """

    _prefix = "quda_tune_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaPreserveDirac(Enum):
    """
    no = 0
    yes = 1
    invalid = -2147483648
    """

    _prefix = "quda_preserve_dirac_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaParity(Enum):
    """
    even = 0
    odd = 1
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_parity"
    _values = {"even": 0, "odd": 1, "invalid": -2147483648}


class QudaDiracType(Enum):
    """
    wilson = 0
    wilsonpc = 1
    clover = 2
    cloverpc = 3
    clover_hasenbusch_twist = 4
    clover_hasenbusch_twistpc = 5
    domain_wall = 6
    domain_wallpc = 7
    domain_wall_4d = 8
    domain_wall_4dpc = 9
    mobius_domain_wall = 10
    mobius_domain_wallpc = 11
    mobius_domain_wall_eofa = 12
    mobius_domain_wallpc_eofa = 13
    staggered = 14
    staggeredpc = 15
    staggeredkd = 16
    asqtad = 17
    asqtadpc = 18
    asqtadkd = 19
    twisted_mass = 20
    twisted_masspc = 21
    twisted_clover = 22
    twisted_cloverpc = 23
    coarse = 24
    coarsepc = 25
    gauge_laplace = 26
    gauge_laplacepc = 27
    gauge_covdev = 28
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_dirac"
    _values = {
        "wilson": 0,
        "wilsonpc": 1,
        "clover": 2,
        "cloverpc": 3,
        "clover_hasenbusch_twist": 4,
        "clover_hasenbusch_twistpc": 5,
        "domain_wall": 6,
        "domain_wallpc": 7,
        "domain_wall_4d": 8,
        "domain_wall_4dpc": 9,
        "mobius_domain_wall": 10,
        "mobius_domain_wallpc": 11,
        "mobius_domain_wall_eofa": 12,
        "mobius_domain_wallpc_eofa": 13,
        "staggered": 14,
        "staggeredpc": 15,
        "staggeredkd": 16,
        "asqtad": 17,
        "asqtadpc": 18,
        "asqtadkd": 19,
        "twisted_mass": 20,
        "twisted_masspc": 21,
        "twisted_clover": 22,
        "twisted_cloverpc": 23,
        "coarse": 24,
        "coarsepc": 25,
        "gauge_laplace": 26,
        "gauge_laplacepc": 27,
        "gauge_covdev": 28,
        "invalid": -2147483648,
    }


class QudaFieldLocation(Enum):
    """
    cpu = 1
    cuda = 2
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_field_location"
    _values = {"cpu": 1, "cuda": 2, "invalid": -2147483648}


class QudaSiteSubset(Enum):
    """
    parity = 1
    full = 2
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_site_subset"
    _values = {"parity": 1, "full": 2, "invalid": -2147483648}


class QudaSiteOrder(Enum):
    """
    lexicographic = 0    # lexicographic ordering
    even_odd = 1    # QUDA and QDP use this
    odd_even = 2    # CPS uses this
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_site_order"
    _values = {"lexicographic": 0, "even_odd": 1, "odd_even": 2, "invalid": -2147483648}


class QudaFieldOrder(Enum):
    """
    float = 1    # spin-color-complex-space
    float2 = 2    # (spin-color-complex)/2-space-(spin-color-complex)%2
    float4 = 4    # (spin-color-complex)/4-space-(spin-color-complex)%4
    float8 = 8    # (spin-color-complex)/8-space-(spin-color-complex)%8
    space_spin_color = 9    # CPS/QDP++ ordering
    space_color_spin = 10    # QLA ordering (spin inside color)
    qdpjit = 11    # QDP field ordering (complex-color-spin-spacetime)
    qop_domain_wall = 12    # QOP domain-wall ordering
    padded_space_spin_color = 13    # TIFR RHMC ordering
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_field_order"
    _values = {
        "float": 1,
        "float2": 2,
        "float4": 4,
        "float8": 8,
        "space_spin_color": 9,
        "space_color_spin": 10,
        "qdpjit": 11,
        "qop_domain_wall": 12,
        "padded_space_spin_color": 13,
        "invalid": -2147483648,
    }


class QudaFieldCreate(Enum):
    """
    null = 0    # create new field
    zero = 1    # create new field and zero it
    copy = 2    # create copy to field
    reference = 3    # create reference to field
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_field_create"
    _values = {"null": 0, "zero": 1, "copy": 2, "reference": 3, "invalid": -2147483648}


class QudaGammaBasis(Enum):
    """
    degrand_rossi = 0
    ukqcd = 1
    chiral = 2
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_gamma_basis"
    _values = {"degrand_rossi": 0, "ukqcd": 1, "chiral": 2, "invalid": -2147483648}


class QudaSourceType(Enum):
    """
    point = 0
    random = 1
    constant = 2
    sinusoidal = 3
    corner = 4
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_source"
    _values = {
        "point": 0,
        "random": 1,
        "constant": 2,
        "sinusoidal": 3,
        "corner": 4,
        "invalid": -2147483648,
    }


class QudaNoiseType(Enum):
    """
    gauss = 0
    uniform = 1
    invalid = -2147483648
    """

    _prefix = "quda_noise_"
    _suffix = ""
    _values = {"gauss": 0, "uniform": 1, "invalid": -2147483648}


class QudaProjectionType(Enum):
    """
    minres = 0
    galerkin = 1
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_projection"
    _values = {"minres": 0, "galerkin": 1, "invalid": -2147483648}


class QudaPCType(Enum):
    """
    4d_pc = 4
    5d_pc = 5
    pc_invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {"4d_pc": 4, "5d_pc": 5, "pc_invalid": -2147483648}


class QudaTwistFlavorType(Enum):
    """
    singlet = 1
    nondeg_doublet = 2
    no = 0
    invalid = -2147483648
    """

    _prefix = "quda_twist_"
    _suffix = ""
    _values = {"singlet": 1, "nondeg_doublet": 2, "no": 0, "invalid": -2147483648}


class QudaTwistDslashType(Enum):
    """
    deg_twist_inv_dslash = 0
    deg_dslash_twist_inv = 1
    deg_dslash_twist_xpay = 2
    nondeg_dslash = 3
    dslash_invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {
        "deg_twist_inv_dslash": 0,
        "deg_dslash_twist_inv": 1,
        "deg_dslash_twist_xpay": 2,
        "nondeg_dslash": 3,
        "dslash_invalid": -2147483648,
    }


class QudaTwistCloverDslashType(Enum):
    """
    deg_clover_twist_inv_dslash = 0
    deg_dslash_clover_twist_inv = 1
    deg_dslash_clover_twist_xpay = 2
    tc_dslash_invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {
        "deg_clover_twist_inv_dslash": 0,
        "deg_dslash_clover_twist_inv": 1,
        "deg_dslash_clover_twist_xpay": 2,
        "tc_dslash_invalid": -2147483648,
    }


class QudaTwistGamma5Type(Enum):
    """
    direct = 0
    inverse = 1
    invalid = -2147483648
    """

    _prefix = "quda_twist_gamma5_"
    _suffix = ""
    _values = {"direct": 0, "inverse": 1, "invalid": -2147483648}


class QudaUseInitGuess(Enum):
    """
    no = 0
    yes = 1
    invalid = -2147483648
    """

    _prefix = "quda_use_init_guess_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaDeflatedGuess(Enum):
    """
    no = 0
    yes = 1
    invalid = -2147483648
    """

    _prefix = "quda_deflated_guess_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaComputeNullVector(Enum):
    """
    no = 0
    yes = 1
    invalid = -2147483648
    """

    _prefix = "quda_compute_null_vector_"
    _suffix = ""
    _values = {"no": 0, "yes": 1, "invalid": -2147483648}


class QudaSetupType(Enum):
    """
    null_vector_setup = 0
    test_vector_setup = 1
    invalid_setup_type = -2147483648
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {
        "null_vector_setup": 0,
        "test_vector_setup": 1,
        "invalid_setup_type": -2147483648,
    }


class QudaTransferType(Enum):
    """
    aggregate = 0
    coarse_kd = 1
    optimized_kd = 2
    optimized_kd_drop_long = 3
    invalid = -2147483648
    """

    _prefix = "quda_transfer_"
    _suffix = ""
    _values = {
        "aggregate": 0,
        "coarse_kd": 1,
        "optimized_kd": 2,
        "optimized_kd_drop_long": 3,
        "invalid": -2147483648,
    }


class QudaBoolean(Enum):
    """
    false = 0
    true = 1
    invalid = -2147483648
    """

    _prefix = "quda_boolean_"
    _suffix = ""
    _values = {"false": 0, "true": 1, "invalid": -2147483648}


class QudaBLASOperation(Enum):
    """
    n = 0    # No transpose
    t = 1    # Transpose only
    c = 2    # Conjugate transpose
    invalid = -2147483648
    """

    _prefix = "quda_blas_op_"
    _suffix = ""
    _values = {"n": 0, "t": 1, "c": 2, "invalid": -2147483648}


class QudaBLASDataType(Enum):
    """
    s = 0    # Single
    d = 1    # Double
    c = 2    # Complex(single)
    z = 3    # Complex(double)
    invalid = -2147483648
    """

    _prefix = "quda_blas_datatype_"
    _suffix = ""
    _values = {"s": 0, "d": 1, "c": 2, "z": 3, "invalid": -2147483648}


class QudaBLASDataOrder(Enum):
    """
    row = 0
    col = 1
    invalid = -2147483648
    """

    _prefix = "quda_blas_dataorder_"
    _suffix = ""
    _values = {"row": 0, "col": 1, "invalid": -2147483648}


class QudaDirection(Enum):
    """
    backwards = -1
    in_place = 0
    forwards = 1
    both_dirs = 2
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {"backwards": -1, "in_place": 0, "forwards": 1, "both_dirs": 2}


class QudaLinkDirection(Enum):
    """
    backwards = 0
    forwards = 1
    bidirectional = 2
    """

    _prefix = "quda_link_"
    _suffix = ""
    _values = {"backwards": 0, "forwards": 1, "bidirectional": 2}


class QudaFieldGeometry(Enum):
    """
    scalar = 1
    vector = 4
    tensor = 6
    coarse = 8
    kdinverse = 16    # Decomposition of Kahler-Dirac block
    invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = "_geometry"
    _values = {
        "scalar": 1,
        "vector": 4,
        "tensor": 6,
        "coarse": 8,
        "kdinverse": 16,
        "invalid": -2147483648,
    }


class QudaGhostExchange(Enum):
    """
    no = 0
    pad = 1
    extended = 2
    invalid = -2147483648
    """

    _prefix = "quda_ghost_exchange_"
    _suffix = ""
    _values = {"no": 0, "pad": 1, "extended": 2, "invalid": -2147483648}


class QudaStaggeredPhase(Enum):
    """
    no = 0
    milc = 1
    cps = 2
    tifr = 3
    invalid = -2147483648
    """

    _prefix = "quda_staggered_phase_"
    _suffix = ""
    _values = {"no": 0, "milc": 1, "cps": 2, "tifr": 3, "invalid": -2147483648}


class QudaContractType(Enum):
    """
    open = 0    # Open spin elementals
    dr = 1    # DegrandRossi
    invalid = -2147483648
    """

    _prefix = "quda_contract_type_"
    _suffix = ""
    _values = {"open": 0, "dr": 1, "invalid": -2147483648}


class QudaContractGamma(Enum):
    """
    i = 0
    g1 = 1
    g2 = 2
    g3 = 3
    g4 = 4
    g5 = 5
    g1g5 = 6
    g2g5 = 7
    g3g5 = 8
    g4g5 = 9
    s12 = 10
    s13 = 11
    s14 = 12
    s21 = 13
    s23 = 14
    s34 = 15
    invalid = -2147483648
    """

    _prefix = "quda_contract_gamma_"
    _suffix = ""
    _values = {
        "i": 0,
        "g1": 1,
        "g2": 2,
        "g3": 3,
        "g4": 4,
        "g5": 5,
        "g1g5": 6,
        "g2g5": 7,
        "g3g5": 8,
        "g4g5": 9,
        "s12": 10,
        "s13": 11,
        "s14": 12,
        "s21": 13,
        "s23": 14,
        "s34": 15,
        "invalid": -2147483648,
    }


class QudaWFlowType(Enum):
    """
    wilson = 0
    symanzik = 1
    invalid = -2147483648
    """

    _prefix = "quda_wflow_type_"
    _suffix = ""
    _values = {"wilson": 0, "symanzik": 1, "invalid": -2147483648}


class QudaExtLibType(Enum):
    """
    cusolve_extlib = 0
    eigen_extlib = 1
    magma_extlib = 2
    extlib_invalid = -2147483648
    """

    _prefix = "quda_"
    _suffix = ""
    _values = {
        "cusolve_extlib": 0,
        "eigen_extlib": 1,
        "magma_extlib": 2,
        "extlib_invalid": -2147483648,
    }
