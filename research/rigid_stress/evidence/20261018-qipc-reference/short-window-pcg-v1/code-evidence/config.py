"""Config — mutable scene configuration backed by AttributeCollection."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from qipc.base import Base
from qipc.geometry.attribute_collection import AttributeCollection
from qipc.geometry.attribute_desc import AttributeDesc

_DEPRECATED_ALIASES: dict[str, str] = {
    "velocity_tol": "newton/velocity_tol",
    "max_newton_iter": "newton/max_iter",
    "max_pcg_iter": "linear_system/max_iter",
    "max_ls_iter": "line_search/max_iter",
    "contact_enable": "contact/enable",
    "contact_enabled": "contact/enable",
    "d_hat": "contact/d_hat",
    "init_collision_pair_capacity": "contact/init_collision_pair_capacity",
    "pre_assemble_bvh_opt1": "extras/pre_assemble_bvh_opt1",
    "capacity_grow_factor": "extras/capacity_grow_factor",
    "capacity_shrink_threshold": "extras/capacity_shrink_threshold",
    "detail/init_contact_sort_capacity": "extras/init_contact_sort_capacity",
}

# -----------------------------------------------------------------------
# Config-specific descriptors (module-level constants, shared by ref)
# -----------------------------------------------------------------------

_DT = AttributeDesc(doc="simulation time step (seconds)", gt=0.0, physical=True)
_GRAVITY = AttributeDesc(doc="gravitational acceleration vector (m/s^2)", physical=True)

_VELOCITY_TOL = AttributeDesc(doc="Newton velocity convergence tolerance", gt=0.0, physical=True)
_CONTACT_TOL_SCALE = AttributeDesc(
    doc="Newton tolerance multiplier for vertices with contact gradient "
    "contributions (split convergence check); 1.0 disables the split and "
    "keeps the single-tolerance criterion",
    ge=1.0,
    physical=True,
)
_MAX_ITER = AttributeDesc(doc="maximum iteration count", ge=1, physical=True)

_TOL_RATE = AttributeDesc(doc="PCG relative tolerance (residual / rhs)", gt=0.0, physical=True)
_NEWTON_AWARE_PCG_ENABLE = AttributeDesc(
    doc="default-enabled Lagrangian PCG-entry gate using conservative inertia "
    "baselines and world-motion amplification; unsupported layouts auto-disable",
    enum={0: "disabled", 1: "enabled (default)"},
    physical=True,
)
_NEWTON_AWARE_PCG_SAFETY_MARGIN = AttributeDesc(
    doc="fraction of Newton's absolute displacement tolerance available to "
    "the conservative PCG-entry direction bound",
    gt=0.0,
    le=1.0,
    physical=True,
)
_PRECONDITIONER = AttributeDesc(
    doc="FEM-side preconditioner",
    enum={"mas": "multilevel additive Schwarz", "diag": "12x12 block-Jacobi fallback"},
    physical=True,
)
_INVERT_PRECISION = AttributeDesc(
    doc="working precision of the preconditioner factorization",
    enum={
        "float": "fp32 (~1e-7 accuracy, fast on consumer GPUs)",
        "double": "fp64 (full precision)",
    },
    physical=True,
)
_MAS_RETOPO_REPAIR_ENABLE = AttributeDesc(
    doc="enable deterministic GPU-local MAS slot repair after topology events",
    enum={0: "disabled", 1: "enabled"},
    physical=True,
)
_MAS_RETOPO_REPAIR_MAX_ITER = AttributeDesc(
    doc="maximum existing-bank proposal rounds per MAS topology repair",
    ge=1,
    physical=True,
)
_MAS_RETOPO_RESERVE_BANKS = AttributeDesc(
    doc="fully empty MAS banks reserved at init for future topology births",
    ge=0,
    physical=True,
)
_MAS_RETOPO_SV_MAX_ITER = AttributeDesc(
    doc="maximum FastSV iterations for unresolved MAS repair components",
    ge=1,
    physical=True,
)
_ABD_PRECONDITIONER = AttributeDesc(
    doc="ABD-side preconditioner",
    enum={
        "tree": "exact kinematic-tree solve (subsumes diag at zero joint edges)",
        "diag": "12x12 block-Jacobi fallback/baseline",
    },
    physical=True,
)
_RIGID_PRECONDITIONER = AttributeDesc(
    doc="pure-rigid-side preconditioner",
    enum={
        "auto": "tree for rigid joint graphs, otherwise 6x6 block-Jacobi",
        "tree": "exact 6x6 kinematic-tree solve",
        "diag": "6x6 block-Jacobi fallback/baseline",
    },
    physical=True,
)
_LOOP_MODE = AttributeDesc(
    doc="loop-closure policy for the tree preconditioner",
    enum={
        "prune": "drop loop edges from M_tree (warns)",
        "error": "reject cyclic joint graphs",
    },
    physical=True,
)
_SOLVER = AttributeDesc(
    doc="linear system solver",
    enum={
        "partition_pcg": "partitioned PCG with component partitioner",
        "linear_pcg": "standard (unpartitioned) PCG",
    },
    physical=True,
)
_VALIDATE_BCOO = AttributeDesc(
    doc="BCOO sorted-order checkpoint after sort-reduce; 0 to skip",
    enum={0: "disabled", 1: "enabled"},
)
_COARSENING_ENABLE = AttributeDesc(
    doc="AGIPC in-solve algebraic coarsening: per Newton iteration the linear "
    "solve runs on a Galerkin-reduced system aggregated from kinematically "
    "coherent FEM vertices, then refines on the fine system "
    "(docs/adaptive_coarsening.md). Forces linear_system/solver=linear_pcg",
    enum={0: "disabled (default)", 1: "enabled"},
    physical=True,
)
_COARSENING_STRAIN_INCREMENT_THRESHOLD = AttributeDesc(
    doc="Green strain increment protect threshold (AGIPC paper sec 4.2, "
    "eq. 3): elements with ||dG||_F above it protect their edges from "
    "aggregation. Dimensionless and deliberately material-independent; "
    "the paper's default 5e-5 balances fidelity and coarsening; loosen "
    "(~10x) for extreme stiffness where solver noise approaches real "
    "strain increments (paper Fig. 10 uses 5e-4). Runtime-mutable via "
    "solver.coarsening_strain_threshold",
    gt=0.0,
    physical=True,
)
_COARSENING_AFFINE_EMBEDDING_THRESHOLD = AttributeDesc(
    doc="adaptive affine embedding (AGIPC paper sec 4.2): supernodes "
    "aggregating more fine vertices than this get 12 affine DOFs "
    "(translation+rotation+scale+shear); smaller ones get 3 translational "
    "DOFs. Paper default 32 (empirical)",
    ge=1,
    physical=True,
)
_COARSENING_POST_COARSENING_MAX_ITER = AttributeDesc(
    doc="post-coarsening refinement (AGIPC paper sec 4.3): warm-started "
    "fine-system CG iteration budget after prolongating the coarse "
    "solution (high-frequency recovery, counters subspace locking). Paper "
    "adopts <=10 and pairs it with a diagonal fine preconditioner "
    "(supplement sec 4)",
    ge=0,
    physical=True,
)
_COARSENING_MAX_ROW_RATIO = AttributeDesc(
    doc="per-Newton bail-out cutoff: when aggregation leaves more than this "
    "fraction of the fine block rows in the coarse system, the iteration "
    "skips the coarse stage entirely and the fine CG runs with its full "
    "budget (a barely-aggregated 'coarse' system costs a full extra solve "
    "for no size reduction; transient frames where the criterion protects "
    "everything hit this constantly)",
    gt=0.0,
    le=1.0,
    physical=True,
)
_COARSENING_NEWTON_STALL_LIMIT = AttributeDesc(
    doc="anti-locking escalation: once a frame exceeds this many Newton "
    "iterations, the remaining iterations bail out to full fine solves "
    "(near convergence the strain increments over-coarsen and the "
    "post-CG budget cannot express the residual correction -- subspace "
    "locking). Set well above the scene's typical per-frame Newton count",
    ge=1,
    physical=True,
)
_COARSENING_COARSE_PRECOND = AttributeDesc(
    doc="coarse-solve preconditioner (docs/adaptive_coarsening.md): the AGIPC "
    "paper runs the coarse PCG under its MAS preconditioner (supplement "
    "sec 4); 'schwarz' is that analog here -- a multilevel additive "
    "Schwarz whose level l aggregates 16^l consecutive coarse block rows "
    "per node, 16 nodes per 48x48 dense bank (supernode ids are "
    "mesh-ordered, so the hierarchy is pure index division). 'jacobi' "
    "keeps the 3x3 block-diagonal fallback for A/B comparisons",
    enum={
        "schwarz": "multilevel additive Schwarz (default, paper analog)",
        "jacobi": "3x3 block-Jacobi (ablation fallback)",
    },
    physical=True,
)
_LINE_SEARCH_MAX_ITER = AttributeDesc(
    doc="maximum line-search iterations per Newton step", ge=1, physical=True
)

_CONTACT_ENABLE = AttributeDesc(
    doc="enable contact detection and barrier forces",
    enum={0: "disabled", 1: "enabled"},
    physical=True,
)
_D_HAT = AttributeDesc(doc="contact barrier distance threshold (m)", gt=0.0, physical=True)
_MPM_INIT_BLOCK_POOL_CAPACITY = AttributeDesc(
    doc="initial MPM block-pool capacity (4^3-cell blocks); overflow yields "
    "and grows the pool, so this only trades start-up memory against a few "
    "reallocations early in a run",
    ge=1,
)
_MPM_ASSEMBLE_FP32 = AttributeDesc(
    doc="compute the MPM elastic HESSIAN blocks in fp32 (the gradient, energy "
    "and solve stay fp64). fp32 rounding (~1e-7 relative) is far inside the "
    "PCG's 1e-4 inexact-solve tolerance, so Newton quality is unaffected; on "
    "GeForce-class GPUs (1/64 fp64 rate) this removes the dominant assembly "
    "cost. 0 = fp64 (default), 1 = fp32",
    enum={0: "fp64", 1: "fp32"},
    physical=True,
)
_FEM_ASSEMBLE_FP32 = AttributeDesc(
    doc="compute the FEM tet elastic HESSIAN in fp32 (the gradient, energy "
    "and solve stay fp64); same inexact-Newton budget argument as "
    "mpm/assemble_fp32. Consumed today by StableNeoHookean3D (the 9x9 SPD "
    "projection dominates its per-tet cost). 0 = fp64 (default), 1 = fp32",
    enum={0: "fp64", 1: "fp32"},
    physical=True,
)
_MPM_LITE_QUADRATURE = AttributeDesc(
    doc="integrate the MPM elastic term at ONE cell-center quadrature point "
    "per active cell instead of every particle (MPM Lite, Feng et al. 2026): "
    "per-Newton-iteration elastic cost scales with cells, not ppc. The stretch "
    "reference is reconstructed once per frame from the cell's V-weighted "
    "Kirchhoff stress (closed form for Hencky). Kinematic transfers, contact "
    "and plasticity are unchanged; a DIFFERENT (one-point hex) discretization "
    "of the same potential, so trajectories differ from the per-particle path. "
    "Single-material scenes only in v0. 0 = per-particle (default), 1 = cell-center",
    enum={0: "per-particle", 1: "cell-center"},
    physical=True,
)
_MAX_STEP_IN_D_HAT = AttributeDesc(
    doc="cap CCD step size at this fraction of d_hat; -1 = disabled", physical=True
)
_CCD_BOUND = AttributeDesc(
    doc="conservative-advancement bound for CCD sweeps",
    enum={
        "additive": "isotropic displacement bound; the ACCD reference routine",
        "directional": "closing-rate bound; same time of contact in fewer advance steps",
    },
    physical=True,
)
_ADAPTIVE_KAPPA_MODE = AttributeDesc(
    doc="granularity of the in-graph adaptive contact-kappa controller the "
    "scene starts with (default per-body: every contact scene runs the "
    "controller unless this key opts out with 'off'); tuning parameters via "
    "Solver.enable_adaptive_kappa",
    enum={
        "off": "authored ContactTabular stiffness only (opt-out)",
        "global": "one scene-wide multiplier keyed on the smallest in-band gap",
        "per-vertex": "per-primitive multipliers keyed on per-vertex gap folds",
        "per-body": "per contact body, patch-uniform max(scale[pair bodies]); the default",
    },
    physical=True,
)
_ADAPTIVE_KAPPA_TICK = AttributeDesc(
    doc="when the adaptive-kappa GROW path fires (consumed when the mode is "
    "not off; default newton); tuning parameters via "
    "Solver.enable_adaptive_kappa",
    enum={
        "frame": "one grow per triggered frame, at the frame boundary",
        "newton": "grow at Newton-iteration boundaries (global/per-body modes only); the default",
    },
    physical=True,
)
_INIT_CAPACITY = AttributeDesc(doc="initial capacity (element count)", ge=1)
_CCD_PARTITION = AttributeDesc(
    doc="enable CCD component partitioning",
    enum={0: "disabled", 1: "enabled"},
    physical=True,
)
_CCD_PARTITION_SV_MAX_ITER = AttributeDesc(
    doc="max Schwarz-Varga iterations for CCD partitioner", ge=1, physical=True
)
_INTERSECTION_CHECK = AttributeDesc(
    doc="run intersection check at init",
    enum={0: "disabled", 1: "enabled"},
    physical=True,
)
_INTERSECTION_CHECK_CAPACITY = AttributeDesc(
    doc="initial ET-pair buffer capacity (grows in-pipeline on overflow)", ge=1
)
_CONTACT_CONSTITUTION = AttributeDesc(
    doc="contact constitution backend",
    enum={
        "auto": "derive from contact tabular",
        "consistent_ipc": "Consistent IPC",
        "gipc": "GIPC",
        "adhesive_ipc": "Adhesive IPC",
        "variational_adhesive_ipc": "Variational adhesive IPC",
    },
    physical=True,
)

_FRICTION_EPS_V = AttributeDesc(doc="friction smoothing velocity threshold", gt=0.0, physical=True)

_GROW_FACTOR = AttributeDesc(doc="geometric grow factor for pool reallocation", ge=1.0)

_CAPACITY_SHRINK = AttributeDesc(
    doc="shrink threshold as fraction of current capacity", gt=0.0, le=1.0
)
_PRE_ASSEMBLE_BVH_OPT1 = AttributeDesc(
    doc="pre-assemble BVH optimization pass 1",
    enum={0: "disabled", 1: "enabled"},
    physical=True,
)

_PCG_DIAG_ENABLE = AttributeDesc(
    doc="enable PCG/Newton diagnostic dump", enum={0: "disabled", 1: "enabled"}
)
_PCG_DIAG_CAPACITY = AttributeDesc(doc="max r/z snapshots per step", ge=1)
_PCG_DIAG_RECORD = AttributeDesc(
    doc="comma-separated vectors to record",
    enum={"r": "residual", "z": "preconditioned residual"},
    multi=True,
)
_PCG_DIAG_DUMP_SYSTEM = AttributeDesc(
    doc="snapshot assembled BCOO + rhs once per Newton iteration",
    enum={0: "disabled", 1: "enabled"},
)
_PCG_DIAG_SYS_CAPACITY = AttributeDesc(doc="max Newton iterations recorded per step", ge=1)
_PCG_DIAG_NNZ_CAPACITY = AttributeDesc(
    doc="BCOO entries per system slot; 0 = use linear-system capacity at init",
    ge=0,
)

_LS_FORENSICS_ENABLE = AttributeDesc(
    doc="dump the energy closure when a line search exhausts (docs/ls_forensics.md)",
    enum={0: "disabled", 1: "enabled"},
)
_LS_FORENSICS_MAX_DUMPS = AttributeDesc(doc="max exhaustion dumps per run", ge=1)
_LS_FORENSICS_HALT_AFTER_DUMP = AttributeDesc(
    doc="diagnostic capture mode: finish the exhausted iteration, then exit "
    "the current Newton loop after staging its closure",
    enum={0: "disabled", 1: "enabled"},
)
_LS_FORENSICS_TEST_BIAS = AttributeDesc(
    doc="TEST ONLY: bias added to every LS trial energy; forces one "
    "deterministic exhaustion per frame (0 = off)",
    physical=True,
)

_CONTACT_CLOSURE_ENABLE = AttributeDesc(
    doc="dump per-Newton contact closure for offline replay (docs/contact_closure.md)",
    enum={0: "disabled", 1: "enabled"},
)
_CONTACT_CLOSURE_FRAME_BEGIN = AttributeDesc(doc="first frame to dump (inclusive)", ge=0)
_CONTACT_CLOSURE_FRAME_END = AttributeDesc(
    doc="last frame to dump (inclusive); -1 = no limit", ge=-1
)
_CONTACT_CLOSURE_MAX_DUMPS = AttributeDesc(doc="max closure snapshots across the entire run", ge=1)

_BVH_TYPE = AttributeDesc(
    doc="broadphase BVH implementation",
    enum={
        "lbvh": "basic LBVH",
        "info_lbvh": "info-guided LBVH",
        "info_lbvh_batched": "batched info-guided LBVH, fp64 AABB bounds",
        "info_lbvh_batched_f32": "batched info-guided LBVH, fp32 conservative bounds",
        "info_lbvh_batched_dop14": "batched info-guided LBVH, fp32 14-DOP bounds (default)",
        "info_stackless": "info-guided stackless BVH",
    },
    physical=True,
)
_BVH_EE_QUERY = AttributeDesc(
    doc="edge-edge BVH traversal schedule",
    enum={
        "dual": "dual-tree node-pair traversal (default)",
        "warp": "production warp-per-edge rollback",
    },
)
_BVH_DUAL_FRONTIER_LEVELS = AttributeDesc(
    doc="dual EE frontier levels; 0 selects device-side auto depth",
    ge=0,
    le=18,
)
_BVH_DUAL_TARGET_WAVES = AttributeDesc(
    doc="auto dual EE target frontier in resident DFS warp waves",
    ge=1.0,
)
_BVH_DUAL_MAX_LEVELS = AttributeDesc(
    doc="hard maximum dual EE frontier expansion levels",
    ge=1,
    le=18,
)
_PARTITION_SV_MAX_ITER = AttributeDesc(
    doc="max Schwarz-Varga iterations for linear-system partitioner", ge=1, physical=True
)


def _resolve_key(key: str) -> str:
    """Map a deprecated key to its new name, emitting a warning."""
    new = _DEPRECATED_ALIASES.get(key)
    if new is not None:
        warnings.warn(
            f"Config key '{key}' is deprecated, use '{new}' instead",
            FutureWarning,
            stacklevel=4,
        )
        return new
    return key


class Config(Base):
    # `_solver` is the live retarget path for the two keys the running solver
    # keeps device copies of (dt, gravity). Without it their setters wrote the
    # host record only, so a post-init assignment looked like it worked and
    # changed nothing -- which is what drove callers to reach into
    # `affine_body.gravity` instead (docs/api.md, "Runtime retargeting").
    __slots__ = ("_ac", "_solver")

    def __init__(self, **kwargs: Any) -> None:
        self._solver = None
        self._ac = AttributeCollection(size=1)
        self._ac.create("dt", np.float64, default=0.01, allow_destroy=False, desc=_DT)
        self._ac.create(
            "gravity",
            (np.float64, 3),
            default=np.array([0.0, -9.8, 0.0]),
            allow_destroy=False,
            desc=_GRAVITY,
        )

        self._ac.create(
            "newton/velocity_tol",
            np.float64,
            default=5e-2,
            allow_destroy=False,
            desc=_VELOCITY_TOL,
        )
        self._ac.create(
            "newton/contact_tol_scale",
            np.float64,
            default=1.0,
            allow_destroy=False,
            desc=_CONTACT_TOL_SCALE,
        )
        self._ac.create(
            "newton/max_iter",
            np.int64,
            default=1024,
            allow_destroy=False,
            desc=_MAX_ITER,
        )

        self._ac.create(
            "linear_system/max_iter",
            np.int64,
            default=1024,
            allow_destroy=False,
            desc=_MAX_ITER,
        )
        self._ac.create(
            "linear_system/tol_rate",
            np.float64,
            default=1e-4,
            allow_destroy=False,
            desc=_TOL_RATE,
        )
        self._ac.create(
            "linear_system/newton_aware_pcg/enable",
            np.int32,
            default=1,
            allow_destroy=False,
            desc=_NEWTON_AWARE_PCG_ENABLE,
        )
        self._ac.create(
            "linear_system/newton_aware_pcg/safety_margin",
            np.float64,
            default=0.5,
            allow_destroy=False,
            desc=_NEWTON_AWARE_PCG_SAFETY_MARGIN,
        )
        self._ac.create(
            "linear_system/preconditioner",
            str,
            default="mas",
            allow_destroy=False,
            desc=_PRECONDITIONER,
        )
        self._ac.create(
            "linear_system/preconditioner/mas/invert_precision",
            str,
            default="float",
            allow_destroy=False,
            desc=_INVERT_PRECISION,
        )
        self._ac.create(
            "linear_system/preconditioner/mas/retopo_repair_enable",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_MAS_RETOPO_REPAIR_ENABLE,
        )
        self._ac.create(
            "linear_system/preconditioner/mas/retopo_repair_max_iter",
            np.int64,
            default=8,
            allow_destroy=False,
            desc=_MAS_RETOPO_REPAIR_MAX_ITER,
        )
        self._ac.create(
            "linear_system/preconditioner/mas/retopo_reserve_banks",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_MAS_RETOPO_RESERVE_BANKS,
        )
        self._ac.create(
            "linear_system/preconditioner/mas/retopo_sv_max_iter",
            np.int64,
            default=64,
            allow_destroy=False,
            desc=_MAS_RETOPO_SV_MAX_ITER,
        )
        self._ac.create(
            "linear_system/abd_preconditioner",
            str,
            default="tree",
            allow_destroy=False,
            desc=_ABD_PRECONDITIONER,
        )
        self._ac.create(
            "linear_system/abd_preconditioner/tree/loop_mode",
            str,
            default="prune",
            allow_destroy=False,
            desc=_LOOP_MODE,
        )
        self._ac.create(
            "linear_system/abd_preconditioner/tree/invert_precision",
            str,
            default="float",
            allow_destroy=False,
            desc=_INVERT_PRECISION,
        )
        self._ac.create(
            "linear_system/rigid_preconditioner",
            str,
            default="auto",
            allow_destroy=False,
            desc=_RIGID_PRECONDITIONER,
        )
        self._ac.create(
            "linear_system/rigid_preconditioner/tree/loop_mode",
            str,
            default="prune",
            allow_destroy=False,
            desc=_LOOP_MODE,
        )
        self._ac.create(
            "linear_system/rigid_preconditioner/tree/invert_precision",
            str,
            default="double",
            allow_destroy=False,
            desc=_INVERT_PRECISION,
        )
        self._ac.create(
            "linear_system/solver",
            str,
            default="partition_pcg",
            allow_destroy=False,
            desc=_SOLVER,
        )
        self._ac.create(
            "linear_system/validate_bcoo",
            np.int64,
            default=1,
            allow_destroy=False,
            desc=_VALIDATE_BCOO,
        )
        self._ac.create(
            "linear_system/partition_sv_max_iter",
            np.int64,
            default=64,
            allow_destroy=False,
            desc=_PARTITION_SV_MAX_ITER,
        )
        self._ac.create(
            "linear_system/adaptive_coarsening/enable",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_COARSENING_ENABLE,
        )
        self._ac.create(
            "linear_system/adaptive_coarsening/strain_increment_threshold",
            np.float64,
            default=5e-5,
            allow_destroy=False,
            desc=_COARSENING_STRAIN_INCREMENT_THRESHOLD,
        )
        self._ac.create(
            "linear_system/adaptive_coarsening/affine_embedding_threshold",
            np.int64,
            default=32,
            allow_destroy=False,
            desc=_COARSENING_AFFINE_EMBEDDING_THRESHOLD,
        )
        self._ac.create(
            "linear_system/adaptive_coarsening/post_coarsening_max_iter",
            np.int64,
            default=10,
            allow_destroy=False,
            desc=_COARSENING_POST_COARSENING_MAX_ITER,
        )
        self._ac.create(
            "linear_system/adaptive_coarsening/max_row_ratio",
            np.float64,
            default=0.5,
            allow_destroy=False,
            desc=_COARSENING_MAX_ROW_RATIO,
        )
        self._ac.create(
            "linear_system/adaptive_coarsening/newton_stall_limit",
            np.int64,
            default=128,
            allow_destroy=False,
            desc=_COARSENING_NEWTON_STALL_LIMIT,
        )
        self._ac.create(
            "linear_system/adaptive_coarsening/coarse_precond",
            str,
            default="schwarz",
            allow_destroy=False,
            desc=_COARSENING_COARSE_PRECOND,
        )

        self._ac.create(
            "line_search/max_iter",
            np.int64,
            default=12,
            allow_destroy=False,
            desc=_LINE_SEARCH_MAX_ITER,
        )

        self._ac.create(
            "contact/enable",
            np.int64,
            default=1,
            allow_destroy=False,
            desc=_CONTACT_ENABLE,
        )
        self._ac.create(
            "contact/d_hat",
            np.float64,
            default=0.01,
            allow_destroy=False,
            desc=_D_HAT,
        )

        # --- MPM subsystem (docs/mpm_coupling_design.md; solver internals
        #     only -- nothing selects a coupling or a grid type) ---
        self._ac.create(
            "mpm/init_block_pool_capacity",
            np.int64,
            default=8192,
            allow_destroy=False,
            desc=_MPM_INIT_BLOCK_POOL_CAPACITY,
        )
        self._ac.create(
            "mpm/assemble_fp32",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_MPM_ASSEMBLE_FP32,
        )
        self._ac.create(
            "fem/assemble_fp32",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_FEM_ASSEMBLE_FP32,
        )
        self._ac.create(
            "mpm/lite_quadrature",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_MPM_LITE_QUADRATURE,
        )
        self._ac.create(
            "contact/max_step_in_d_hat",
            np.float64,
            default=-1.0,
            allow_destroy=False,
            desc=_MAX_STEP_IN_D_HAT,
        )
        self._ac.create(
            "contact/ccd_bound",
            str,
            default="directional",
            allow_destroy=False,
            desc=_CCD_BOUND,
        )
        self._ac.create(
            "contact/adaptive_kappa_mode",
            str,
            default="per-body",
            allow_destroy=False,
            desc=_ADAPTIVE_KAPPA_MODE,
        )
        self._ac.create(
            "contact/adaptive_kappa_tick",
            str,
            default="newton",
            allow_destroy=False,
            desc=_ADAPTIVE_KAPPA_TICK,
        )
        self._ac.create(
            "contact/init_collision_pair_capacity",
            np.int64,
            default=1_000,
            allow_destroy=False,
            desc=_INIT_CAPACITY,
        )
        self._ac.create(
            "contact/ccd_partition",
            np.int64,
            default=1,
            allow_destroy=False,
            desc=_CCD_PARTITION,
        )
        self._ac.create(
            "contact/ccd_partition_sv_max_iter",
            np.int64,
            default=64,
            allow_destroy=False,
            desc=_CCD_PARTITION_SV_MAX_ITER,
        )
        self._ac.create(
            "contact/intersection_check",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_INTERSECTION_CHECK,
        )
        self._ac.create(
            "contact/intersection_check_capacity",
            np.int64,
            default=1024,
            allow_destroy=False,
            desc=_INTERSECTION_CHECK_CAPACITY,
        )
        self._ac.create(
            "contact/constitution",
            str,
            default="auto",
            allow_destroy=False,
            desc=_CONTACT_CONSTITUTION,
        )

        self._ac.create(
            "friction/eps_v",
            np.float64,
            default=1e-2,
            allow_destroy=False,
            desc=_FRICTION_EPS_V,
        )

        self._ac.create(
            "topo/grow_factor",
            np.float64,
            default=1.5,
            allow_destroy=False,
            desc=_GROW_FACTOR,
        )

        self._ac.create(
            "extras/init_contact_sort_capacity",
            np.int64,
            default=1_000_000,
            allow_destroy=False,
            desc=_INIT_CAPACITY,
        )
        self._ac.create(
            "extras/pre_assemble_bvh_opt1",
            np.int64,
            default=1,
            allow_destroy=False,
            desc=_PRE_ASSEMBLE_BVH_OPT1,
        )
        self._ac.create(
            "extras/capacity_grow_factor",
            np.float64,
            default=1.2,
            allow_destroy=False,
            desc=_GROW_FACTOR,
        )
        self._ac.create(
            "extras/capacity_shrink_threshold",
            np.float64,
            default=0.8,
            allow_destroy=False,
            desc=_CAPACITY_SHRINK,
        )

        self._ac.create(
            "extras/pcg_diagnostics/enable",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_PCG_DIAG_ENABLE,
        )
        self._ac.create(
            "extras/pcg_diagnostics/capacity",
            np.int64,
            default=2048,
            allow_destroy=False,
            desc=_PCG_DIAG_CAPACITY,
        )
        self._ac.create(
            "extras/pcg_diagnostics/record",
            str,
            default="r,z",
            allow_destroy=False,
            desc=_PCG_DIAG_RECORD,
        )
        self._ac.create(
            "extras/pcg_diagnostics/dump_system",
            np.int64,
            default=1,
            allow_destroy=False,
            desc=_PCG_DIAG_DUMP_SYSTEM,
        )
        self._ac.create(
            "extras/pcg_diagnostics/system_capacity",
            np.int64,
            default=16,
            allow_destroy=False,
            desc=_PCG_DIAG_SYS_CAPACITY,
        )
        self._ac.create(
            "extras/pcg_diagnostics/nnz_capacity",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_PCG_DIAG_NNZ_CAPACITY,
        )

        self._ac.create(
            "extras/ls_forensics/enable",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_LS_FORENSICS_ENABLE,
        )
        self._ac.create(
            "extras/ls_forensics/max_dumps",
            np.int64,
            default=4,
            allow_destroy=False,
            desc=_LS_FORENSICS_MAX_DUMPS,
        )
        self._ac.create(
            "extras/ls_forensics/halt_after_dump",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_LS_FORENSICS_HALT_AFTER_DUMP,
        )
        self._ac.create(
            "extras/ls_forensics/test_energy_bias",
            np.float64,
            default=0.0,
            allow_destroy=False,
            desc=_LS_FORENSICS_TEST_BIAS,
        )

        self._ac.create(
            "extras/contact_closure/enable",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_CONTACT_CLOSURE_ENABLE,
        )
        self._ac.create(
            "extras/contact_closure/frame_begin",
            np.int64,
            default=0,
            allow_destroy=False,
            desc=_CONTACT_CLOSURE_FRAME_BEGIN,
        )
        self._ac.create(
            "extras/contact_closure/frame_end",
            np.int64,
            default=-1,
            allow_destroy=False,
            desc=_CONTACT_CLOSURE_FRAME_END,
        )
        self._ac.create(
            "extras/contact_closure/max_dumps",
            np.int64,
            default=200,
            allow_destroy=False,
            desc=_CONTACT_CLOSURE_MAX_DUMPS,
        )

        self._ac.create(
            "bvh/type",
            str,
            # fp32 14-DOP default: halves the broadphase candidate stream
            # (-46..-61% measured) feeding filter/sort/CCD, matches or beats
            # fp64-AABB whole-frame time on every measured scene including
            # the uniform-displacement worst case (docs/bvh_lab_report.md).
            default="info_lbvh_batched_dop14",
            allow_destroy=False,
            desc=_BVH_TYPE,
        )
        self._ac.create(
            "bvh/ee_query",
            str,
            default="dual",
            allow_destroy=False,
            desc=_BVH_EE_QUERY,
        )
        self._ac.create(
            "bvh/dual/frontier_levels",
            np.int32,
            default=0,
            allow_destroy=False,
            desc=_BVH_DUAL_FRONTIER_LEVELS,
        )
        self._ac.create(
            "bvh/dual/target_waves",
            np.float64,
            default=24.0,
            allow_destroy=False,
            desc=_BVH_DUAL_TARGET_WAVES,
        )
        self._ac.create(
            "bvh/dual/max_levels",
            np.int32,
            default=18,
            allow_destroy=False,
            desc=_BVH_DUAL_MAX_LEVELS,
        )

        for key, value in kwargs.items():
            key = _resolve_key(key)
            if key in self._ac:
                attr = self._ac[key]
                if attr.dtype == np.dtype("O"):
                    attr.cpu()[0] = value
                else:
                    self._ac[key] = self._to_row(value, attr.dtype)
            elif isinstance(value, str):
                self._ac.create(key, str, default=value)
            else:
                arr = np.atleast_1d(np.asarray(value))
                if arr.ndim == 1:
                    dtype = (arr.dtype, arr.shape[0]) if arr.shape[0] > 1 else arr.dtype
                else:
                    dtype = (arr.dtype, arr.shape)
                self._ac.create(key, dtype)
                if isinstance(dtype, tuple):
                    self._ac[key] = arr.reshape(1, *arr.shape)
                else:
                    self._ac[key] = arr.reshape(1)

    def _repr_brief(self) -> str:
        g = self.gravity
        return f"<Config: dt={self.dt}, gravity={g}>"

    @staticmethod
    def _to_row(value: Any, dtype: np.dtype) -> np.ndarray:
        arr = np.asarray(value, dtype=dtype)
        return arr.reshape(1, *arr.shape) if arr.ndim > 0 else arr.reshape(1)

    @property
    def dt(self) -> float:
        return float(self._ac["dt"].cpu()[0])

    @dt.setter
    def dt(self, value: float) -> None:
        self._ac["dt"].cpu()[0] = value
        if self._solver is not None:
            self._solver.set_dt(float(value))

    @property
    def gravity(self) -> tuple[float, float, float]:
        g = self._ac["gravity"].cpu()[0]
        return (float(g[0]), float(g[1]), float(g[2]))

    @gravity.setter
    def gravity(self, value: tuple[float, float, float]) -> None:
        if self._solver is not None:
            raise RuntimeError(
                "Changing gravity at runtime is not supported. Gravity is now a "
                "per-DOF input: use scene.affine_body.gravity (B,3) for ABD bodies "
                "or scene.finite_element.gravity (V,3) for FEM vertices."
            )
        self._ac["gravity"].cpu()[0] = np.asarray(value, dtype=np.float64)

    @property
    def newton_velocity_tol(self) -> float:
        return float(self._ac["newton/velocity_tol"].cpu()[0])

    @property
    def newton_contact_tol_scale(self) -> float:
        return float(self._ac["newton/contact_tol_scale"].cpu()[0])

    @property
    def newton_max_iter(self) -> int:
        return int(self._ac["newton/max_iter"].cpu()[0])

    @property
    def linear_system_max_iter(self) -> int:
        return int(self._ac["linear_system/max_iter"].cpu()[0])

    @property
    def linear_system_tol_rate(self) -> float:
        return float(self._ac["linear_system/tol_rate"].cpu()[0])

    @property
    def coarsening_enable(self) -> bool:
        return bool(self._ac["linear_system/adaptive_coarsening/enable"].cpu()[0])

    @property
    def coarsening_strain_threshold(self) -> float:
        return float(
            self._ac["linear_system/adaptive_coarsening/strain_increment_threshold"].cpu()[0]
        )

    @property
    def coarsening_affine_min_size(self) -> int:
        return int(
            self._ac["linear_system/adaptive_coarsening/affine_embedding_threshold"].cpu()[0]
        )

    @property
    def coarsening_post_cg_max_iter(self) -> int:
        return int(self._ac["linear_system/adaptive_coarsening/post_coarsening_max_iter"].cpu()[0])

    @property
    def coarsening_max_row_ratio(self) -> float:
        return float(self._ac["linear_system/adaptive_coarsening/max_row_ratio"].cpu()[0])

    @property
    def coarsening_newton_stall_limit(self) -> int:
        return int(self._ac["linear_system/adaptive_coarsening/newton_stall_limit"].cpu()[0])

    @property
    def line_search_max_iter(self) -> int:
        return int(self._ac["line_search/max_iter"].cpu()[0])

    @property
    def contact_enable(self) -> bool:
        return bool(self._ac["contact/enable"].cpu()[0])

    @property
    def contact_d_hat(self) -> float:
        return float(self._ac["contact/d_hat"].cpu()[0])

    @property
    def init_collision_pair_capacity(self) -> int:
        return int(self._ac["contact/init_collision_pair_capacity"].cpu()[0])

    @property
    def ccd_partition(self) -> bool:
        return bool(self._ac["contact/ccd_partition"].cpu()[0])

    @property
    def ccd_partition_sv_max_iter(self) -> int:
        return int(self._ac["contact/ccd_partition_sv_max_iter"].cpu()[0])

    @property
    def intersection_check(self) -> bool:
        return bool(self._ac["contact/intersection_check"].cpu()[0])

    @property
    def contact_constitution(self) -> str:
        return str(self._ac["contact/constitution"].cpu()[0])

    @property
    def adaptive_kappa_mode(self) -> str:
        return str(self._ac["contact/adaptive_kappa_mode"].cpu()[0])

    @property
    def adaptive_kappa_tick(self) -> str:
        return str(self._ac["contact/adaptive_kappa_tick"].cpu()[0])

    @property
    def friction_eps_v(self) -> float:
        return float(self._ac["friction/eps_v"].cpu()[0])

    @friction_eps_v.setter
    def friction_eps_v(self, value: float) -> None:
        self._ac["friction/eps_v"].cpu()[0] = value

    def __getitem__(self, key: str) -> Any:
        key = _resolve_key(key)
        return self._ac[key].cpu()[0]

    def __setitem__(self, key: str, value: Any) -> None:
        key = _resolve_key(key)
        if key in self._ac:
            attr = self._ac[key]
            if attr.dtype == np.dtype("O"):
                attr.cpu()[0] = value
            else:
                attr.cpu()[0] = np.asarray(value, dtype=attr.dtype)
        elif isinstance(value, str):
            self._ac.create(key, str, default=value)
        else:
            arr = np.atleast_1d(np.asarray(value))
            if arr.ndim == 1:
                dtype = (arr.dtype, arr.shape[0]) if arr.shape[0] > 1 else arr.dtype
            else:
                dtype = (arr.dtype, arr.shape)
            self._ac.create(key, dtype)
            if isinstance(dtype, tuple):
                self._ac[key] = arr.reshape(1, *arr.shape)
            else:
                self._ac[key] = arr.reshape(1)

    def __contains__(self, key: str) -> bool:
        resolved = _DEPRECATED_ALIASES.get(key, key)
        return resolved in self._ac

    def attach_solver(self, solver) -> None:
        """Route later ``dt``/``gravity`` assignments to the running solver."""
        self._solver = solver

    def freeze(self) -> None:
        self._ac.freeze()

    def thaw(self) -> None:
        self._ac.thaw()

    def check(self) -> list[str]:
        """Return human-readable problems for every config key that violates its descriptor."""
        return self._ac.check()

    def describe(self, key: str | None = None) -> str:
        """Return a human-readable description of one or all config keys.

        If *key* is ``None``, describe every registered key.
        """
        if key is not None:
            return self._describe_one(key)
        parts: list[str] = []
        for name in sorted(self._ac):
            parts.append(self._describe_one(name))
        return "\n".join(parts)

    def _describe_one(self, key: str) -> str:
        attr = self._ac[key]
        val = attr.cpu()[0]
        dtype_str = "str" if attr.dtype == np.dtype("O") else str(attr.dtype)
        line = f"{key}  (type={dtype_str}, default={attr._default_value}, current={val})"
        if attr.desc is not None:
            s = attr.desc.summary()
            if s:
                line += f"\n  {s}"
        return line
