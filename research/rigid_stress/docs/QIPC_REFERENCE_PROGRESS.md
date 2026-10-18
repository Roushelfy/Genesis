# QIPC reference comparison execution status

The [full goal](QIPC_REFERENCE_COMPARISON_GOAL.md) is active. Previous shell and
same-mesh FEM evidence does not qualify a common three-method contact reference.

## Source inspection and initial build

2026-10-18 UTC: task branch fast-forwarded to goal commit918ccf43f. Remote
`Roushelfy/cuda-graph-qipc:main` is still
`fe252bd30acc6c717507b74add31512812f48761`. A detached original checkout is at
`../qipc-stress-reference-original`. Four required submodules are pinned to
their authored commits; no tracked QIPC source is patched.

The revolute virtual-work torque fix `ee1673f74` is already in main. The
separate theta-unwrap commit `c8b27b451` and research integration641459b88/later
FP64/PCG-quality branches were inspected as selection/qualification context;
their binaries are not silently loaded into this main checkout. Existing
research quality is task-specific, with recorded cap/initial-gap failures; it
does not qualify the new hollow-egg scene.

The actual SNH energy includes `-mu/2 log(Ic+1)` and a shifted volume term;
its stored parameters are `mu_snh=4mu/3`, `lambda_snh=lambda+5mu/6`. Stress
recovery must differentiate that exact energy, rather than a textbook
neo-Hookean formula or projected Hessian. StVK-Hencky uses another energy.

The first native build is prepared with runtime checks, full FP64-capable
source, Release, sm120, no viewer/stub generation, and an unchanged existing
Python3.12/Torch2.11 environment. CMake, objects, runtime package, CCCL, caches,
logs and exact commands live under the new data run. The source checkout's
generated CCCL path is a compatibility link into that data root. Dependency
source caches are reused read-only; no environment is upgraded or overwritten.
The reusable submitter now captures QIPC/CMake/vendored sources and submodule
identities, plus the new comparison package. Its original Genesis defaults
remain available.

Actual build726135 completed all453compile/link steps (Release solver image
62422704B, SHA256`cebd257411c97a01e5d0c78ca8d5fb6271955c9b3b5c20ce3e53dab95e175cf1`),
then failed its import because this main's top-level package imports the viewer
even for headless use. The failure/log/runtime remain immutable. Build726150
adds the viewer through the same completed incremental build cache, with fresh
logs/runtime output and no tracked QIPC source patch. No solver rebuild is
inferred without inspecting the actual build log.

Controls726147 completed. A host C++ probe compiled against the **unmodified
generated SNH energy/gradient header** agrees with the CUDA Cauchy/Piola observer
to7.22e-16 at E=1. Rotation covariance and small-strain E/nu controls pass.
SNH and Hencky finite-strain differences remain distinct. Fixed-domain
refinement keeps the level1 mass at6.5298482104g while lumped inertia error
decreases9.871% to0.6175%; the separately rounder level2 geometry has a different
volume/mass and is not mislabeled numerical refinement. These checks do not
qualify contact dynamics or a complete grasp.

Public `RigidStressOptions.dpr(...)` / `.scr(...)` constructors are implemented
as complete option presets with the old explicit identifiers. All other defaults
and numerical kernels are unchanged. Full-field/max shared-operator tests now
exercise DPR and verify complete SCR serialized equivalence. CPU/CUDA execution
of that focused test passed (job726151; one test per backend).

Build726150 completed/imported successfully. Its eight compile/link steps build
the viewer; the solver SHA above is unchanged. The first failed runtime and
import log remain preserved. Successful runtime extensions are explicitly
linked into the isolated original checkout.

## First actual dynamics and numerical controls

Original-main hollow-egg/plane workflow726153 completed20 steps of1ms,
including actual barrier/friction contact gradients, unprojected elastic
Cauchy fields and momentum/energy diagnostics. Its final local force residual
was1.98% of nodal contact-force norm, with15 linear-cap steps. Successful
process completion does not establish physical-reference qualification.

Bounded controls726158 retain the same3840-tet input and unmodified main.
With diagnostics enabled, default float MAS reaches4096 PCG iterations on15/20
steps; its final recursive residual increases in the last Newton system.
Double MAS has0/20 such cap steps and final local force residual4.45e-6
relative to contact. Standard versus partitioned PCG under double MAS gives
13294.9801 versus13294.9806Pa final raw VM peak. Actual sparse systems and
residual histories are retained. This is a precision diagnosis, not spatial/
contact/friction convergence or grasp qualification.

The same job ran independent CPU P1 isotropic-stiffness/lumped-mass implicit
Euler controls against actual QIPC FEM dynamics, for SNH/Hencky and two load
scales. Final volume-weighted tensor discrepancies range7.0e-7 to4.23e-4.
The tight1e-8m/s SNH/full-load point hits its256-Newton cap after line-search
stagnation; retain that failure and inspect stopping/energy precision before
admission. Constitutive and dynamic differences are recorded separately.

Source audit finds original main does not import MJCF damping/armature; drive
kv also lies outside its proportional-force cap. These are material mismatches
for the original Franka task. Integration641459b8826bb061e1bba01dfe9eba39793c3855
contains authored passive terms and hard minimal-coordinate joints, including
commit a9206cd11. Its previous native build has an exact matching-source receipt
and SHA8763018e8ca49e2b7c156cc5cdcf42318b03dfd0363618c3b824daaca39e1213.
An isolated detached candidate explicitly reuses that image. It is a different
candidate from fe252bd30 main; new egg and robot controls must qualify it before
use. Old task quality is not transferred to this egg.

No three-way stress-error ranking, complete-grasp qualification or new rate is
claimed. Exact receipts live under `runs/20261018-qipc-reference/` on data,
including `linear-controls-v1` and `meta/integration-native-reuse-v1`.

## Completion gates

1. Exact-energy stress/parameter and independent small-control validation.
2. Matched 3D hollow egg/robot inputs and both prescribed-motion and capped
   control protocols, with dynamic histories and declared contact mismatches.
3. Reference mesh/thickness/contact, dt, solve and friction regularization
   uncertainty on controls and a complete representative grasp.
4. Common-position/time tensor/resultant/peak errors for DPR/SCR/shell,
   reference-load replay versus point-wrench reconstruction and full feedback.
5. Lifecycle, batch/restart and failure evidence; separate stock/diagnostic
   variants and retain unresolved quantities.
6. Full-field primary/max-only secondary cost, useful tuned settings, matched
   hardware/memory/profiles, figures, detailed report/reproduction/source closure.
7. Complete requirement audit, focused checks, final commit/push.

All gates remain required; they are not reduced to the inexpensive first checks.
