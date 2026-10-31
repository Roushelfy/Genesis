# Native rigid stress recovery: automatic surface-to-shell preparation

**Active user goal, 2026-10-31:** implement [automatic surface-to-shell preparation](docs/AUTOMATIC_SURFACE_TO_SHELL_GOAL_20261031.md).
A valid surface plus explicit shell backend/thickness should run without an authored Q4 NPZ, retaining
automatic radius + DPR, headless max and viewer full output. Reuse existing Q4 elasticity and native
Quadrants numerics; mesh conversion is cached build work. Validate closed/open and nonconvex shell inputs,
mechanical quality, numerical consistency and complete-path cost on low models before broad refinements.

This is an implementation goal, not a claim that automatic 2D conversion already works. The current
custom-shell example still needs a supplied Q4 NPZ until this goal is delivered. Keep the volume default,
explicit authored assets, original acceptance/lifecycle and all scoped prior evidence.

## Previous delivery: automatic DPR fast path and defaults

**Completed user goal, 2026-10-28:** the [automatic DPR
goal](docs/AUTOMATIC_DPR_FAST_PATH_DEFAULTS_GOAL_20261028.md) has implemented public defaults, native
selective integration, full-field/history qualification, matched performance/memory evidence and tested public
examples. Read the [results](docs/AUTOMATIC_DPR_FAST_PATH_RESULTS_20261028.md), [running
guide](docs/RUNNING_AUTOMATIC_DPR.md) and [22-clause
packet](evidence/20261028-automatic-dpr-fast-path-v1/README.md).

Ordinary volume DPR uses pressure/patch/Hertz/Hertz and adaptive angular q16 with independent checking and
selective q64/q128 repair. Original fit/wrench/K/lifecycle gates remain. Volume remains the backend default
and Q4 remains shell loading default; the measured shell recipe explicitly selects planar
k32/q10/radial/blocks/cap128, FP64/history0 and GraphOff. Sharp-box planar curvature is an explicit surrogate.

Measured engine is 070cb0c96386250dda3a85db703d5705d5c3fccc. B8192 max/full module gains over angle128 are
6.269/6.237; complete policy-inference gains are 5.617/5.577. All 102 original primary processes are accounted
for, including 12 retained edge16 warmup failures. Final eight 1200-step B4/B64 scripted/policy max/full
histories pass; the separate 41-point path exercises eight switches. Final 48 default/lifecycle tests and 20
loaded test executions pass.

Admission is empirically qualified on the declared finite corpus, not a universal spatial bound. Old strict
load-comparison negatives, extra adaptive memory and sharp-edge/continuum uncertainty remain explicit.
Original 12 unfinished files,64 prior and 21 campaign frozen sources remain preserved. This closes the
requested goal; unchanged prior campaigns are not new implementation prerequisites.

## Previous delivery: planar loading DPR optimization
**Completed, 2026-10-27:** the [native planar loading goal](docs/PLANAR_LOADING_DPR_NATIVE_GOAL_20261027.md)
has integrated code, independent accuracy/contact/lifecycle controls and a
29-clause/14-family delivery ledger. Read [the measured report](docs/PLANAR_LOADING_RESULTS_20261027.md),
[the execution ledger](docs/PLANAR_LOADING_PROGRESS_20261027.md) and
[the compact evidence packet](evidence/20261027-planar-loading-v1/README.md).

The original Q4 elasticity and1452/1446 elastic DOF remain unchanged. Native
Quadrants preparation/runtime preserve original Hertz sizing, source points,
friction, material identity, pressure law and original acceptance. Independent
CPU source is `797203553f246ca155bbef909c1bb33e7cd7b90c`; measured native source
is `6dedc79579ac0d113e4baac26eb538bcf51ca57d`. Later delivery commits are not
measurement-engine identities.

Keep Q4 as public default. The complete measured-workflow opt-in recipe is
planar k32/q10/radial/blocks/cap128, FP64 simulation/inverse, history0 and
process-local GraphOff. All168 fresh timing processes,48 paired blocks and16
cohorts qualify. At B32768, max/full module gains over tuned Q4 are1.994/1.940,
and complete policy-inference gains are2.033/1.987. B4 frozen module is about9%
slower; planar uses more memory. Full-field error on the measured64-condition
trajectory is at most0.0828008% against resolved Q4 integration.

All28 memory and14 profile workers pass;26 focused regressions pass. The
policy-max8192-to32768 gain10.0479906723% exceeds the frozen10% plateau
criterion. Full-memory screening rejects the optional65536 probe, so no
universal or global maximum is claimed. Original negatives, twelve unfinished
files and33 old/31 new frozen sources remain preserved. This closes the
requested discrete native/throughput comparison; ideal sharp edges, continuum
peaks and arbitrary nonconvex field certification retain their stated limits.
The workflow measures seeded MLP inference, not training.

## Previous delivery: matched CPU12 GPU comparison

**Completed, 2026-10-27:** the [matched CPU12 specification](docs/ATLAS_CPU12_GPU_PARITY_GOAL_20261026.md)
has an explicit decision for the CPU milestone and each native backend. Read
[the matched GPU report](docs/CPU12_GPU_RESULTS_20261026.md),
[the native shell/volume report](docs/CPU12_NATIVE_RESULTS_20261026.md) and
[the compact evidence packet](evidence/20261026-cpu12-gpu-parity-v1/README.md).
Its thirteen-cell requirement ledger distinguishes positive numerical checks
from scoped spatial/performance negatives.

The exact CPU mother, P/B and twelve spaces match in Quadrants, including S,
loads, all selectors/banks, common relief, original reduced residuals and
lifecycle. CPU/GPU tensor differences are below 2.7e-13. The published CPU
construction itself fails M/Q budgets. Fixed-support finer-coarse controls
retain this spatial negative. Packed scheduling improves an internal component,
but the complete matched module is 4.43–4.67 times slower than the optimized
FP64 inverse/surface mother in paired B16/B64 max/full measurements.

The native twelve-center shell and volume adaptations have actual smaller
operators, bounded tuning, geometric/trajectory/lifecycle controls, full-path
profiles, B4/B1024 max/full policy inference workflows and independent memory
measurements. Shell still fails Q; volume passes the declared four/64-condition
trajectories but fails nonconvex remote-hotspot controls. Both are slower than
the optimized mother in the measured complete workflows. Keep the mother as
the maintained default. Archive the native experiment at `94ec782fd` with
restoration commands; retain reusable matched-QD validation and the tested
cooperative/packed implementation. Twenty focused regressions pass.

This closes the requested discrete comparison and tuned backend decisions,
not arbitrary-surrogate or continuum sharp-edge accuracy. The workflow measures
observation, seeded MLP inference, controller and rigid/contact/stress work;
it makes no trained-policy or optimizer/training-throughput claim. The captured
maintained implementation pin is `09bc9fa4a` (engine identical to `323438018`);
the final documentation/archive delivery has a later branch HEAD. Original
unfinished files, prior frozen sources and earlier pressure/fallback/angular
winners and H/FP32/screen/history evidence remain preserved. Avoid unchanged
reruns unless a new user goal reopens the scope.

## Previous delivery: native local-refinement atlas

**Completed, 2026-10-26:** genuine shared-coarse/local Galerkin spaces,
per-feature source aggregation and merged recovery have native implementation,
numerical controls and separate shell/volume decisions. Read
[the measured report](docs/LOCAL_REFINEMENT_RESULTS_20261026.md),
[the execution ledger](docs/LOCAL_REFINEMENT_PROGRESS_20261026.md) and
[the six-cell evidence packet](evidence/20261026-local-refinement-v1/README.md).

Smaller feature operators pass original numerical, common-mother, rank and
virtual-work controls. Bounded spatial tuning, complete Franka max/full
accuracy/cost, heterogeneous/lifecycle controls, profiling and memory show no
useful complete-path gain on the measured models. Aggregation and one recovery
have useful internal gains, but depend on the unretained atlas. Experimental
code remains runnable at `822ad2854`; production code matches `4bc4ada25`.
Twenty cleanup regressions pass. Earlier pressure/fallback/angular winners
and the scoped H/FP32/screen/history negatives remain preserved.

The [review and delivered specification](docs/ATLAS_LOCAL_REFINEMENT_REVIEW_AND_GOAL.md)
and [CPU source guide](docs/ATLAS_CPU_REFERENCE_GUIDE.md) explain this follow-up.
Its real smaller spaces, separate numerical/spatial gates and tuned negative
results close the three reopened families in both backends. This does not
qualify arbitrary nonconvex spatial surrogates or continuum sharp-edge peaks,
and does not rule out other reduced-space constructions. Avoid unchanged reruns
unless a new user goal supplies new scope.

## Historical candidate delivery, scoped by the current review

**Completed, 2026-10-26:** all fourteen native candidate/backend decisions are
delivered. Native integration, numerical controls, targeted tuning, module and
complete Franka comparisons, memory and production cleanup are complete.
None of the seven incremental candidates provides a reproducible useful gain
in the measured regimes; their runnable native code and negative evidence are
retained. The maintained engine preserves the prior pressure/fallback winners
and adds the independently qualified optional angular/q128 pressure repair.
Read [the final report](docs/NATIVE_CANDIDATE_RESULTS_20261025.md),
[the seven-by-two ledger](docs/NATIVE_CANDIDATE_PROGRESS_20261025.md) and
[the requirement audit](evidence/20261025-native-candidates-v1/metadata/requirements-final.json).
Same-model checks do not qualify continuum sharp-edge peaks; broad physical
convergence remains outside this delivered integration/throughput goal.

**Delivered specification, 2026-10-25:** take all seven GPU candidates through
real native integration, targeted tuning and matched end-to-end measurement
for both midsurface shell and three-dimensional volume FEM. Track the seven
candidates separately for each backend; neither backend's result closes the
other. Each has its own mother space, gauge/relief, load maps and outputs.
Follow [the integration goal](docs/NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md).

The curved atlas mapping and common gauge/inertia relief are now explicit
implementation work. Simplified planar, H-only or recovery-only prototypes
cannot close these items. Connect each candidate to real native inputs,
required acceptance/state, max/full output and lifecycle. Keep measured
winners; revert production candidates only after a reasonably tuned native
comparison supports the negative result.

Keep all production numerical preparation/runtime in Quadrants. Preserve
the mature DPR load model and arbitrary moving/asymmetric convex/nonconvex
contacts, friction, surface sides, batching, volume/shell choice and lifecycle.
Use low-resolution numerical consistency and full Franka throughput first.
The [earlier candidate report](docs/GPU_THROUGHPUT_20261025.md) is the baseline
and historical evidence, with its unintegrated candidates still open under
this stronger goal.

---

## Previous delivery: GPU candidate evaluation

**Completed, 2026-10-25:** the applicable candidates, integrated pressure and
exact-fallback improvements, focused regressions, paired module/scene timings,
memory and negative results are delivered. Read [the report](docs/GPU_THROUGHPUT_20261025.md)
and [the requirement audit](evidence/20261025-gpu-throughput-v1/metadata/requirements-final.json).
The heterogeneous volume bank retains the same frame-221 pressure-fit rejection
before and after optimization; no successful rate is claimed for that workload.
Curved-mesh atlas mapping and continuum contact peaks remain unqualified.

**Delivered specification, 2026-10-25:** implement and evaluate the applicable GPU
optimizations in [the follow-up goal](docs/GPU_THROUGHPUT_FOLLOWUP_GOAL_20261025.md).
Maximize useful multi-environment throughput in the existing rigid solver,
with production numerical preparation and runtime entirely in Quadrants.

Cover shared load preparation and multi-RHS grouping; few feature meshes per
contact with continuous transitions; common-mother recovery and max/full
fusion; complete/packed response operators with independently tested FP32 and
other supported precisions; conservative global max screening; history
prediction, failed-environment compaction, buffer rotation and cached Kq;
and applicable condensation/reduction controls. Measure each candidate's
incremental value against the already optimized native implementation.
Prioritize the actual profile, including pressure integration and fitting.

Preserve mature DPR loads, arbitrary moving/asymmetric convex/nonconvex
contacts, normal/friction eccentricity, surface sides, volume/shell choice,
max/full output, numerical acceptance, batching and lifecycle behavior.
Use coarse meshes for numerical consistency and throughput tuning before
new high-resolution physical convergence work. Deliver usable improvements,
paired module/Franka scene benchmarks, memory, recommendations and a complete
candidate ledger; document negative and inapplicable outcomes.

This explicit follow-up supersedes the previous completion/atlas-only scope
for the candidates above. Earlier delivered results remain accepted within
their recorded scope.

---

## Previous delivered goal: native thin-shell DPR integration

**Completion, 2026-10-24:** the native implementation, focused regressions,
complete-scene/module throughput, profiles, memory, configuration comparisons
and targeted global physical checks are delivered. Read
[the final report](docs/SHELL_DPR_NATIVE_20261024.md) and
[the requirement audit](evidence/20261024-native-shell-dpr-v1/metadata/requirements-final.json).
The task is complete within that scope; local three-dimensional contact peaks
remain explicitly unqualified. The requirements below record the delivered goal.

**User direction, 2026-10-24:** implement an optional thin-shell elastic-recovery
backend inside the existing rigid stress feature, entirely in Quadrants.
Follow [the active goal](docs/SHELL_DPR_INTEGRATION_GOAL_20261024.md).

Keep mature native automatic-radius + finite-pressure DPR and its numerical
checks. Preserve arbitrary changing convex/nonconvex and asymmetric contacts,
normal/friction forces including eccentric moments, max/full outputs, batching,
isolation and lifecycle behavior. Keep volume FEM available.

The gripper box keeps ideal sharp edges for the current implementation, as
confirmed by the user. Fillet/chamfer dimensions are not a prerequisite; any
reported edge peak must retain its sampling and convergence limitations.

Start from the completed [CPU study](shell_cpu/README.md) and
[algorithm notes](shell_cpu/ALGORITHM.md). First integrate a low-resolution
native path and verify numerical consistency. Prioritize detailed profiling
and all applicable throughput optimization on the full Franka/hollow-egg
workflow, across small and useful large batches. Keep high-resolution physical
convergence campaigns behind this performance phase.

Report shell in-plane membrane/bending stress honestly: the CPU tiny-contact
egg peak is unresolved and does not certify a three-dimensional maximum.
Preserve the native DPR load law; the main CPU projected-Hertz variant is not
an exact native transfer replacement.

## Previous delivery

The volume-DPR throughput task is complete within its measured numerical and
performance scope. Retain [its report](docs/DPR_THROUGHPUT_20261023.md) as the
baseline and [its requirements](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md) as
historical context. Earlier conflicting enrichment, SCR-repair, physical
ranking and sharp-edge campaigns do not override the active shell goal.

Follow existing storage and reproduction policies. Deliver usable native code,
focused tests, matched module/scene timings, memory, recommended configurations
and explicit accuracy limitations; documentation alone does not complete the goal.

