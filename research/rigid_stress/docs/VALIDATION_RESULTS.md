# Acceptance evidence for rigid full-shell stress recovery

This audit follows A--I in `GOAL.md` and keeps equation accuracy, load-model
accuracy and physical discretization accuracy separate. Numerical runtime
solver kernels are fixed at `80942f8ee9efa3ca002c3db142a1a30b95ed066e`. Later
memory-lifetime fixes share immutable operators with the eager oracle,
release temporary eager displacements and release the replay's previous RHS
before mapping the next frame after measured B=256 memory failures. The
current complete source commit is recorded separately. The final
calibration selection, held-out oracle and timing results are still running;
their pending items below must be completed before full-goal acceptance.

Reproduction commands and allocation requirements are in
[RUNNING.md](RUNNING.md). The fresh-process
[acceptance suite](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/suite.json)
passes all thirteen selected checks. That short suite uses coarse full meshes
for integration tests. Fine-shell physical and numerical acceptance relies on
the separate evidence below, rather than treating the short suite as a mesh
convergence test.

## A. CPU reference and clean full P2 mechanics

The historical `reference/` tree is preserved. The new implementation assembles
the complete fixed shell stiffness, consistent mass, six rigid modes and
mass-weighted inertia relief independently, retaining a CPU FP64 direct path.
CPU-only imports do not require Genesis, Torch, CuPy or cuDSS. The optional
dense Torch factor is an arithmetic diagnostic, not the scalable backend.

[Operator comparisons](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/mechanics/result.json)
compare every assembled stiffness/mass entry, rigid modes, Gram matrix,
compatible loads, complete residual and peaks against the preserved independent
P2 math, with both MMD and surface-column nested dissection. Maximum scaled
stiffness-entry error is 3.17e-16; mass-entry error is zero. The
[CPU adapter check](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/cpu/result.json)
also exercises a different gauge and independent partial resets.

[Device field checks](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/gpu-fields/result.json)
use B=1/3/8/17, nonsquare C/F/strided RHS permutations, complete residuals,
zero/nonfinite inputs, freefall and centrifugal loading. GPU affine strain
patches and arbitrary displacement fields compare the fused global reduction
against independent full `D B u` evaluation at every tetrahedron corner.
The exact six-field centrifugal cache and fused nodal wrench products compare
every full RHS entry with CPU assembly. Freefall stays below 1e-3 Pa and spin
produces nonzero stress. Fine level-6
[native direct/CPU comparisons](../evidence/20261009-gpu/20261009-clean-math-cudss-level6.json)
pass, with full relative residual about 2.12e-12 and peak error 3.12e-13.

The additional [native gauge/pose check](../evidence/20261009-performance/raw/20261009-invariance/result.json)
passes alternate gauge rows, complete native GPU equilibrium and canonical
displacement against the CPU oracle. It also sends five random world poses
through the actual adapter's frame/sign conversion in each F/C layout, with
seven synthetic environments and contact counts 0--6. Maximum full RHS entry
difference is 2.89e-15 N, peak difference with a 1 Pa denominator floor is
4.94e-11, and alternate-gauge canonical displacement error is 2.51e-10.
These fixtures are explicitly untimed synthetic QA, distinct from actual
trajectory checks. Its exact [script](../evidence/20261009-performance/raw/20261009-invariance/check_invariance.py)
runs with `PYTHONPATH="$PWD"` from the repository root and requires `--output`.

## B. Ordinary rigid Panda/egg scene

The Panda is a rigid URDF robot and the egg is an ordinary rigid link with
mass, COM and inertia derived from the hollow shell. The complete level-6
collision exterior has 81,920 triangles, convexified without decimation.
Elastic displacement does not change collision or motion. The phases include
approach, clamp, lift, hold/translate, deliberately weakened grip/slip, lower,
release and reset. Varied environments have independent delays and initial
pose, friction and finite-footprint parameters.

The actual nominal level-6 rollout verifies lift and hold **1/1**, with maximum
tangential force 0.23455 N. Its
[600-frame video](../evidence/20261009-performance/visualization/nominal.mp4)
and [hold frame](../evidence/20261009-performance/visualization/hold.png)
show the real Panda/egg scene. The held-out level-6 seeds 610000--610031
complete 1,200 steps and 32 resets but achieve **0/32 keep-hold success**.
This is an untrained scripted controller; auxiliary stress acceptance does
not imply robust grasping. Failed grasps remain in the measurements.

The fine varied rollout includes 335 egg-as-A and 68,593 egg-as-B contacts,
nonzero table/support transitions and maximum tangential force 1.52299 N.
Its record-only run does not claim per-frame CPU solves. A separate
[four-environment nominal run](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/panda-nominal-substeps/result.summary.json)
uses four physical substeps and a partial reset at step 107; all four grasps
succeed and CPU peaks are checked at every substep. The
[32-environment coarse suite](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/panda-varied32/result.summary.json)
checks actual varied contacts and independent temporal histories against CPU
direct peaks; its success is 5/32, and its coarse peaks are not the final
physical mesh profile.

## C. Finite footprint, complete loads and wrench conservation

The declared law uses compact Gaussian pad weights with nonnegative pressure
and a constant traction/normal ratio matching each actual contact force.
An exterior footprint anchor may move along the force line, preserving the
point resultant and its moment exactly. Radius, position, direction and
contact count are runtime inputs. Surface integration retains every admitted
positive order-10 quadrature point; spatial indexing caches geometry only.
Signed P2 nodal weights do not imply tensile physical pressure.

[CPU finite-wrench tests](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/wrench/result.json)
cover changing asymmetric patches, positivity, friction admissibility, active
constraints and rejected infeasible/tensile data. The general CPU vector
traction path also covers an admitted pure moment. The production force-only
observer explicitly disables spin/rolling friction and rejects an enabled
configuration; it does not silently drop those moments.

[GPU mapping tests](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/gpu-pressure/result.json)
compare complete nodal loads, force, moment and same-input CPU peaks for
counts 0--7, arbitrary positions, saturated friction, overlapping footprints,
scan/grid and atomic/warp reductions. Interior points shifted by 8 mm retain
their wrench under force-line anchoring. Infeasible inputs are rejected.
Mapping force and moment errors are reported separately in N and N m.

The fine actual varied rollout's maximum pressure moment mismatch is
9.13e-19 N m. The Newton--Euler force and torque discrepancies are 6.34e-4 N
and 1.47e-5 N m; public rigid constraint gradients agree to 7.01e-16 N and
4.33e-17 N m. The finite rigid iteration/integration discrepancies remain
visible rather than being hidden by a modified recovery wrench.

A real narrowphase normal of length 1.000005055531268 violated the constraint
frame's unit-normal assumption and produced an inadmissible reconstructed
friction force. The engine now normalizes a normal before storing it and
solving constraints. CPU and GPU regressions pass for the actual case and
other normal lengths. This fixes the constraint frame, without load clipping
or a friction increase.

## D--E. Shared native sparse factors and measured candidates

The scalable backend is public cuDSS 0.8 FP64 sparse direct factorization,
one immutable factor per complete operator and many RHS. Level 6 has
2,457,630 DOFs, 491,520 affine P2 tetrahedra and 2,322,190,270 native factor
nonzeros. Native analysis/factorization are reported outside timing. The
exported SuperLU/cuSPARSE SpSM baseline remains available for supported sizes.
Fine CPU CVXOPT native solves remain available even when factor export fails.

The device hot path retains complete mapping, gravity, centrifugal load,
inertia relief, every gauge-row residual and a full-domain four-corner VM
maximum. Only a scalar per environment enters policy observations. Invalid
mapping/recovery gates that environment's observation to NaN. The
[substep regression](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/gpu-substeps/result.json)
checks one/four substeps, partial resets and rejection/recovery; sampling only
the last of four substeps would miss 385,592 Pa in that test.

[Device history tests](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/native-temporal/result.json)
cover independent Q/KQ/Gram caches, rank deficiency, circular rollover,
invalid timesteps, abrupt changes, no-load environments, subset resets,
compact/padded correction, raw/scaled FP32 refinement and FP64 fallback.
Cached and rebuilt Gram, memory layout, chunks and all-failed controls are
also measured in the [performance checkpoint](PROGRESS_20261009_PERFORMANCE.md).
History and FP32 are retained options, not assumed speedups.

[Captured native recovery](../evidence/20261009-performance/raw/20261009-acceptance-suite-final/native-graph/result.json)
passes 96 changing-RHS CPU cases across default/nondefault caller streams and
F/C layouts, including zero loads and spin. It captures only complete recovery;
live mapping/body work remains outside capture and inside timed wall scope.
No private third-party library mutation is used. The measured allocator,
layout, temporary-tensor, graph and natural-order ablations are retained.

## F. Physical mesh and quadrature acceptance

The fixed material is E=10 GPa, nu=0.3, density=2000 kg/m3 and wall thickness
0.5 mm. These are declared assumptions, not measured biological properties.
The [force-line suite](../evidence/20261009-gpu/20261009-convergence-force-line-live.json)
uses complete levels 4/5/6 and quadrature orders 6/10/16 for twelve asymmetric
finite loads plus six actual approach/clamp/lift/hold/slip/release snapshots.
All eighteen final refinements pass the 2% criterion. Maximum level-5-to-6
peak change is **1.3645%** and order-10-to-16 change is **8.412e-9 relative**.
Hot positions and nonmonotone sequences remain in the raw evidence.

The independent [wall-layer sweep](../evidence/20261009-gpu/20261009-convergence-force-line-wall.json)
uses one/two/four layers at level 4; maximum two-to-four-layer change is
1.0625%. This separate check is not a combined universal error bound.
The [raw point-anchor load-law suite](../evidence/20261009-gpu/20261009-convergence-full-pad-level6-native.json)
also passes complete levels 4/5/6, but its different pressure placement is
reported as a load-model sensitivity rather than a uniquely identified peak.
No quarter model or shrinking point-load footprint is used for acceptance.

## G--H. Final numerical and throughput acceptance

The [held-out CPU oracle](../evidence/20261009-performance/raw/20261009-heldout-oracle/summary.json)
completes all sixty level-6 frames at stride 20, 32 columns each. Its 1,884
reference peaks above 1 Pa have maximum/p95/mean relative error
8.670e-12 / 2.095e-12 / 7.480e-13. The 36 near-zero cases have maximum
absolute error 2.161e-9 Pa. Maximum complete RHS entry difference is
3.990e-17 N, maximum complete absolute residual 2.865e-12 N and maximum
relative residual above the absolute floor 1.557e-10. Both budgets pass.
The [raw report](../evidence/20261009-performance/raw/20261009-heldout-oracle/strict-cpu.json),
per-column CSV and [RHS manifest](../evidence/20261009-performance/raw/20261009-heldout-oracle/rhs-manifest.json)
retain all samples and sixty complete RHS files on data (35.543 GB compressed,
37.749 GB logical RHS). Every compressed file is independently hashed.

Pending: freeze the fastest eligible complete calibration candidate, and finish rigid, matched policy-rigid,
stress replay, live and policy held-out scopes. Each primary scope requires
three complete independently phased grasps with at least ten seconds per
repeat. Full replay warmup separately compares every GPU frame with eager
same-RHS FP64 direct recovery; sixty CPU samples are not 1,200 CPU solves.

Residual acceptance uses `abs <= max(1e-11 N, rtol*||b||)` with strict rtol
1e-6 and throughput rtol 1e-3. Peak budgets are 1e-4 and 1e-2 relative for
reference peaks above 1 Pa, with a separate 1e-3 Pa near-zero budget. Reports
must retain max/p95/mean error, absolute residual, near-zero errors and counts.
The same validated FP64 point can satisfy both budgets; no precision or
fidelity relaxation is implied. Physical mesh error is reported separately.

## I. Artifacts and branch isolation

Setup/demo/check/benchmark entry points and their explicit data paths are in
[RUNNING.md](RUNNING.md). Small raw CSV/JSON, source configurations and the
real-scene video are committed. Large contact histories and complete RHS stay
under the explicit data root with SHA-256 provenance in artifact manifests.
Measurements retain starting source hashes, actual hardware/native library
versions, memory accounting and exact command scopes.

Pending: commit the frozen winning configuration, final oracle/timing raw
evidence, final performance report and completed audit, then push the task
branch. The source checkout and the user's unrelated jobs/branches remain
untouched. Historical checkpoints keep their original incomplete status.
