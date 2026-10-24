# Native thin-shell DPR: implementation, accuracy and measured cost

**Delivered, 2026-10-24.** All twelve integration requirement groups are verified
within the measured scope and limitations below.

The optional native Q4 shell backend is implemented in the existing rigid
stress recovery. Volume FEM remains the default. The shell provides useful
membrane/bending observations and separate resultants; it is not a faster
replacement for volume DPR on the measured coarse egg. The gripper retains
the requested sharp edges. The unresolved tiny-contact 3D peak remains
outside this delivery's accuracy claims.

See the [running guide](RUNNING_NATIVE_SHELL_DPR.md),
[integration requirements](SHELL_DPR_INTEGRATION_GOAL_20261024.md),
and [tuning chronology](SHELL_DPR_NATIVE_PROGRESS_20261024.md).

## Implementation and acceptance

Production preparation and runtime use Quadrants: curved Q4 operators,
translation/rotation DOFs, membrane/bending/shear, rotary inertia, six-mode
relief, centrifugal loading, prefactorized solves and exact surface-load
response. Identical links share fixed data. The physical bilinear inner/outer
faces retain native support-ball DPR and eccentric friction couples. The
implementation does not substitute the CPU projected-Hertz model.

Max mode reduces the exact authored samples without allocating full fields.
Full mode exposes both sides at four Gauss points and body-frame membrane,
bending and transverse resultants. These are sampled in-plane stresses, not
physical global 3D maxima. Public cadence, environment isolation, release,
slot changes, partial reset and checkpoint invalidation are retained.

Final native engine measurements use `50a8e3bf3` (`native-shell-v16`). Later
delivery changes add documentation, analysis helpers and an explicit shell
label to the example's printed maximum. The remote atlas CPU study was merged
without changing this implementation or its active scope.

Nine shell regressions and six focused volume pressure regressions pass.
Independent controls cover stiffness, rigid modes, spinning loads, full fields,
warped/concave faces, asymmetric inner/outer loading, rejected inputs and
release/recontact. Final contact generalized-load differences reach 3.22e-7
on the coarse egg; the independent oracle's 48/96-order change is 3.16e-7.
Same-load field differences are below 8e-11. A separate analytical two-side
disk-moment control passes both Hertz and original DPR profiles, with load
errors below 1.8e-15 and field errors below 7e-12.

All 600 Franka frames are retained. Compared with the pre-fusion tuned source,
rigid positions are bitwise identical and load/field/radius audits pass their
original budgets. Seven captured complete fields agree with independent CPU
shell recovery to 2.59e-10 relative error; complete residual is at most
2.44e-13. Membrane/bending/transverse differences are below
4.45e-12/4.25e-10/9.60e-11. These are implementation checks for the same model.

## Complete workflow and memory

RTX PRO 6000 Blackwell Server Edition; FP64; `dt=0.01`; one substep;
level-1 egg; frozen four-condition bank repeated across environments;
native controller; original contact settings; automatic Hertz radius and
Hertz pressure; quadrature 16; explicit planar box-edge curvature surrogate.
Each timing process warms up for 600 steps, then measures three separate
600-step cycles including approach, close, friction, motion and release.
Setup, exports, reference solves and intrusive profiling are excluded.
The saved-state restart precedes each timed cycle; these cycle rates exclude
the between-cycle reset, whose correctness is covered separately.
These rates describe observation of the fixed controller, not stress-feedback
control or proof that every grasp succeeds physically.

| Environments | Configuration | Median ms/step | Env-steps/s | Stress MiB |
|---:|---|---:|---:|---:|
| 4 | Rigid | 2.26 | 1,768 | — |
| 4 | Volume max | 5.02 | 797 | 65.4 |
| 4 | Volume full | 4.98 | 803 | 65.8 |
| 4 | Shell max | 7.13 | 561 | 42.9 |
| 4 | Shell full | 7.16 | 559 | 43.8 |
| 64 | Rigid | 2.37 | 26,956 | — |
| 64 | Volume max | 5.14 | 12,443 | 78.6 |
| 64 | Volume full | 5.21 | 12,279 | 85.2 |
| 64 | Shell max | 7.40 | 8,643 | 54.4 |
| 64 | Shell full | 7.32 | 8,741 | 68.0 |
| 1,024 | Rigid | 2.94 | 348,073 | — |
| 1,024 | Volume max | 9.42 | 108,682 | 289.1 |
| 1,024 | Volume full | 9.48 | 108,063 | 394.1 |
| 1,024 | Shell max | 15.80 | 64,792 | 237.4 |
| 1,024 | Shell full | 15.79 | 64,835 | 454.9 |
| 8,192 | Rigid | 9.45 | 866,443 | — |
| 8,192 | Volume max | 45.95 | 178,262 | 1,860.6 |
| 8,192 | Volume full | 46.32 | 176,862 | 2,700.6 |
| 8,192 | Shell max | 85.65 | 95,650 | 1,603.8 |
| 8,192 | Shell full | 87.86 | 93,236 | 3,343.8 |

All 20 complete-cycle configurations have zero invalid environment steps in
every repeat. Attempted and admitted rates therefore coincide.

Memory in the table is unique native **stress-buffer** allocation, including
shared factors/responses and per-environment state/output. It is not total
process/GPU memory. Raw receipts also include Torch peak allocations, reset
state, device free memory and individual buffers. Full shell output stores
separate resultants in addition to two-sided stress, so at large batches it
uses more memory than volume full output even though it has fewer DOFs.

The shell has 1,452 DOFs and 240 Q4 elements; volume has 2,430 DOFs and 480
quadratic tetrahedra. Their outer geometry and collision scene match. The
shell's offset inner surface differs from the volume interpolation by up to
54.26 micrometres; the asset metadata records this discretization difference.
The timing comparison is not a claim that both fields are physically equal.

| Environments | Volume max ms | Volume full ms | Shell max ms | Shell full ms |
|---:|---:|---:|---:|---:|
| 4 | 1.07 | 1.07 | 4.58 | 4.55 |
| 64 | 1.03 | 1.07 | 4.52 | 4.54 |
| 1,024 | 3.64 | 3.80 | 10.45 | 10.88 |
| 8,192 | 22.80 | 23.04 | 64.19 | 66.49 |

Module measurements use the actual loaded frame after 130 warmup steps,
then five repeats of 100 recoveries, with its matching pre-integration frame.
They are frozen-input module costs, not complete robot throughput. Manual
6-mm-scale radius, with the same Hertz pressure profile, costs about
56.0/55.6 ms per complete B64 max/full step. This changes the support size
and physics; its ratio to automatic radius is not an equivalent-model
optimization speedup.

## Profiling and optimization decisions

At B64/full, intrusive measurements show these improvements:

| Change | Before | After | Scope |
|---|---:|---:|---|
| Analytic Q4 inverse | 13.53 ms | 3.81 ms | Anchor search |
| Parallel output invalidation | 14.15 ms | 0.051 ms | Acceptance |
| Conservative force-line bounds | 3.93 ms | 1.09 ms | Anchor search |
| Q4 moment reuse | 12.63 ms | 7.49 ms | Pressure integration/fit |
| Precomputed bilinear coefficients | 9.09 ms | 8.44 ms | Anchor/pressure/scatter |
| Reused constrained-fit geometry | 6.85 ms | 4.11 ms | Pressure integration/fit |
| Four-warp face scheduling | 4.11 ms | 1.92 ms | Pressure integration/fit |

These stage measurements are not additive and their ratios are not multiplied
into an end-to-end speedup. Compact force/rotation mapping reduces scatter
atomics from twelve to six per physical node. Before the last pressure change,
paired recovery fusion improves B4/B64/B1024 module medians by approximately
3.45%/2.72%/1.12%, with unchanged memory and acceptance.

The final secondary device profile disables graph capture to expose kernel
work. At B64, constrained pressure correction is 1.44 ms and anchor search
1.07 ms. At B8192, numerical Q4 moment integration is 33.99 ms, constrained
correction 10.68 ms, scatter 5.20 ms, sampled full stress 4.92 ms, and packed
response 3.68 ms. Thus pressure integration, rather than the linear solve,
remains the dominant large-batch cost. Graph-enabled zero profiler counters
are not interpreted as zero work or launch overhead.
The fused native witness/curvature/radius kernel takes about 0.15 ms at B64
and 0.41 ms at B8192; association takes about 0.15 and 0.43 ms respectively.

A separate 50-step loaded moving trace (ticks 130–179) at B64 records 153/233/239 device kernels per
step for rigid/volume-full/shell-full. Driver launches are 30/30/58 per step,
with 13 CUDA runtime launches in each path; kernel count and launch count are
different quantities. Host-to-device copies are 7/7/15 per step, primarily
small runtime parameters. Device-to-host copies and stream synchronizations
are 0.1 per step in all three traces; one final device synchronization is also
included. Host/device durations overlap and are not added to headline rates.
Host-to-device traffic is 8,864/11,184/12,792 bytes per step respectively;
device-to-host traffic is 25.6 bytes per step. An earlier approach-stage trace
is also retained, separately from the final loaded trace.

At B64, shell operator/factor/inverse/surface-response preparation takes about
2.45/1.23/0.51/0.33 seconds, plus 0.58 seconds for surface quadrature/grid;
volume takes 1.27/14.15/0.51/0.27 plus 0.48 seconds. Whole-scene cold setup is
about 106 versus 133 seconds in these processes. These include JIT and are
separate from steady-state timing.

At B8192/full, tested module configurations cost 65.20 ms for default packed
response, 64.13 ms for the old block/residual scheduling, 80.75 ms for full
inverse, and 149.66 ms for FP32 inverse storage. The 1.6% scheduling difference
does not establish a useful default change. FP32 storage also disables the
current surface-response/fused route and needs residual correction; this is
a configuration comparison, not an isolated FP32 arithmetic benchmark.

Diagonal equilibration passes controls but gives no runtime gain here.
Natural ordering increases factor storage from 12,466 to 63,606 blocks and
construction from 1.47 to 17.70 seconds. Defaults retain unscaled minimum
degree and translation-only gauge. Legacy gauge and mixed storage pass the
focused FP64-arithmetic controls. Full FP32 arithmetic fails the strict plate
control (original-operator residual 5.32e-4), and its rejected fields remain
retained. Full stress-response matrices and new persistent witness caches were
not introduced: current device packing/fixed-geometry reuse is measured, while
quadrature dominates and full output response would add quadratic storage.

## Targeted physical checks after tuning

The native sphere uses production inertia relief under uniform pressure.
The plate uses native operators/factor with research-only simply supported
constraints, not a new public support API. Both are compared with analytical
solutions and independent `pyfe3d==0.10.0` mixed-integration shell elements
under identical applied nodal loads. The original/current applied loads,
native/reference fields and residuals are retained.

| Control | Coarse | Fine |
|---|---:|---:|
| Sphere mean membrane error, n=4/8 | 4.79% | 1.24% |
| Sphere mean radial displacement error | 6.95% | 2.04% |
| Sphere full tensor difference from independent FEM | 9.04% | 3.36% |
| Plate center displacement error, n=8/16 | 0.484% | 0.0763% |
| Plate complete bending tensor error | 9.38% | 4.71% |
| Plate full tensor difference from independent FEM | 12.53% | 6.19% |

Sphere membrane/displacement references are `pR/(2h)` and
`pR²(1−ν)/(2Eh)`; plate uses the independent Navier Fourier solution at the
same Gauss points. The sphere uses R=25 mm, h=0.5 mm and p=100 Pa; the plate
uses side=100 mm, h=0.5 mm and p=10 Pa, both E=10 GPa and ν=0.3. Sphere
complete residual is below 6.8e-15 and plate free-row residual below 4.9e-11.
The independent formulation has its own discretization error; it is not
treated as exact. Code_Aster was unavailable in the compute image. A historical
OpenSees note inherited in the raw availability metadata is not a current
installation probe or an executed comparison.

These resolved global-load checks do not validate the egg's approximately
45-micrometre contact footprint in a 0.5-mm wall. Neither shell in-plane
stress nor the coarse volume baseline establishes its local 3D peak. Sharp
edge singularity/sampling limitations remain explicit.

## Recommended use and reproduction

Keep volume as the default when throughput or volume Cauchy components are
needed. Select shell for thin-wall membrane/bending/resultant observations
with suitable geometry and loading scales. Use max output when only the
sampled maximum is needed, FP64 arithmetic/storage, the default packed response,
minimum-degree ordering and verified Hertz moment reuse. Choose batch size
from the measured rate and memory together; the useful measured range ends
at 8,192 and no claim is made about a larger optimal batch.
The throughput recommendations cover this level-1 asset and GPU. Higher mesh
levels, asynchronous phase distributions, other GPUs and shell PCG throughput
are unmeasured here.

Full runtime data, logs, arrays, source inventories and failed attempts are
under `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261024-native-shell-dpr/`.
Scientific work runs only in Slurm `rtx-mid`, image `1_26`, with private caches.
The optional independent reference is installed in a project data overlay;
the active environment was not modified. Small source assets and publication
metadata remain in Git. The original twelve unfinished files and all frozen
checkouts are preserved. Initial compilation failures, an earlier diagnostic
JSON failure, the missing-reference attempt and FP32 rejection remain recorded.

The [compact evidence packet](../evidence/20261024-native-shell-dpr-v1/README.md)
contains [all measurement summaries](../evidence/20261024-native-shell-dpr-v1/metadata/measurements-final.json),
[the requirement audit](../evidence/20261024-native-shell-dpr-v1/metadata/requirements-final.json),
focused controls, exact task/submission receipts and the small condition bank.
Its [manifest](../evidence/20261024-native-shell-dpr-v1/manifest.json) hashes copied
records and references complete external artifacts. The retained attempt
ledger contains 152 successful tasks, 11 failed attempts (including the
expected FP32 rejection), and six tasks from submissions aborted before Slurm.
All required final campaigns have successful receipts.

The [delivery validation](../evidence/20261024-native-shell-dpr-v1/metadata/delivery-validation.json)
checks all 873 frozen engine inventory files, unchanged production numerical
AST after documentation edits, the identical benchmark/regression source,
and preservation of the twelve original unfinished files.
