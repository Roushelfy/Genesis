# Native Quadrants low-resolution performance, 2026-10-10

This milestone integrates auxiliary stress observation into the existing
rigid solver. All production numerical work uses owned Quadrants kernels;
there is no external stress numerical backend. The normal Panda example
imports no research implementation. High-resolution physical convergence
was not run after the direction change. The low-resolution stresses below
are numerical/performance results, not physically converged egg stresses.

## Configuration and measurement contract

- NVIDIA RTX PRO 6000 Blackwell Server Edition, 101,975,851,008 bytes VRAM;
  Xeon Platinum 8562Y+ host. Allocated `rtx-mid`, container `genesis:1_26`.
- Python 3.10.12, Genesis 1.4.3, Quadrants 1.3.3 (00423caa), Torch 2.8.0.
- Complete hollow affine-P2 shell: level 1, 810 nodes, 2,430 DOFs,
  480 tetrahedra, 80 exterior faces. E=10 GPa, nu=0.3, rho=2,000 kg/m3.
  Explicit hollow rigid URDF mass 0.006529848210401324 kg, COM/inertia.
- FP64 scene/operators/loads/residual/peak. Positive Q=10x10 Duffy/Gauss
  samples per face (8,000 total), unchanged finite pad law, radius baseline
  6 mm. Varied environments use independent seeds 510000+i, initial pose,
  friction, radius and episode delay; radius also changes continuously.
- dt=0.01 s, one stress recovery per physical substep, one substep per step.
  Elliptic friction, 100 rigid constraint iterations, tolerance 1e-12,
  contact pruning/hibernation/rolling/torsional friction disabled.
- Every full rate uses at least 1,200 steps and 10 s per repeat; final
  measurements use three repeats. The aggregate is total counted environment
  transitions / total wall time, with synchronization at timing boundaries.
  Controller, ordinary rigid simulation, radius updates, independent resets
  and device policy/observations are included in their stated scopes.
- Warmup diagnostics (900 steps), setup/JIT, profiling and trace export are
  outside steady-state timing. Setup timings include JIT and vary with caches.
  Recovery-only repeats a frozen actual snapshot and matching pre-integration
  frame; live measurements contain changing contacts and whole episodes.

## Implementation and numerical evidence

Link opt-in, device maximum observation, substep ordering, error handling and
partial-reset invalidation use the existing solver lifecycle. Disabled stress
retains the original fused rigid step. Stress-enabled contact force and pose
are sampled after constraint resolution and before rigid integration. All
sorted contacts on either force side are transformed into the authored frame.
No symmetry, fixed contact-response basis or prescribed hotspot is used.

The engine constructs P2 stiffness, consistent mass, rigid modes, six gauge
pins, relief Gram/inverse and six centrifugal fields natively. A shared sparse
block LDL factor follows a Python symbolic minimum-degree order. The small
mesh also admits a 47,239,200-byte full pinned inverse, built by native unit
loads and native solves; auto uses it only within a 64-MiB storage budget.
Larger meshes automatically retain the sparse factor and avoid quadratic
inverse storage. No large-mesh throughput is claimed.

The independent SciPy/Numba oracle uses the identical mesh and six pins.
Operator relative differences are below 2e-13. Random asymmetric loads,
centrifugal terms, zero/freefall cases, finite frictional patches, constrained
pressure, invalid inputs, actual rotated contacts, one/four substeps,
selected setters, reset independence and disabled-feature trajectory equality
are covered by native tests. Test logs publish complete residuals, displacement
errors and global peak errors. In the GPU FP64 checks, random-load relative
displacement errors are of order 1e-11 and peak errors below 1e-6 Pa;
actual rotated support-contact peak errors are below 2e-6 Pa.

Every runtime solve checks the complete residual, including the six gauge
rows, against max(1e-11 N, 1e-8 ||rhs||) in the benchmark. The isolated random
oracle test uses 1e-7 relative and verifies a 1e-4 peak budget separately.
FP32 inverse storage with two corrections failed arbitrary loads before
fallback was added; the retained masked native sparse fallback restores
acceptance. Invalid observations are NaN and raise through rigid errno.

At B=2048 a genuine contact at tick 632, environment 1996, slot 1 is rejected
by both native and independent CPU finite-pressure mapping. CPU force/moment
errors are 1.82e-8 N and 6.00e-8 Nm, above the declared wrench budget. Its
6-mm footprint is not enlarged to obtain a rate. B=1024 is the largest
successfully tested complete workload; B=2048 has **no valid throughput**.
This is a finite-footprint/sampling limitation, not a GPU memory limit.

## Measured optimization sequence

At B=8, isolated wall milliseconds identify the original bottlenecks:

| Configuration | Linear solve | Pressure | Scatter |
|---|---:|---:|---:|
| Native serial LDL + full quadrature scan | 68.641 | 25.908 | 14.421 |
| Cooperative warp LDL + full scan | 5.386 | 25.9 | 14.39 |
| Cooperative LDL + exact 16^3 grid | 5.459 | 1.347 | 0.770 |
| Shared FP64 inverse + exact grid | 0.276 | 1.497 | 0.783 |

The first three profiles used the same nominal trajectory but re-associated
the post-integration pose. Subsequent profiling explicitly restores the
matching cached contact-solve frame. The early tables diagnose stage changes;
they are not an identical-input final ablation. Full live comparisons use
the correct integrated substep ordering in all variants.

The valid normalized serial baseline, commit 65c2f25f plus the archived
normalization/seed patch, runs B=8 varied live at 57.430--57.517 transitions/s
over three repeats. FP64 inverse iteration v2 runs the same full workload at
894.807--902.274 transitions/s, aggregate 899.492: **15.66x** improvement.
This comparison precedes the last scheduling/memory changes and is labeled
as an iteration checkpoint, not the final revision's rate.

Optimization choices preserve the load model, mesh, tolerance and sampling:

- Cache the authored frame once per environment; select force sides and
  transform contacts in the native association kernel. Device getters alias
  existing buffers; no host contact extraction occurs in production.
- Exact spatial grid bounds reduce 8,000 samples to a mean 281 visited and
  135 inside per valid slot in the B=1024 snapshot. No eligible sample is
  dropped. Anchor search still visits all 80 faces to preserve arbitrary
  force-line intersections; its measured cost is only about 0.19 ms.
- Normalize pressure objective/Gram by integrated pad area. This removes a
  real roundoff stall in an actual sliding contact without changing pressure.
  An analytic nonnegative-profile bound accepts the common closed form;
  constrained Newton/line search remains the correctness fallback.
- A CUDA warp cooperates on each sparse triangular solve. The memory-bounded
  full inverse is faster at this low mesh and shared across all environments.
  B=1024 same-source comparison: inverse 33,602 transitions/s versus sparse
  cooperative direct 27,776; both use strict complete residuals.
- FP32 inverse storage halves its shared bytes but triggers frequent
  corrections/fallbacks. B=1024 measured 11,267 transitions/s and 91,011
  fallback observations among 90x1024 warmup snapshots. FP64 is selected.
- Mask correction, residual recheck and sparse fallback on device, retaining
  accepted environments' checked norms. No failed-count host synchronization,
  packing or dynamic external RHS interface is required.
- Direct methods allocate no Krylov node buffers. At B=1024 this removes
  59,719,680 unused bytes. Zero-length placeholders are omitted from exposed
  data, because Quadrants 1.3.3 cannot export those via CUDA DLPack.
- Split pressure's common integration/closed form from constrained correction
  in one captured native graph. The combined split/memory iteration measured
  34,458 transitions/s versus 33,602 on the preceding checkpoint (single
  repeats, 2.5%); isolated pressure GPU cost was unchanged, so this is not
  claimed as a pressure-arithmetic speedup.

## Detailed current profile and candidate decisions

B=1024, varied, matching frozen actual substep snapshot, no temporal history:

| Stage | Wall ms |
|---|---:|
| Begin step | 0.050 |
| Association/material frame | 0.177 |
| Force-line anchor search | 0.193 |
| Candidate weights/pressure constraints | 7.492 |
| Nodal scatter/zero load | 2.804 |
| Inertia relief/centrifugal | 0.199 |
| RHS reduction/direct initialization | 0.041 |
| Full inverse application | 7.375 |
| Complete residual | 0.347 |
| Global unaveraged peak | 1.115 |
| Acceptance | 0.027 |

These isolated sequential wall times include dispatch and captured graphs;
they do not replace end-to-end timing. Masked correction/fallback dispatches
are included in live rates and traced, but omitted from this initial-solve
stage table. There were no FP64 inverse warmup fallbacks in the measured sweep.

Pressure subdivisions repeat work and are not additive to that table:
grid/distance query 1.227 ms; query plus weighted Gram 5.774 ms; local closed
form 0.052 ms. The mean constrained evaluation count is 0.0625 across 3,808
valid slots; maximum 3. Thus sample traversal/weight evaluation and divergent
contact work dominate small matrix inversion. Scatter atomics remain material.
Caching all candidate weights at the padded scene capacity would reserve a
large contact x sample buffer despite only 3.72 valid contacts/environment;
no unbounded candidate cache or truncation is introduced.

A separate intrusive 50-step CUPTI trace on cdf0bc17 records 21,398 GPU kernels,
5,196 copies totalling 1,077,406 bytes, 3,296 stream synchronizations and one
device synchronization. Kernel interval union is 1,222 ms of 2,058 ms trace
wall time (59.4%). This includes ordinary rigid/controller work and Quadrants
metadata copies; it does not attribute all overhead to stress or establish
GPU utilization during unprofiled timing. Native graph kernels are omitted
by the Quadrants kernel profiler; trace and wall stages expose them.

The resident-batch sweep (single-repeat development points) is 171, 2,770,
8,411, 21,302 and 33,176 transitions/s at B=1,32,128,512,1024. This motivates
B=1024. RHS storage is already environment-contiguous and resident. Native
masked kernels consume all environments directly; chunking/packing would add
dispatches without reducing the shared inverse or factor. No chunked rate
is claimed. FP32/Tensor Core accumulation was not substituted for the strict
FP64 numerical path: measured mixed storage already loses to correction.
Simple scalar/block PCG reached 4,000 iterations with complete residual
5.1e-6--1.7e-5 N and failed acceptance; stronger large-mesh multigrid is a
separate workload from this 2,430-DOF small inverse and is not claimed solved.

Temporal reuse was investigated on 4,800 actual second-episode RHS columns:
ideal h=0/1/4 projection passed 22/64/1,909 columns at the runtime force budget.
Native stable four-column QR alone costs 1.337 ms at B=1024, motivating an
optional paired displacement/load history with two-pass reorthogonalization,
cached basis, full residual rejection, masked solve and independent reset.
Only corrected accepted environments append history. Its actual live
comparison and final selected configuration are reported below.

The full-domain peak is already fused, with no stress field materialization.
Its 1.1-ms cost is smaller than pressure and inverse application. A fixed
hotspot or an unproved hierarchical exclusion would compromise the required
global observation and was not used. Load relief uses six precomputed fields
and fused native passes; the approximately 0.2-ms stage is not the bottleneck.

## Final measurements

Final repeated rates, history selection, quality and source revision are
filled from the completed raw artifacts. See `../evidence/20261010-native/`
for logs, compressed original JSON, hashes and literal commands; full traces
and large diagnostic RHS arrays remain under the explicit runtime data root.

The API/support boundaries are in [NATIVE_API.md](NATIVE_API.md) and exact
entrypoints in [RUNNING.md](RUNNING.md). Remaining performance limits are the
finite-pressure integration/scatter, selected linear/prediction path and
ordinary rigid/host scheduling. Physical convergence remains a later study.
