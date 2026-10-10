# Native Quadrants low-resolution performance, 2026-10-10

> Historical milestone: the original tables and coefficient-1 workload below
> precede contact repair and the current optimization round. The repaired
> Q10 local-integration model, actual coefficient variation, later numerical
> checks and selection status are documented in
> [CONTACT_REPAIR_20261010.md](CONTACT_REPAIR_20261010.md) and
> [OPTIMIZATION_LEDGER_20261010.md](OPTIMIZATION_LEDGER_20261010.md).
> Final acceptance remains open. In particular, the old 2c6e5ecc B=49152
> policy trajectory rejects an unresolved contact fit and has no valid rate.

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
  Egg friction ratios vary over [0.7, 1.4], but the collider combines
  coefficients by maximum: the other bodies' coefficient 1 dominates the
  egg's 0.6 times ratio. Actual sampled contact coefficients are therefore
  1.0 in this workload; the finite-pressure unit tests vary coefficients.
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
partial-reset invalidation use the existing solver lifecycle. Checkpoint
export includes configuration and auxiliary arrays, but restoring rigid
state invalidates stress peaks, displacement and history through ordinary
geometry/dynamics notices; the next physical step recomputes them. Disabled stress
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
The final selected source (`dbfdf20d`) passes all 16 GPU cases in 162.77 s
and all 14 CPU numerical cases in 73.07 s. GPU tests include both actual
one/four-substep lifecycle cases. FP64 random-load peak error is at most
3.94e-7 Pa and relative displacement error at most 4.14e-11. Corrected
mixed-inverse random tests use two corrections without sparse fallback,
maximum complete residual 4.93e-9 N and peak error 1.53e-6 Pa; disabling
corrections exercises the checked sparse fallback on all four nonzero loads.

Every runtime solve checks the complete residual, including the six gauge
rows, against max(1e-11 N, 1e-8 ||rhs||) in the benchmark. The isolated random
oracle test uses 1e-7 relative and verifies a 1e-4 peak budget separately.
An early mixed-precision correction-sign bug made corrections diverge; the
masked native sparse fallback preserved final acceptance. The bug was found
in the final code review and corrected, with separate numerical/throughput
reruns. Invalid observations are NaN and raise through rigid errno.

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
- FP32 inverse storage halves its shared bytes. The early implementation
  measured 11,267 transitions/s and 91,011 fallback observations among
  90x1024 warmup snapshots, but its correction sign was wrong. That result
  is retained as a debugging checkpoint and does not select precision;
  the corrected same-workload measurement is reported below.
  With the corrected sign, B=1024 reaches 12,783 transitions/s before warp
  pressure, with 68,468 fallback snapshots and 181,232 corrections among
  92,160 sampled environment observations. At the stricter live 1e-8
  force tolerance, mixed storage remains slower than FP64; FP64 is selected.
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
- Pack only active contact/environment pairs with a native atomic counter,
  without reading the count on the host. One CUDA warp integrates each
  contact's Gram and closed form. A second warp kernel evaluates constrained
  pressure, with one shared evaluator body for gradient and line search.
  This avoids duplicated sample logic and long serial contact tails, retaining
  the same Newton/Armijo algorithm, candidates, law and accuracy. Warp Gram
  alone measured 35,740 transitions/s; cooperating on constrained evaluations
  measured 42,986 in the selection probe. The final three-repeat comparison
  against corrected serial pressure is 42,867 versus 34,047 (+25.9%).
  The device pair buffer costs 6,144,004 extra bytes at B=1024. CPU and the
  explicit serial option retain the verified scalar kernels.

## Detailed current profile and candidate decisions

B=1024, varied, matching frozen actual substep snapshot, no temporal history:

| Stage | Wall ms |
|---|---:|
| Begin step | 0.053 |
| Association/material frame | 0.176 |
| Force-line anchor search | 0.192 |
| Candidate weights/pressure constraints | 1.383 |
| Nodal scatter/zero load | 2.857 |
| Inertia relief/centrifugal | 0.199 |
| RHS reduction/direct initialization | 0.041 |
| Full inverse application | 7.291 |
| Complete residual | 0.343 |
| Global unaveraged peak | 1.119 |
| Acceptance | 0.027 |

These isolated sequential wall times include dispatch and captured graphs;
they do not replace end-to-end timing. Masked correction/fallback dispatches
are included in live rates and traced, but omitted from this initial-solve
stage table. There were no FP64 inverse warmup fallbacks in the measured sweep.

Pressure subdivisions use serial diagnostic kernels, repeat work and are
not additive to the cooperative production table:
grid/distance query 1.223 ms; query plus weighted Gram 5.751 ms; local closed
form 0.052 ms. The mean constrained evaluation count is 0.0625 across 3,808
valid slots; maximum 3. The final native kernel profiler measures 0.457 ms
for warp Gram/closed form and 0.830 ms for warp constrained evaluations.
Before cooperation, pressure cost 7.49 ms; afterwards inverse application
and scatter dominate auxiliary recovery. Scatter atomics remain material.
Caching all candidate weights at the padded scene capacity would reserve a
large contact x sample buffer despite only 3.72 valid contacts/environment;
no unbounded candidate cache or truncation is introduced.

A separate intrusive 50-step CUPTI trace on dbfdf20d records 21,648 GPU kernels,
5,296 copies totalling 1,094,206 bytes, 3,296 stream synchronizations and one
device synchronization. Kernel interval union is 895 ms of 1,671 ms trace
wall time (53.6%). This includes ordinary rigid/controller work and Quadrants
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
Only corrected accepted environments append history. Same-source live h=0
runs at 34,487 transitions/s and h=4 at 31,471 (single-repeat selection probe),
despite 48,021/92,160 warmup snapshot hits (52.1%). Native persistent storage
grows from 268,673,433 to 587,237,785 bytes. Prediction, QR/basis upkeep,
additional residual and masked inverse tail outweigh saved solves. **h=0 is
selected**. The tested optional h=4 path remains explicit for reproducible
comparison and repeated-load workloads; no temporal speedup is claimed here.

The full-domain peak is already fused, with no stress field materialization.
Its 1.1-ms cost is smaller than inverse application and scatter. A fixed
hotspot or an unproved hierarchical exclusion would compromise the required
global observation and was not used. Load relief uses six precomputed fields
and fused native passes; the approximately 0.2-ms stage is not the bottleneck.

## Final measurements

Selected configuration: B=1024, level 1, Q10, FP64 shared inverse, cooperative
solve/pressure and h=0. Selected engine/example/benchmark/test source is
`dbfdf20d`. Disabled rigid scopes were measured on `6a048825`; their original
fused path and workload are unchanged by the subsequent stress fixes.

| Scope | Aggregate environment transitions/s | Three-repeat range | Steps/repeat |
|---|---:|---:|---:|
| Rigid only | 108,231 | 107,690–108,627 | 1,200 |
| Policy + rigid | 105,571 | 105,498–105,612 | 1,200 |
| Frozen-input auxiliary recovery | 75,845 | 75,570–76,032 | 1,200 |
| Live rigid + auxiliary stress | **42,867** | 42,843–42,887 | 1,200 |
| Policy + live rigid + auxiliary stress | **42,515** | 42,399–42,589 | 1,200 |

Each live/policy repeat includes 1,024 independent resets; each frozen-input
repeat includes 513 reset calls. Frozen-input recovery is a stage capacity
measurement, not changing-contact trajectory throughput. Live batch steps/s
is 41.86; policy batch steps/s is 41.52. Policy is seeded FP64 inference
(26→128 tanh→128 tanh→7 linear), with device observations including stress
and 1e-4 rad residual actions. No training throughput is claimed.

The final B=8 live aggregate is 1,200.376 transitions/s (1,196.747–1,203.851),
with 1,496–1,505 steps/repeat to meet ten seconds. The matched normalized
serial baseline is 57.464: **20.89x** final improvement. Pressure cooperation
also improves final frozen recovery from 50,194 to 75,845 (+51.1%) and
policy from 33,506 to 42,515 (+26.9%) against corrected serial-pressure runs.

Live and policy warmup each satisfy the stated bilateral lift/hold criterion
for 940/1,024 environments: at least five sampled frames in phase [0.5,0.65],
height >0.12 m and nonzero contacts on both fingers. Remaining environments
are retained in throughput, including slip/release. Nonzero egg contacts range
0–13 (live) and 0–12 (policy). Summed per-environment tangential force reaches
0.871/0.873 N; actual contact radii span 4.083–8.279 mm. Maximum sampled
complete residual is 1.693e-10/1.738e-10 N, with zero FP64 corrections and
fallbacks in both warmups. Peak values are low-mesh numerical observations,
not a physical accuracy claim. A separate every-step native audit across
900 warmup plus 1,200 reset-trajectory steps for both scopes records **zero
invalid observations in 4,300,800 environment steps**, including immediately
before independent resets. Its overhead is outside throughput measurements.

The selected stress allocation ledger has 82 arrays totalling **274,817,437
bytes**, including the shared 47,239,200-byte inverse. Direct methods omit
unused Krylov arrays. Torch peak allocation is 2,527,744 bytes for live and
14,028,800 for policy; this excludes Quadrants/rigid/context allocations.
A 200-ms whole-card NVML sampler on the preceding `6a048825` iteration
observed a 3,551-MiB peak across 2,129 samples. This is a sampled earlier
whole-card footprint, not the final source's exact high-water mark; the
selected warp pair buffer adds the ledgered 6,144,004 bytes. Free-device
snapshots and per-buffer shape/dtype/bytes are preserved in every final JSON.
The selected profile reports setup 21.93 s, including topology 0.097 s,
operators 0.656 s, factor 13.477 s and inverse 0.175 s; these include JIT
and are outside steady-state timing, not cache-independent build costs.

See `../evidence/20261010-native/` for complete logs, compressed original
JSON, SHA256 hashes, asset hashes and literal commands. Large traces and
diagnostic RHS arrays remain under the explicit runtime data root. Some
development measurements used working patches, identified in the manifest;
they are not presented as immutable final-source comparisons. The earlier
checkpoint-test failure assumed restored peaks should remain valid; the
lifecycle contract invalidates them. Corrected final tests pass in full.

The API/support boundaries are in [NATIVE_API.md](NATIVE_API.md) and exact
entrypoints in [RUNNING.md](RUNNING.md). Remaining performance limits are the
shared inverse application, nodal scatter and
ordinary rigid/host scheduling. Physical convergence remains a later study.
