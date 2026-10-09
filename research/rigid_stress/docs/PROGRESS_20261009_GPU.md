# GPU acceptance checkpoint, 2026-10-09

This is an implementation/acceptance checkpoint. GOAL.md remains active: fine-grid performance sweeps,
the frozen held-out configuration and the final policy/live Pareto report are still being measured.
All numerical experiments ran through `gs-srun` on `rtx-mid`, never on the login node.

## Implemented and device tested

The production CPU math now owns full P2 mesh construction, bounded-memory assembly, consistent mass,
inertia relief, rank-safe independent histories and the exact all-corner scalar maximum. Historical
reference scripts remain unchanged and are imported only by explicit oracle checks.

GPU recovery supports one exported sparse factor pair with cuSPARSE SpSM or one native cuDSS SPD factor.
Public cuDSS 0.8 C APIs factor the unchanged reduced stiffness on device with int64 indices. Dense RHS
descriptors for different batch sizes share that factor; B=1 analysis followed by B=3/8/17 solves passes.
CPU native CHOLMOD and GPU native cuDSS avoid exporting the very large finest factor.

Temporal recovery includes variable-step prediction, cached Q/KQ/Gram, rank truncation, double
orthogonalization, circular slots, selected resets, compact/padded/adaptive correction, chunks,
two history layouts and FP32/scaled refinement with explicit FP64 recomputation on failure.
All acceptance decisions use the complete FP64 residual including gauge rows. No per-env factor exists.

Geometry-only quadrature grids and complete scans agree with CPU pressure integration. The declared
force-line exterior anchor preserves the original source moment; pressure remains nonnegative and
admissibility uses the unchanged supplied pad normal/friction. See [PAD_LAW.md](PAD_LAW.md).

Public pre/post substep callbacks pair the contact solve with its pre-integration authored frame.
CPU and GPU callback ordering tests pass for B=0/2 and one/three substeps. Plane/egg stress tests at
one/four substeps pass full-field CPU oracle checks; sampling only the last substep misses a 385,592 Pa peak.

Direct exported-factor CUDA Graph capture works and passes changing-RHS checks, maximum peak error
7.03e-14. Temporal capture is unavailable in the installed CuPy path: its cuBLAS layout conversion
rejects capture. Raw exceptions are retained. This does not establish capture of the entire Genesis scene.

## Physical refinement

The force-line law was held fixed over full levels 4/5/6, two wall layers and positive surface quadrature
orders 6/10/16. The suite contains 12 asymmetric frictional loads, including the declared minimum
4.08 mm radius, and six complete actual nominal phase wrenches with gravity/centrifugal inertia.

| Check | Largest final change | Proposed limit |
|---|---:|---:|
| Full surface refinement L5 to L6, 18 cases | 1.3646% | 2% |
| Finest quadrature 10 to 16 | 8.42e-9 relative | 2% |
| Independent wall layers 2 to 4 at L4, 18 cases | 1.0625% | 2% |

The finest mesh has 2,457,630 DOFs and 491,520 affine tetrahedra. These are empirical representative
checks, not a universal physical error certificate. Wall and surface checks are independently reported.
Same-mesh algebraic error is separate from these changes.

Native finest CPU preprocessing took about 626 s in the raw-anchor run. Exporting that CHOLMOD factor
raised MemoryError, while native CPU solves and native GPU factorization succeeded. Earlier SuperLU
attempts also raised MemoryError. The exact library limitation is unproven; do not infer unavailable hardware.
Native finest GPU B=17 recovery passes CPU FP64: full relative residual 2.13e-12 and peak error 3.12e-13.

## Actual rigid trajectories

The reusable scene uses FP64 rigid arithmetic, convex elliptic contacts, 100 iterations, tolerance 1e-12,
dt=0.01 s, no pruning/hibernation/spin/rolling friction and the full undeformed exterior collision mesh.
Each seed varies offsets, orientations, friction and radius. Independent episode delays and resets are retained.

| Trajectory | CPU peak error maximum | Moment error | Grasp success |
|---|---:|---:|---:|
| Nominal, 600 steps, direct | 4.44e-13 relative | 7.58e-18 N m | 1/1 |
| 32 varied seeds, 1,200 steps, h=4 | 1.52e-6 relative | 4.17e-17 N m | 5/32 |
| Four environments, partial reset at step 107 | 6.19e-7 relative | 1.08e-17 N m | 1/4 |

These rows use the coarse L2 mesh and are equation/contact/reset tests. Five successes out of 32 is
a weak scripted grasp policy, and failures remain in the evidence. Fine-grid held-out trajectories are
being recorded separately. Relative residuals at numerically zero RHS can exceed rtol while satisfying
the explicit 1e-11 N absolute criterion; benchmark outputs now report this floor separately.

Newton-Euler imbalances in the 32-env run reach 5.84e-4 N and 1.44e-5 N m. In the four-env diagnostic,
force and moment balances agree with the rigid constraint gradient to 9.69e-16 N and 5.06e-17 N m.
Thus checked finite physical-solver residuals are exposed, rather than hidden by auxiliary inertia relief.
This is distinct from same-RHS stress-solve precision.

## Timing boundaries and current evidence

Hardware is NVIDIA RTX PRO 6000 Blackwell Server Edition, about 95 GiB, driver API 13020 and CuPy
CUDA runtime 12090. It is not RTX 6000 Ada. Raw JSON contains package versions and configuration.

Three ten-second coarse B=8 live direct repeats give 807/827/826 aggregate environment transitions/s.
They include controls, rigid motion, complete contact mapping, body/inertia loads, full scalar peak,
residual checks and asynchronous resets. They are preliminary: warmup was only 30 steps and the mesh
is coarse. They must not be presented as the accepted physically refined steady-state result.

The new replay benchmark warms all 1,200 recorded frames and verifies the identical assembled RHS
against CPU FP64 outside timing. Its coarse h=4 B=8 repeats give 1,094/1,109/1,109 recovery transitions/s,
including mapping and resets. About 99% of transitions still require correction. It is a stress-replay
scope, not simulator throughput, and motivates measuring direct recovery before selecting history.

Current CLIs are `benchmark_live` for rigid/live/policy loops and `benchmark_replay` for complete
stress replay plus separate assembled-RHS ablations. They enforce at least ten seconds/three repeats,
complete-grasp warmup, history resets, synchronized GPU boundaries and wall totals. Sampled CUDA
device-wide memory includes native/context allocations; pool counters alone are not total VRAM.
The memory sampling interval is reported and short spikes can be missed.

## Artifacts and ongoing work

Small raw numerical/convergence CSV/JSON and trajectory summaries are in
[`evidence/20261009-gpu`](../evidence/20261009-gpu). Its manifest gives hashes and explicit data-side
paths for complete trajectory JSON/JSONL/NPZ/logs. The initial checkpoint also contains a real scene video.

Remaining work under GOAL.md: measure finest-grid batching/chunks/history/precision/layout/adaptive/peak/graph
ablations, freeze the fastest validated strict and throughput configurations on calibration seeds,
verify their complete held-out RHS against CPU FP64, measure all four scopes plus matched policy baseline,
record fine-grid nominal visualization, and push the final report and measured Pareto point.
