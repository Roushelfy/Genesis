# CPU cost and exact response-operator controls

These are **single-thread CPU 3D P2 FEM research measurements**, not native
Genesis, GPU, robot, complete RL scene throughput, or a continuous maximum.
The fixed box geometry, bottom clamp, E=10 GPa, nu=0.3 and finite Hertz-profile
prescribed tractions are defined by `../volume/model.py`.

## Matched queries and outputs

The headline run reuses the volume accuracy study's exact **3,822 common
physical probes** and **16 prescribed-load cases**, with the same Frobenius
metric (off-diagonal tensor coefficients weighted twice). All six symmetric
stress components are produced on every probe. B1 is the first case, B16 is
all 16 cases, B64 is four tiled copies; it is not 64 independent conditions.
The source distribution, radius and normal/friction resultant are reintegrated
on the actual selected mesh every update. An independent physical contact
problem is not solved by this performance control.

There are three immutable feature meshes (13,845 / 13,812 / 13,872 DOF), each
with hmin=1.1 mm around one anchor. The joint-refined mesh has 24,432 DOF.
Accuracy versus a continuum or finer reference is **not** inferred from cost.
The atlas and joint mesh are different Galerkin approximations. Their error
comparison belongs to the volume accuracy report.

## Exact offline responses, rather than point snapshots

For each mesh, the cached response contains every force DOF on the complete
top boundary, including all three Cartesian components. It is `S K^-1 B_top`
with the model's actual scaled solve. Current positions/radii are not rounded
to anchor points. Thus the query is not restricted to training contact points.
Selected-column execution drops only exactly zero load rows, with no threshold
or approximate field truncation. Active environment columns are grouped by
selected feature mesh, and contributions are added as **tensors before VM**.

The joint-refined mesh receives the **same dense/selected response optimization**.
This fairness control is necessary: speeding up a generic solve by precomputed
responses is not itself a feature-mesh-bank advantage.

## Measured repeated-step costs

AMD EPYC 9V74, NumPy 2.3.5, SciPy 1.17.0, one BLAS/OMP thread. Five interleaved
repeats reverse method order on alternate repeats after other heavy CPU jobs
finished. Values below are median milliseconds **per complete batch**. Min/max
and every repeat are retained in `results.json`; isolated outliers remain.

| Operation | B1 ms | B16 ms | B64 ms |
|---|---:|---:|---:|
| Atlas current load transfer | 14.539 | 353.118 | 1369.657 |
| Atlas cached LU, recovery only | 9.534 | 94.667 | 309.421 |
| Atlas selected H, recovery only | 7.347 | 97.007 | 128.663 |
| Joint mesh cached LU, recovery only | 20.386 | 146.436 | 566.642 |
| Joint mesh selected H, recovery only | 7.876 | 80.846 | 110.809 |
| Atlas transfer + cached LU | 24.040 | 458.103 | 1729.817 |
| Joint transfer + cached LU | 41.361 | 654.397 | 2655.526 |
| Atlas transfer + dense H | 23.146 | 467.129 | 1527.348 |
| Atlas transfer + selected H | 22.576 | 444.915 | 1482.473 |
| Joint transfer + dense H | 32.969 | 569.167 | 2203.867 |
| Joint transfer + selected H | 27.629 | 579.792 | 2146.898 |

Comparing each approach's best measured **full-tensor** path among LU, dense H
and selected H:

| Batch | Atlas ms | Joint-refined ms | Joint / atlas | Atlas latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 22.576 | 27.629 | 1.224x | 18.29% |
| 16 | 444.915 | 569.167 | 1.279x | 21.83% |
| 64 | 1482.473 | 2146.898 | 1.448x | 30.95% |

The overall gain is principally **cheaper actual pressure transfer/point
location on each smaller feature mesh**. At B64, response-only selected-H
recovery is actually slower for the atlas (128.66 ms) than the joint mesh
(110.81 ms). At B16, dense joint H is faster than selected joint H. These
negative results rule out a universal claim that an atlas or column pruning
always improves matrix multiplication.

`atlas_transfer_plus_response_vm_max` uses the **dense** H path. Compare it with
the dense-H full-field counterpart, not selected-H full fields. No consistent
CPU reduction benefit is inferred from small, overlapping timing differences.

## Setup and memory

Feature-mesh H storage totals **633.47 MB** versus **350.58 MB** for the joint
mesh, in decimal bytes. They are shared immutable libraries across environments,
not per-environment copies. H construction costs total **13.11 s** for the
three atlas models versus **16.51 s** for the joint model. The five compared
models' mesh/assembly/factor preparation totals 7.76 s (`setup_seconds`);
stress-map assembly is an additional offline operation not timed separately.
Per-model assembly/factor costs and sparse storage are retained. This is not a
claim that the entire atlas has lower offline storage or preparation cost.
The process peak RSS was **2.806 GB**, including all comparison models and
libraries, and is not a per-environment or stage-specific memory measure.

## Numerical verification

- Every headline exact response versus cached-LU field/VM comparison passes;
  per-environment Frobenius field errors are below 5.3e-15.
- Top-boundary support excludes only roundoff-level interior load entries;
  the maximum excluded-load norm ratio is below 1e-15 and checked explicitly.
- Force and moment conservation pass unchanged numerical gates.
- Complete unload has no load buckets and returns exactly zero tensors/VM.
- Source hashes before/after match. Runtime input/radius checks and actual
  load generation are included in full-path timings.

The independent `condensation_check.py` executes a separate **exact static
condensation** identity on the 1,701-DOF P2 box: retain all 243 loaded top DOFs,
eliminate 1,215 interior free DOFs, and compare six current finite-pressure
normal/friction loads with the original full solve. Displacement and residual
differences are below 5e-15; symmetry and positive definiteness of the Schur
matrix pass. This retains the full interface and has zero interior RHS under
a bottom clamp. It is not evidence for compressed ports, truncated corrections
or free-body inertia relief with unrepresented body loads.

## Reproduce

From the study package's parent, under one BLAS/OMP thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python atlas_study/performance/benchmark.py --batches 1 16 64 --repeats 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python atlas_study/performance/condensation_check.py
python atlas_study/performance/plot_results.py
```

The earlier `smoke_fixed.json` uses hmin=2 mm, q8, 646 probes and one repeat
under concurrent work; it is an API/numerical smoke only, not headline timing.
The initial smoke serializer failure is retained in `smoke.log`.
