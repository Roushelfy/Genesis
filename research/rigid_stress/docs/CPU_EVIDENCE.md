# Previous CPU evidence and limits

The machine for the latest batch/history and precision studies was an Intel
Xeon Platinum 8573C, BLAS single-thread. Numerical reference: **full** hollow
egg, straight P2 tets, 9,630 DOFs / 1,920 tets, current LU nnz 7,331,676.
Material fixed; contacts move asymmetrically with normal+tangential loads.
The mesh is deliberately coarse and **not physically stress-converged**.

Load sequences are prescribed friction-cone-admissible snapshots, not a
Genesis/Coulomb grasp solve or a robot policy rollout. Environments have
independent histories, different phases/amplitudes of one moving-contact
sequence. There is no fixed contact basis, physical symmetry or fixed hotspot
in the latest methods. The old fixed-basis temporal example imported as a
dependency is explicitly nonuniversal and cannot be used as the new baseline.

## Batch/compaction/history

Complete recovery timing starts at already assembled RHS and includes
prediction, fresh residuals, pack/solve/scatter, history update and full-domain
peak. It excludes contact mapping, rigid dynamics, policies and independent
reference checks. Main paths FP64, h=8, relative equation tolerance 1e-3.
Two repeats reversed method order; full reports/raw JSON are committed.

| Method | B=8 ms/env-step | B=32 ms/env-step |
|---|---:|---:|
| Serial direct factor solves | 9.7722 | 8.3992 |
| Shared-factor batch direct | 5.8617 | 5.3069 |
| Legacy temporal, each env separately | 7.9274 | 6.6113 |
| Batch temporal, full zero-padded failed RHS | 8.9104 | 7.1630 |
| Batch temporal, compact failed RHS | 5.7731 | 3.5974 |
| Batch temporal, compact + cached history | **4.5099** | **3.0143** |

The last path improves the old serial temporal implementation by 1.76x/2.19x.
Peak errors vs direct are max 0.12345% at B=8 and 0.110122% at B=32; this is an
approximation profile, not equal precision to full direct solves. It does not
meet the proposed strict 0.01% profile. B=8 ran 512 steps/environment spanning
a whole synthetic grasp; B=32 ran 128 continuous steps/environment with phases
spanning stages across the pool, not one complete grasp for every env.

Failure counts: B=8, 1,089/3,680 nonzero steps; B=32, 813/3,545. The cached and
legacy paths use the same failed states. Shared batch microbenchmarks achieved
roughly 1.4–1.7x at B=8–128; at 25% failures, FP64 compaction achieved 4.33x
(B=32) / 5.74x (B=128), with pack cost 0.0172/0.1841 ms. Cached Gram/FIFO
history improved isolated projection by 1.73–2.09x and updates by 1.58–1.94x.
These are isolated CPU stage measurements, not independent multiplicative
speedups for a GPU whole pipeline.

Counterexamples are important to implement automatic fallback:

- Abrupt/all-failed control: direct batch 9.8383 vs cached temporal 11.4942
  ms/env-step. Temporal history loses.
- Strict residual 1e-6 control: 89.49% failed; direct batch 5.6824 vs cached
  temporal 11.6808 ms/env-step. The optimized peak error there was about
  3.36e-5%, but paying for history did not buy throughput.

See `evidence/batch_history_report.txt`, `batch_history_results.json`,
`batch_history_summary.json` and `batch_history_table.csv`.

## FP32 evidence

Offline geometry/K/M/contact mapping remain FP64. Runtime tests cast complete
loads, do inertia relief, LU recovery and full corner peak in FP32. This is
not FP32 assembly or live GPU contact mapping. The study contains 624 frames,
570 with reference peak >1 Pa.

| Runtime path | Max relative peak error | p95 peak error |
|---|---:|---:|
| FP32 | 0.5923% | 0.1398% |
| Diagonally scaled FP32 | 0.3775% | 0.1592% |
| FP32 factor + one FP64-residual refinement | 0.005011% | 0.000618% |

Pure FP32 had full relative residual as high as 1.26%; about 7.5% of active
frames exceeded 0.1% peak error, none exceeded 1% in this dataset. A nominal
zero state produced a spurious peak up to about 0.007 Pa. These measured peak
errors do not authorize FP32 to skip residual checking in arbitrary future
contacts. Scaling did not improve every error statistic.

Matched CPU isolated solve medians were roughly FP64 7.00 ms, FP32 4.86 ms,
and one high-precision residual correction 10.70 ms. Previous CPU batch-32
precision probes did not always show FP32 speed gains. Measure actual GPU
paths and fallback frequency. There is no real hardware TF32 result.

See `evidence/fp32_precision_results.json` and `fp32_precision_report.txt`.

## Convergence and imported historical restrictions

The previous finest convergence work used quarter-model physical symmetry
conditions and prescribed symmetric finite pads. Its reported peak near
7.63–7.70 MPa cannot be transferred to arbitrary Franka loads. It is retained
only to document why P2/finite contact area helped the earlier convergence
issue. New acceptance requires full-shell meshes and asymmetric friction.

The latest general peak kernel evaluates all corners of every element. For
affine P2 geometry this is an exact discrete maximum; it does not establish
the continuous mechanics peak. Synthetic/traction/full-body evidence is in
the other committed JSON and report files. No CPU numbers above include real
Genesis or RL throughput; no current GPU throughput is measured.

## New handoff checks

`evidence/seed_cpu_validation.json` records checks run after copying sources:
four independent envs, compact correction/reset handling, force-preserving
arbitrary patch loads, centrifugal inertia relief, full residual/gauge
invariance, P2 peak vs full stress, optional Torch dense-batch/peak/compaction
on CPU. The point-to-finite-patch moment discrepancy is explicitly reported,
not passed as wrench conservation. See `seed_environment.json` for installed
versions. This check still does not execute `franka_egg.py` or CUDA.
