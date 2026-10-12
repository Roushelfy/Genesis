# Exact-sample surface fan launch trials

The isolated candidate reduces empty CUDA sample tasks by compacting candidate
faces in stable face order. Every retained face uses the original squared
coordinates, Gauss rule, true material normal, intrinsic-distance query and
positive profile. Verified one-face disks use the original circular rule.
Physical radii, actual friction, force/moment targets, final 1e-8 wrench
acceptance and complete equilibrium budgets remain unchanged. No contact or
positive sample is dropped. The face test is the original closest-point gate,
used only to prune launches; it does not replace surface topology with a
Euclidean footprint.

This prototype passes all 56 CPU cases in 462.43 s and all 56 CUDA cases in
647.52 s. Matched measurements use RTX PRO 6000 Blackwell Server Edition,
Quadrants 1.3.3, FP64, the complete low shell (480 P2 tetrahedra, 810 nodes,
2430 DOFs), Q10, s8, dt=0.01 and one rigid substep. The same 32-condition
bank, including delays, is copied across environments. Each live measurement
includes 600 warmup steps, three complete 1280-step repetitions, all input
updates and resets, and every environment's failure counter.

| Batch | Original live env-step/s | Candidate live env-step/s | Ratio | Extra frontier bytes |
|---:|---:|---:|---:|---:|
| 32 | 3176.465471 | 3247.358731 | 1.022318 | 82,948 |
| 1024 | 28276.747753 | 39565.723142 | 1.399232 | 2,654,212 |

Within each row, both variants run on the same allocated GPU. All repetitions
have zero invalid environment steps and zero scatter overflow; reset counts
are 37 and 1184 per repetition respectively. B1024 candidate repetitions are
39453.989, 39633.754 and 39609.909 env-step/s. The extra maximum readback and
face-list construction occur inside the measured step. Candidate buffers
are excluded by the old prototype's native enumerator and explicitly added:
167,709,622 total native/candidate bytes at B32 and 3,100,199,510 at B1024.

Snapshot profiling reduces the B1024 intrinsic pipeline from 31.879304 to
19.883841 ms. Its native replay agrees within 1.943e-16 N nodal force and
9.786e-9 Pa global peak; original point difference is zero. Final normalized
force/moment errors are 8.930e-9 and 4.144e-9. Complete CPU equilibrium,
including gauge rows, consumes at most 0.0051643 of its unchanged budget.
These snapshot checks supplement the independent CPU full-field regression;
the profiled snapshot is not a live throughput measurement.

The data, exact candidate/generator/driver, source snapshots before and after,
all raw logs/JSON, complete per-environment counters and publication verifier
are retained in
[`fan-frontier-checkpoint-v1`](../evidence/20261011-nonconvex/fan-frontier-checkpoint-v1/manifest.json).
Its 268 entries reference 107 unique blobs totaling 8,468,124 bytes. Large
logs/JSON are losslessly compressed; original and stored hashes are separate.
Manifest SHA256:
`575fbfdbef4366d8109a6c5c03ae86d4d25d85928fd192616d32c01033021aa6`.

Production integration stores the frontier in typed, enumerated native scratch
and exposes a scheduling ablation. Its own regression and matched timing are
still running; the table qualifies the isolated prototype only. This copied
bank does not establish diverse policy acceptance, a throughput maximum or an
optimization plateau. The original componentwise load law's policy failure
and the explicit source-prior model's production validation remain separate
open work.
