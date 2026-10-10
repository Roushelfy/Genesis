# Native stress throughput with copied conditions, 2026-10-10

This study increases the environment batch without introducing additional
randomized contacts. The 1024-condition bank fixes initial joints, egg poses,
radii, friction ratios, all three IK targets and independent episode delays.
Environment i uses bank row i modulo 1024. Every environment still performs
independent rigid contact resolution, complete native stress recovery and reset;
stress observations are not copied between replicas.

The bank SHA256 is
`1d924b60c65d3e092c4c6aac27966da2f2482ca376df04c877253a13bef780b8`.
The physical/numerical configuration remains L1 full hollow shell, 810 P2
nodes / 2430 DOFs, 480 tetrahedra, Q10, FP64, dt=0.01 s, one substep,
default FP64 inverse, cooperative pressure and history zero. Complete residual
and finite-pressure wrench budgets are unchanged. This is batch scaling,
not new randomized-contact coverage or physical convergence.

Discovery uses NVIDIA RTX PRO 6000 Blackwell Server Edition on rtx-210-029,
GPU UUID `GPU-e069af0d-6e02-224f-cbb8-e618485e3ad9`, container genesis:1_26,
eight allocated CPUs and 128 GiB host memory. Runtime artifacts are under
`$RIGID_STRESS_DATA_ROOT/runs/20261010-copy-scaling/`.
Source `ca4854db` adds frozen condition loading/export. From B=32768,
`99dada49` vectorizes only untimed warmup diagnostics; timed code, production
kernels and the benchmark's compiled-kernel source offsets are unchanged.

Each discovery point uses 900 actual warmup steps, then at least 1200 live
trajectory steps and ten seconds, with boundary synchronization. Timed work
includes ordinary controller commands, changing radii, rigid physics, contact
mapping, solve, complete residual, global peak and independent resets. Setup,
JIT, warmup diagnostics and JSON export are outside timing. Final selection
requires three repeats and a separate every-step native validity audit.

## Discovery measurements

| Environments | Environment transitions/s, one repeat |
|---:|---:|
| 1024 | 41,010.50 |
| 2048 | 58,492.41 |
| 4096 | 74,011.11 |
| 8192 | 84,203.02 |
| 16384 | 88,726.20 |
| 32768 | 89,352.91 |

B=65536 is rejected during ordinary rigid build, before stress construction:
the Jacobian shape (2328, 15, 65536) exceeds the engine's maximum of
2^31-1 elements in a single tensor. This is an existing index/array-size
guard, not a pressure mapping failure or GPU allocation failure.
The mathematical bound for this shape is B=61497; the largest complete
1024-condition multiple is B=61440. The guard is retained.

Refinement tests B=24576,49152,61440 on the same host, GPU UUID
`GPU-5c0d0cca-f892-19d7-d4f3-b68ab73a520b`. B=24576 measures 88,222.72
transitions/s in one repeat. Final candidate/baseline comparisons require
matched measurements on the same card rather than assuming card equality.

The previous 42,867 result was on a different GPU allocation. Scaling gains
use the current allocation's 1024 baseline. At B=2048, the two replicas have
identical sampled heights, phases and nonzero-contact counts. Maximum sampled
stress difference is 2.666e-8 Pa and maximum complete residual 1.825e-10 N.
Tiny stress differences reflect native atomic accumulation; stress remains
auxiliary and does not change the rigid trajectory.

The user stopped further peak scanning at 16:06 UTC to prioritize the contact
model repair. Slurm jobs 704413 (refinement) and 704449 (final repeats) were
cancelled. Completed points remain discovery evidence. Interrupted points
have no accepted rate, and the three-repeat final peak selection and validity
audit were not completed. No certified maximum is claimed from this scan.

See [RUNNING.md](RUNNING.md#copied-condition-native-batch-scaling) for commands.
