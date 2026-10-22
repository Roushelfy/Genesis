# Continuous Coulomb control and native history layout

A new bounded radial-memory model removes the observed surface Coulomb
violation in changing-load/vector controls. Its Quadrants implementation costs
0.808ms per B1024 update with16 intervals, versus4.411ms for the same equations
using serial solves on the same GPU. This history component timing includes
generated inputs but neither stress recovery nor a moving rigid scene. General
vector-friction physical accuracy and estimator integration remain open.

## Model and controls

The old radial-line clipping model has sampled surface traction ratio1.213108
in the retained variable-load/vector path. Its four source files remain unchanged.
Line clipping does not generally bound inverse-Abel surface traction when load
and direction change. The new line is piecewise linear in s=r², zero at the
edge, with each negative slope vector constrained to the unit disk. Its inverse
Abel transform is a positive weighted integral, giving a sufficient continuous
Coulomb bound. Weighted closest-line minimization retains exact supplied force.
Physical-radius remapping and frame rotation transport the prior. This declared
general friction approximation is not exact arbitrary-vector contact physics.

CPU controls retain16 constant-load collinear analytical sample comparisons at
16/32 intervals and582 variable-load/vector updates at16/32/64 intervals.
Original5% tensor/peak,1e-9 force and1+1e-10 slope gates pass. The sampled
surface ratio is at most1.0000000000000266. Frame, release and reset checks pass;
independent variable-vector physical qualification remains false.

Three CVXOPT attempts retain unknown/scaling failures; initial SLSQP retains its
directional-derivative failure. Final CPU acceptance uses independent nonnegative
dual KKT and strong-convexity distance certificates plus unchanged force/cone
gates. Raw optimizer flags remain unchanged: the last64-interval case reports
iteration limit1000 with a separately certified line-distance bound1.64e-8.

## Native behavior and measured cost

All candidate native numerical work is Quadrants. ADMM uses a shared radial
tridiagonal system and exact two-component force correction. Sums of feasible
increments produce bounded line values. Force/dual/line thresholds remain
1e-9/1e-10/1e-9. Failure clears outputs and latches lost history until release
or reset. Live association, contact identity and frame transport remain caller
responsibilities pending solver integration.

CPU/GPU scalar checks cover analytical projections, frame covariance, isolation,
partial reset, release, gross sliding/reversal, rejection and lost-history recovery.
Independent97-step scalar/vector/frame paths compare with CPU variational results
at16/32 intervals. Final maximum line and surface traction differences are
4.128e-6 and3.093e-5 of mu*p0, inside original1e-5/1e-4 targets.

Reducing penalty from1 to0.1 preserves equations/gates and reduces iterations.
One warp per region then keeps interval values in registers and uses subgroup
collectives. A shared16x16/32x32 inverse of the same radial system replaces serial
Thomas solves;2/8KiB is numerical workspace, not a contact stress basis. CPU
and larger intervals retain serial execution.

| Intervals | Batch | Serial rho0.1 (ms) | Cooperative rho0.1 (ms) |
| ---: | ---: | ---: | ---: |
| 16 | 1 | 2.892 | 0.391 |
| 16 | 128 | 4.393 | 0.532 |
| 16 | 1024 | 4.411 | 0.808 |
| 32 | 1 | 11.293 | 0.758 |
| 32 | 128 | 18.728 | 1.066 |
| 32 | 1024 | 18.774 | 1.625 |

These matched warmed wall times come from one allocation: median of three
64-frame blocks, independently phased environments, no per-step host readback.
Each benchmark's separate64-frame status/force/cone audit passes; B1024 admits
65536/65536 updates. All paired timing-repeat ranges separate. B1024 reductions
are81.69%/91.35%, yielding1.268M/0.630M component env-updates/s. A separate
reader finds layout line differences at most4.58e-10, coefficient differences
3.32e-8 and one iteration difference. No complete-scene throughput follows.

Initial compile failures and a misplaced new scalar-test assertion remain
captured; final GPU/CPU suites pass. The original512-iteration rejection is
preserved; the cap is2048 with adaptive stopping, not forced2048 iterations.

## Reproduction and remaining delivery

The [terminal packet](../evidence/20261019-qualified-ranking/hard-cone-history-native-v1/README.md)
retains all attempts, final sources, same-allocation timings and independent
inspection. Use captured task lists, public engine alias`d4503b9e4`, `rtx-mid`,
image`1_26`, the existing physical environment and explicit data/private caches.
All complete arrays/logs stay under the established ranking data directory.

`study_hertz_cone_history` runs CPU controls; `check_native_hertz_cone --backend
cpu|gpu` covers scalar/lifecycle cases. `check_native_cone_vectors --backend gpu
--reference <CPU-study> --benchmark` runs vectors and timing; the serial wrapper
keeps identical equations/penalty. `inspect_native_cone_layout --paired <campaign>`
reads captures without resimulation. Every command requires a new data `--output`.

Independent variable-load/vector friction references, native near-field/shell
recovery, live contact identity transport and accepted moving-scene max/full
performance remain part of the active goal. The component is not promoted to a
generally qualified public estimator.
