# Native rigid stress recovery

Optional stress recovery for Genesis's existing rigid solver. Geometry remains
rigid; a shared linear-elastic model recovers a global raw maximum von Mises
stress or the full stress field from normal and frictional contact loads.
Production numerics use Quadrants and do not depend on the research prototype.

Start with [running the example and tests](docs/RUNNING.md), then use the
[API](docs/NATIVE_API.md) and exact benchmark reproduction below.
The [active goal](GOAL.md) is to complete current pressure/intrinsic comparison
data and an evidence-based configuration guide.
[Cleanup results](docs/BRANCH_CLEANUP_RESULTS.md) and [the archive index](archive/README.md)
explain the retained tools and historical source/evidence recovery.

## Use and implementation

- [Native API and options](docs/NATIVE_API.md).
- [Franka/full hollow egg example](../../examples/rigid/franka_egg_stress.py).
- [Production implementation](../../genesis/engine/solvers/rigid/stress/).
- [Recommended settings and benchmark reproduction](docs/RUNNING_PRACTICAL_THROUGHPUT.md).
- [Load-model assumptions](docs/BODY_SOURCE_PRIOR_MODEL.md).

## Current accepted results and limits

[Throughput results](docs/PRACTICAL_THROUGHPUT_RESULTS_20261017.md):
480 P2 tetrahedra,2430 DOF, FP64 intrinsic body_source_prior.
The measured starting recommendation is history0, max output and B16384 on
RTX PRO6000 Blackwell:116912 live /116695 policy valid env-step/s and about
40.5GiB sampled process memory. B32768 reaches118929/118579 with about79GiB.
These are workload-specific aggregate rates and policy inference, not full
RL training throughput or individual-environment FPS.
The [throughput acceptance audit](docs/PRACTICAL_THROUGHPUT_COMPLETION_AUDIT_20261017.md)
records the exact qualification scope.

[Application results](docs/PRACTICAL_APPLICATION_RESULTS_20261016.md) separate
elastic agreement, reconstructed-load error and omitted feedback.
The feature is a conditional stress/reward proxy: contact footprint assumptions,
reference uncertainty and phase-dependent grasp rankings remain material limits.
Absolute physical fracture thresholds are uncalibrated. The faster pressure
model has a diagnosed local contact/friction-admissibility limitation.
Full/max modes remain supported; the rates above measure max output.

Earlier rates, standalone CPU/GPU backends and broad convergence studies are
historical evidence. Consult them through linked reports or the archive rather
than treating them as current instructions or current-feature measurements.
