# Completed comparison: stress recovery with Kashu shell PR22

Follow [docs/SHELL_SOLVER_COMPARISON_GOAL.md](docs/SHELL_SOLVER_COMPARISON_GOAL.md).

The bounded comparison has been executed. See the
[results and decision](docs/SHELL_COMPARISON_RESULTS_20261018.md),
[running guide](docs/RUNNING_SHELL_COMPARISON.md),
[requirement audit](docs/SHELL_COMPARISON_AUDIT.md) and
[compact evidence](evidence/20261018-shell-comparison/comparison-v1/README.md).
The report retains reproduced defects and unresolved physical limits; completion
does not mean the shell grasp was accuracy qualified or its defects repaired.
No broad follow-up campaign is active.

Compare our pressure/intrinsic stress recovery with
[Kashu7100/Genesis PR22](https://github.com/Kashu7100/Genesis/pull/22)
for full stress distributions, contact robustness, numerical consistency,
Franka/egg grasping behavior and accuracy-versus-throughput/memory.
Pin source versions, unify stress units/measures and separate prescribed-load
mechanics from contact/deformation feedback. Reproduce reported shell problems
before treating them as conclusions; preserve the unmodified baseline.

Reuse the completed [pressure comparison](docs/PRESSURE_COMPARISON_RESULTS_20261018.md)
and [reproduction guide](docs/RUNNING_PRESSURE_COMPARISON.md).
Keep the native Quadrants feature and cleaned layout intact. Use inexpensive
controlled cases, bounded application checks and compact reproducible evidence.
Historical campaigns and fracture calibration remain outside this stage.
