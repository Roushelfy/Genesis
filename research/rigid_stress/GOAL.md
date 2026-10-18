# Active goal: three stress methods against a QIPC reference

Follow [docs/QIPC_REFERENCE_COMPARISON_GOAL.md](docs/QIPC_REFERENCE_COMPARISON_GOAL.md).

Deliver a detailed comparison of Directional Patch Recovery (DPR, currently
pressure), Surface-Constrained Recovery (SCR, intrinsic + body_source_prior),
and Kashu shell PR22. Use qualified cuda-graph-qipc 3D solid FEM contact
references for full Franka/hollow-egg grasp histories. Compare errors under
common observables, speed/memory, stability and repeatability; explain reference
uncertainty, load reconstruction and omitted deformation feedback.

Run common prescribed-motion and representative original force-limited-control
protocols. Keep normal force and friction, arbitrary convex/nonconvex changing
contacts, full-field primary timing and max-only secondary timing. Make the two
rigid route names understandable without changing their physics by renaming.

Reuse the [completed shell comparison](docs/SHELL_COMPARISON_RESULTS_20261018.md),
[pressure comparison](docs/PRESSURE_COMPARISON_RESULTS_20261018.md) and existing
R/L/M/independent FEM evidence. They are baselines, not already qualified common
ground truth for this new stage. Keep the native Quadrants feature and cleaned
layout intact. Use engineering judgment on experiment size; preserve failures,
exact sources and reproducible results.
