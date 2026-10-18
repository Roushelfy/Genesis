# Native rigid stress recovery

The feature is part of the existing native rigid solver. It reconstructs
conditional elastic stress from changing rigid contacts, including friction,
and supports raw global max/full output, complete numerical checks, environment
isolation and partial resets. Production numerical work remains in Quadrants.

The active [three-method QIPC reference goal](docs/QIPC_REFERENCE_COMPARISON_GOAL.md)
extends those results to a common qualified contact/elasticity reference for
complete Franka/egg grasps. It compares Directional Patch Recovery (DPR,
currently pressure), Surface-Constrained Recovery (SCR, intrinsic +
body_source_prior) and deformable shell for error, speed, memory and stability.
`RigidStressOptions.dpr(...)` and `.scr(...)` select these exact existing routes;
the presets preserve their physics/defaults. The [running guide](docs/RUNNING_QIPC_REFERENCE.md)
and [progress record](docs/QIPC_REFERENCE_PROGRESS.md) retain current controls,
complete candidate histories, failed reference attempts and open qualification gates.
The [working results](docs/QIPC_REFERENCE_RESULTS_20261018.md) and
[compact checkpoint](evidence/20261018-qipc-reference/checkpoint-v1/README.md)
retain the measured complete-history discrepancies and the unresolved spatial/
timestep uncertainty; the goal is not complete.

The [shell PR22 comparison](docs/SHELL_COMPARISON_RESULTS_20261018.md) is complete
within its [declared scope](docs/SHELL_SOLVER_COMPARISON_GOAL.md). It pins both
implementations, defines compatible full fields, executes analytic/same-load
and contact/Franka controls, and measures full-field cost on matched hardware.
Its [running guide](docs/RUNNING_SHELL_COMPARISON.md),
[audit](docs/SHELL_COMPARISON_AUDIT.md) and
[evidence packet](evidence/20261018-shell-comparison/comparison-v1/README.md)
preserve exact source/input/hardware identities, failures and editable figures.

At B4096, full-output native pressure/tuned/intrinsic achieve239264/255089/99024
numerically valid aggregate env-step/s. Original shell32/64 measure36282/24885
diagnostic env-step/s; no accuracy-qualified shell grasp setting was established.
Shell deformation feedback remains an additional capability, while its contact,
time integration and state-history limitations are documented. The native
production engine and previous accepted evidence remain unchanged.

The current [pressure/intrinsic comparison](docs/PRESSURE_COMPARISON_RESULTS_20261018.md)
is complete within its declared scope. Its [running guide](docs/RUNNING_PRESSURE_COMPARISON.md)
and [compact evidence packet](evidence/20261017-pressure-comparison/comparison-v1/README.md)
provide current paired measurements, configuration choices, model limits,
source/input/hardware identities and inspected editable figures.

At B16384 on RTX PRO6000 Blackwell, the default max/history0 pressure route
measures296859 live/296297 policy valid aggregate env-step/s, versus
113933/113971 for intrinsic body_source_prior. Separate matched pressure tuning
confirms6.68%/6.38% throughput gains, with about21.8GiB sampled process peaks.
These are inference rollouts; raw repeats, matched rigid overhead, memory,
max/full and capacity evidence are in the report.

Choose pressure as a declared fast proxy where local-normal and topology
limitations are acceptable. Prefer intrinsic body_source_prior when those
constraints matter. Both retain substantial footprint/reference uncertainty;
neither calibrates an absolute physical damage threshold. The matched two-model
comparison and separate radius/reward sensitivity support conditional choices.

Start with [API](docs/NATIVE_API.md), [running instructions](docs/RUNNING.md),
[source-prior model](docs/BODY_SOURCE_PRIOR_MODEL.md) and [goal/status](GOAL.md).
[Application results](docs/PRACTICAL_APPLICATION_RESULTS_20261016.md) retain
independent FEM/R/L/M and feedback limits. The preceding
[throughput study](docs/PRACTICAL_THROUGHPUT_RESULTS_20261017.md) retains distinct
code/history comparisons and separately tuned intrinsic capacity. Its older
rates are not substitutes for the current paired pressure measurements.

Tests and useful reference modules remain at their original import paths.
[Cleanup results](docs/BRANCH_CLEANUP_RESULTS.md) and [archive](archive/README.md)
preserve historical identities, failed cases and unresolved research without
reviving old campaigns. The original seven canonical packets remain unchanged.
