# Native rigid stress recovery

The feature is part of the existing native rigid solver. It reconstructs
conditional elastic stress from changing rigid contacts, including friction,
and supports raw global max/full output, complete numerical checks, environment
isolation and partial resets. Production numerical work remains in Quadrants.

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
