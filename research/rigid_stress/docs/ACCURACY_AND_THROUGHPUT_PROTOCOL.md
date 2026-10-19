# Separate accuracy validation and production throughput

User direction, 2026-10-19: test accuracy and performance separately. Continue
the [qualified error comparison](QUALIFIED_STRESS_RANKING_GOAL.md), while moving
production throughput measurement and optimization forward now. Expensive
reference convergence is not a prerequisite for short performance iterations;
numerical consistency and runtime validity remain required.

The feature stays inside the Genesis rigid solver and all production numerical
work stays in Quadrants. Preserve fixed-shape preprocessing, arbitrary changing
convex/nonconvex contacts, asymmetric normal/friction loads, max/full output,
batch isolation and partial resets. Keep the independent FEM/reference tools
as validation infrastructure.

## Three separate measurements

| Measurement | Workload and accounting | Main result |
| --- | --- | --- |
| Accuracy | Small representative physical cases, qualified fine FEM reference, complete saved fields, convergence and independent residual checks. Reference and offline analysis cost are separate. | Per-configuration stress/peak errors, uncertainty, stability and failures. |
| Stress-module throughput | Replay representative recorded contact histories already resident on GPU, batched through actual native reconstruction/recovery/max kernels. Count recovery updates and state the cadence; exclude physical dynamics. | GPU module time, env-updates/s, memory and bottleneck profile. |
| End-to-end production throughput | Actual native Genesis scenes and evolving environments, comparing pure rigid, rigid plus maximum stress and rigid plus complete stress. Include production contact, integration, required validity checks and the requested output contract. | Added scene-step cost and aggregate env-step/s; actual policy transitions/s where measured. |

Use the existing low-resolution Franka/egg scene and useful simple contact
controls for fast iterations. Include changing contact and friction instead
of benchmarking only a fixed settled symmetric contact. Label homogeneous
duplicated-input microbenchmarks; they do not replace independently evolving
batched scene runs. Physical Franka reference qualification is not revived.

Begin with single-scene and useful increasing batch sizes, then tune toward
the best valid throughput and practical memory capacity. B4 alone does not
establish large-batch RL throughput. Record timestep, substeps, policy
decimation, output location and observation cadence. Do not equate a stress
update, a physical substep, a scene step and a policy transition.

## Keep the timed production path representative

Keep the reference FEM, convergence analysis, detailed independent mechanical
audits, plotting, rendering, per-step file journals and full-field host dumps
out of production timing. Validate before/after timing and with separately
instrumented runs using the same configuration and representative states.

Preserve required production numerical acceptance, error handling and history
logic. Removing a required check, dropping a difficult environment, omitting
real contacts or timing a successful prefix of a failed run is not an admitted
speedup. Check complete run validity and representative output conformance
without injecting research-only CPU synchronization into every timed step.

For the maximum-stress RL mode, reduce on GPU and keep reward inputs there.
Full-output mode can return a GPU tensor without mandatory CPU transfer;
measure a requested host readback separately or include it in a clearly named
host-output mode. Sparse sampling is an additional configuration, never a
substitute for every-step performance or every-step reliability.

Compare identical rigid settings, scenes, controls and physical time horizons
against a paired pure-rigid baseline. Recovery must not change motion when it
has no feedback. Include unavoidable dispatch, transfers and allocation in
end-to-end time; report reusable preprocessing/factorization separately.
A GPU-resident contact replay measures the module, not total simulator speed.

Use warmed repeated primary wall-clock intervals and retain dispersion.
Use separate detailed profiling for contact extraction, source preparation,
load reconstruction, elastic recovery, history/fallback work, maximum
reduction, dispatch, synchronization and copies. Trace the measured bottleneck
before optimizing. Mark inclusive versus exclusive phases and overlapping GPU
work; do not sum incompatible timings. The current all-source SCR reconstruction
is already a measured bottleneck, so removing logs alone is not a demonstrated
solution. Keep the research adapter's serial per-environment pad work separate
from the production rigid solver's batching behavior.

## Join accuracy and speed by the actual configuration

A valid error-versus-performance row binds algorithm/variant, material and
geometry, recovery mesh, source/contact representation, reconstruction radius,
precision, solver tolerances, timestep/history policy, output mode and cadence.
Batch-size changes require numerical/environment conformance controls.

Changing a setting for speed is encouraged. First verify the optimized path
against the matched numerical baseline on representative states, then obtain
or reuse applicable physical-error evidence. Changes to the load model,
contact sources, mesh, precision or solver accuracy need the corresponding
revalidation; they cannot automatically inherit an earlier error value.
GPU/CPU numerical agreement alone does not qualify a new physical load model.

In particular, the current approximately8% native normal-field errors belong
to the declared polar-land egg, compliant pads, full physical centroid sources
and fixed2mm recovery radius. A default native-collider run with a few contact
points is a different load configuration. Measure its production throughput,
but leave its physical-error cell pending until that configuration is checked.
Do not combine its speed with the compliant-pad error as one validated result.

Existing pad workflow timings remain valid for their documented research
scope, including their audits, transfers and I/O. Preserve them under that
name; do not present them as production-only rigid-stress throughput or
retroactively subtract estimated overhead to manufacture a faster measurement.

The current DPR/SCR every-step friction failures and successful requested
keyframes remain distinct evidence. A sampling change cannot close the
every-step reliability issue. Address feasible numerical/input inconsistencies
at their source while retaining genuine infeasibility rejection and the
declared physical contract; do not suppress failures to obtain a timing.

## Delivery

Publish separate accuracy, module-performance and production-performance
tables, plus a configuration-matched error/throughput summary. Include pure
rigid versus max/full overhead, every-step versus sparse output, batch scaling,
memory and failure rates, with source/configuration identities and runnable
commands. Keep unsupported or pending error cells explicit.

Continue physical-reference and friction qualification independently; these
measurements do not waive the original error-ranking goal. Keep performance
iterations low-resolution and numerically checked, without repeating expensive
fine reference studies after unrelated dispatch/copy optimizations. Follow the
existing [running guide](RUNNING_PARALLEL_PLATES.md) and
[storage policy](../../../DATA_LOCATION.md). Preserve all earlier evidence.
