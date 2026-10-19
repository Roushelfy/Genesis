# Complete nonlinear friction qualification pipeline

Gate controls729509 pass their expected outcomes, but the fine reference is
still running. No final friction error/rank or final comparative figure is
admitted by these controls. The full [goal](QUALIFIED_STRESS_RANKING_GOAL.md) and
[completion checklist](QUALIFIED_STRESS_COMPLETION_CHECKLIST.md) remain active.

## Evidence required from the actual reference

`qualify_friction_reference` requires seven complete, independently admitted
StVK histories:500um/two-wall,250um/two-wall,250um/three-wall,125um/two-wall,
125um/three-wall,125um q6 and125um half timestep. All retain the declared0.1N
normal and sustained-friction endpoints, free rigid modes and physical pads.
The first six histories have400 steps at1ms; the half-step history has800 at0.5ms.
Final q10 and the explicit q6 control remain distinct.

The preflight checks both windows, every original mechanical gate, independent
current momentum and all fourteen endpoint frame hashes. Qualification also
checks consistent physical inputs across all seven parents, original ordered
actual/reference fields, corresponding meshes and the canonical-force repair
controls. The common0.1N load alone cannot identify an endpoint: normal and
friction frames have distinct level identities and hashes.

| Comparison family | Required angular/depth sampling per endpoint | Purpose |
| --- | --- | --- |
| Historical500→250um surface and250um wall | q9/q25, depth4 | Retain both independent coarse trends. |
|125um surface at two walls,125um wall, surface at three walls, joint refinement | q9/q25, depth4 | Require complete fields, raw/common peaks and surface/wall trends. |
| Fine q6 versus q10 and0.5ms versus1ms | q9/depth2, q9/depth4, q25/depth4 | Measure actual fine contact/time and integration sensitivity. |

The matrix contains36 actual comparisons across two endpoints. Spatial targets
remain2% for all five compatible tensor/resultant fields and5% for both raw and
both common peaks. Fine control targets remain0.5%/1%. Angular/depth checks retain
0.005 absolute relative-error units for field-contrast changes and1% for peak-value changes. Neither passing peaks nor passing
thickness alone can override an unresolved complete surface field.

The output retains every actual gate, input hash, change metric and eighteen
metric/endpoint envelopes. Each envelope sums observed spatial, time, contact,
angular and depth changes. These are conditional numerical sensitivity measures,
not statistical intervals or certified continuum bounds. A failed/missing input
writes a rejected result and exits nonzero. Only complete success publishes the
accepted fine frame identities for downstream method comparison.

## Executed controls

729509_0 passes19 checks using actual existing results. The250um wall comparisons
pass;500→250um surface comparisons remain rejected at approximately2.749% despite
their passing peaks. All four admitted coarse time/contact controls retain their
outcomes. In-memory negative controls reject altered frame hashes, swapped
normalization order, missing/duplicate peaks and nonfinite metrics/positions.
Original scientific artifacts are unchanged.

729509_1 intentionally fails: a passing coarse-parent receipt does not establish
the required seven-parent matrix and produces no accepted fine frames.
729509_2 passes all ten admission checks on the required historical500um parent.
These controls verify the reader/gates, not the still-missing fine results.

The plotting-data reader's CPU-only control729516 passes all five own-history,
reaction-total and native-contact agreement checks, producing1600 complete
history rows and eight unsmoothed traction arrays from the completed500um case.
The only reader change makes GPU inventory optional for this CPU analysis;
the physical extraction and its acceptance are unchanged. Final extraction is
separately queued for the fine reference after qualification.

## Queued execution and reproduction

| Job | Prerequisite | Work and publication-time state |
| --- | --- | --- |
|729487 /729479 | q10 reference729312, then explicit four-parent admission | Fine spatial fields; pending. |
|729478 /729480 |729312 and729314, then explicit six-parent admission | Fine time/contact fields; pending. |
|729513 | afterany on729479/729480 | Seven-parent preflight followed by36-comparison qualification; pending. |
|729518 | successful qualification729513 |18 DPR/SCR/Shell q1/q9/q25 comparisons across both0.1N endpoints; pending. |
|729519 | successful qualification729513 | Complete actual-force histories and traction data for the final figures; pending. |

`afterany` allows qualification to report failed or missing upstream evidence;
the explicit content checks establish admission. Downstream method/data work
uses `afterok` on a qualifier that exits nonzero on any failed required gate.
All queues preserve captured worker sources and use fresh data-side outputs.

The method fields retain raw Mises/tensile maxima and locations, compatible
common peaks, five complete field metrics, dimensional errors and full membrane/
bending arrays. q1 exports are for display; q9/q25 supply numerical comparisons.
The additional normal0.1N endpoint checks the fine StVK material/geometry alongside
the friction endpoint. Earlier multi-load linear-reference normal results keep
their original model and scope. Native fields are admitted requested observations;
their separate failed every-step friction runs remain failures.

Captured recipes are `meta/canonical-friction-reference-qualification-v1-tasks.json`,
`meta/canonical-stvk-method-fields-v1-tasks.json` and
`meta/canonical-friction-plot-data-v1-tasks.json` under the ranking data root.
The explicit thresholds/source contract is
`meta/friction-reference-qualification-contract-v1.json`. Reproduce through
the [running guide](RUNNING_PARALLEL_PLATES.md), rtx-mid and
`GENESIS_IMAGE_VER=1_26 gs-srun`, preserving input identities and replacing output
paths with fresh locations. Large fields, force histories and arrays remain
under the project data directory. Final ranks, captions and inspected figures
still require the actual successful fine results.


The [pipeline packet](../evidence/20261019-qualified-ranking/friction-qualification-pipeline-v1/README.md)
retains92 compact payloads/503,395 bytes: exact worker variants, actual terminal
positive/negative controls, target contract and immutable queued task inputs.
Eight traction arrays and one complete force-history CSV remain on data with
hashes. Live fine results and pending method/figure outputs are excluded.
