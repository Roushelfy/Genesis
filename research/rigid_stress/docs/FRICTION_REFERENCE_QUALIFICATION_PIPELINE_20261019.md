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

## Current replacement queue and reproduction

| Job | Prerequisite | Current captured work |
| --- | --- | --- |
|730607 /729479_0/_1 |20-check admission of terminal250/125um two-wall parents | Both q9/q25 surface comparisons complete; component730622_0 passes65 checks and is reused unchanged. |
|730978 /730979 /730980 | Independent periodic-state/control reinspection730990 passes99 checks | CPU final-q10/contact-q6/half-dt continuations inherit200/200/400 steps; running in disjoint node pools. |
|731010 /731011 |730978, then strict four-parent admission | Six wall/surface3/joint field groups; two completed surface2 groups are linked into the explicit eight-group view. |
|731014 /731016_0/_1 |730978 and730979, then strict pair admission | Fine contact-order q9/depth2, q9/depth4, q25/depth4; pending. |
|731013 /731016_2/_3 | All three continued parents, then strict six-parent admission | Half-dt fields retain full-history admission; pending. |
|731019 | afterany on731011/731016 | Seven-parent content guard, explicit continuation lineage,36 comparisons/18 envelopes; pending. |
|731020 | Successful qualification731019 |18 DPR/SCR/Shell q1/q9/q25 fields with complete surface exports; pending. |
|731021 | Successful qualification731019 | Full histories and actual force/traction data from the same fine reference; pending. |
|731022 | afterany on731020 |54 dimensional error rows/18 conditional metric rankings or ties; pending. |
|731023 | afterany on731022/731021 | Guarded final figures with an explicit matching fields campaign; pending. |
|731025 | afterany on731022 | Four admitted friction cost configurations/36 applicable error associations; pending. |

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

The [dimensional-error/ranking audit](STVK_METHOD_RANKING_PIPELINE_20261019.md) adds independent complete-array reductions, original candidate admission and conditional per-metric intervals. Its controls pass; final54 error rows remain pending on the qualified fine data.


The [completed250→125um surface component](FRICTION_SPATIAL250_20261019.md)
passes its original2% complete-field/5% peak and angular targets at both endpoints.
The new CLI produces no accepted reference frames and cannot substitute for
this full qualifier. Only the two independently ready original array elements
were released after20-check pair admission; the other six source/task/dependency
contracts remain unchanged. Expected rejection controls preserve the complete
parent-row guard. Coarse trends, wall/joint and fine time/contact/depth evidence
still gate the full36-comparison decision and18 conditional envelopes.


## Historical independent contact-order readiness, 2026-10-20

This publication-time queue is superseded by the node-wide preemptions and
replacement queue above. Its captured commands and original journal remain intact.

The original contact-order tasks729480_0/_1 use only the125um/three-wall
q6 and q10 parents, each400 steps. Guard730699 now waits on the actual
729314_0/729312_1 task successes and then runs the exact existing ten-check
per-parent history/momentum/endpoint admission worker. Only these two pending
array-element dependencies change. Their original q9/depth2, q9/depth4 and
q25/depth4 commands, captures, thresholds and output paths remain byte-identical.
This lets independent contact controls proceed without waiting for the800-step
half-dt control; it does not accept partial histories or infer admission from
scheduler success. A failed content check prevents either field task from running.

The other two time-control tasks retain729478. The seven-parent preflight and
all36 comparisons,18 envelopes, final method/data/rank/figure/cost gates retain
their original dependencies and requirements. The guard and both selected
field tasks remain pending at publication; all three physical writers remain
running. There is no new scientific result or qualified-reference claim.

The [contact-order scheduling packet](../evidence/20261019-qualified-ranking/contact125-pair-scheduling-v1/README.md)
retains23 payloads/165,478 bytes: immutable launch/source/task receipts, the
before/after scheduler journal, unchanged worker hash and original field captures.
Live fields and histories are excluded. Reproduce the guard using its captured
task JSON through shell_comparison.submit with explicit fresh data outputs,
--cpu-only, node-local private caches and the established rtx-mid/container path.


## Explicit continued-parent inputs after actual preemption

The three original remaining parents stop at321/342/418 complete records after
actual Slurm preemption onrtx-209-067. Only complete200/200/400-step normal
endpoints have full saved state. Original journals and frames stay untouched.
Three fresh continuations retain full body/plate state, lagged normal force,
absolute physical time and all original mechanical/constitutive parameters.
Each still needs its complete400/400/800-step own history and momentum audit.

The original PARENTS defaults remain unchanged. --parent-overrides accepts
explicit relative paths for named continuations. The current map is
meta/canonical125-resilient-parent-overrides-v1.json. Every override must bind
the correct original terminal job/submission, source modules, unchanged prefix
and copied endpoint bytes, actual mass/mesh and separate inherited/new step
accounting.26 controls pass, including wrong-parent/provenance/accounting and
invalid-path rejection. An incomplete parent set produces no accepted frames.
The periodic output option is bookkeeping and retains the same physical problem.

The data-side stvk-canonical125-resilient-fields-v1/assembly.json records
relative links: groups0/1 reuse the two completed original comparisons, while
groups2–7 map the six fresh affected computations. Raw ordered field identities
are checked independently. Fine contact and half-dt tasks use the three explicit
continued paths. All spatial/control/peak/uncertainty targets remain unchanged.

Current immutable recipes are the *resilient*-tasks.json files named in
meta/canonical125-resilient-dag-contract-v1.json. The original numerical targets
remain in meta/friction-reference-qualification-contract-v1.json; current worker
hashes are in each new analysis-files.json. Use captured task JSON through the
existing Slurm/container runner with fresh explicit data outputs. The figure
entry additionally accepts --fields-campaign so the new q1 arrays match its
qualified ranking; the actual normal-only negative exits1 with no figures.

The [recovery packet](../evidence/20261019-qualified-ranking/canonical125-resilient-recovery-v1/README.md)
retains794 payloads/5,326,422 bytes and18 externally hashed arrays on data. It
includes original preemptions, both superseded recovery attempts, failed JSON
postprocessing and read-only reinspection, tested source variants and the exact
replacement graph. Live scientific outputs and final scientific products are
excluded. Full qualification, method ranks, inspected final figures and matched
friction delivery remain pending.
