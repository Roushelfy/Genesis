# Execution with bounded temporary output after the storage outage

The data mount still rejects exclusive creation with errno122 at12:31UTC.
The temporary output root was explicitly announced before execution and
recorded in [DATA_LOCATION.md](../../../DATA_LOCATION.md):

`/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/`

Call this root `F`; original large inputs/results remain at the existing data
root `R`. No original run, compatibility link or environment is migrated.
The initial persistent-output budget was1GB; it is explicitly revised to2GB
at13:44UTC before crossing that initial budget, to retain selected running
histories, the sparse adaptive pilot and compact audit/figure outputs.
At the first complete validation
snapshot it uses about215MB. Actual totals are monitored while bounded selected
histories proceed. Compile/model/temp caches use private compute-node `/tmp`
via `--node-local-cache`; worker receipts show a tmpfs, and successful exit
traps remove only their private scratch. Normal submissions retain their
existing data-side cache default. Scientific work still uses Slurm `rtx-mid`
and `GENESIS_IMAGE_VER=1_26 gs-srun`.

## Completed validation, job726497

All eight groups and nine tasks in `F/validation-after-storage-v1/` exit0.
The captured task file rewrites only the new group6 metric-output references;
all old large inputs are read in place.

| Evidence | Actual result |
| --- | --- |
| Interrupted evidence admission |12 synthetic controls pass: missing/corrupt result, truncated/middle-corrupt journal, missing/failed terminal receipts, absent finite/cap observations and nondefault cap values are explicit. Raw damaged bytes remain unchanged. |
| Independent Shell max observer | Two environments,420 actual controller ticks of loading followed by200 physical observations. VM computed independently from complete Cauchy differs from max-only by at most2.9802322387695312e-8Pa. A NaN probe affects only the selected environment. |
| Max output application contract |420 warmup and40 measured ticks complete. The substep hook count is checked at every scene interval. This wiring control is not primary timing. |
| Observer payload | Full53784B versus max352B for the tested two-environment surface0 scene. Max covers all physical substeps; full exports the last-substep field. This is not a claim that max must be faster. |
| History audit |23 completed/interrupted histories summarized with signed extrema, explicit missing observations, actual cap counts and process/journal admission. |
| Refined three-way comparison | Surface2/wall4 gets its own rebuilt element/barycentric metric lookup. All600 common observations are covered. Tangent history discrepancies are DPR0.999381, SCR0.999376, stock Shell4.654631; still diagnostic. |
| Fixed-wall surface contrast | Surface3/wall2 versus surface2/wall2 changes the common tangent norm0.411607 and bending0.713727. Spatial qualification remains open. |
| Physical finger reaction contrast | Surface2/wall4 changes force/moment norms0.339955/0.334382 versus coarse; half-dt changes0.096648/0.109692. Opposing finger loads are kept separate. |

Maximum actual net residual versus egg weight is0.002208 coarse,0.006439
surface2/wall4 and0.000555 surface3/wall2. Recorded PCG-cap counts are263/549/0
respectively; no recorded Newton cap occurs. These numerical diagnostics remain
separate from the opposing-finger reaction and spatial/timestep contrasts.

The native probe reports maximum contact penetration5.537mm prescribed and
6.094mm capped. Friction utilization reaches1 up to floating-point roundoff,
with the declared normal-force floor and no clamping. Shell cutoff controls
retain large finite raw peaks: volume-surface2 prescribed2.439e11Pa and capped
1.403e16Pa, with minimum vertices below the plane. Finite completion and even
zero linear caps do not establish physical reliability.

## Subsequent jobs and load-replay scope

| Campaign under `F` | Job | Scope / status at submission |
| --- | --- | --- |
| `shell-full-max-paired-v1` |726506 | Both matched full/max groups complete. B32/FP64/ten substeps,900 warmup and three900-tick repeats. Stock medians358.142/356.189 and surface2-volume65.4668/65.3310 diagnostic env-step/s. Original primary rates stay unchanged. |
| `capped-fresh-coarse-v1` |726509 | Fresh original-initial-state9s QIPC cases0/2, same existing coarse configuration and numerical variant. Actual public native/controller checkpoints every0.5s. No interrupted friction history is reconstructed. Running. |
| `shell-resolved-replay-pilot-v1` |726513 | Two7-time Shell dead-load static replays, six outer relaxation iterations. Process complete; each variant admits3/7 under the strict static-force budget. |
| `shell-resolved-replay-pilot-v2` |726515 | Same pilot,24 outer iterations and absolute residual output. Each variant still admits3/7; all recorded linear PCG caps are zero. |
| `shell-resolved-full-replay-v1` |726518 | Both groups complete600 statics, but both actually use nominal thickness. Group0 is valid nominal; group1 is a mislabeled nominal duplicate because task generation omitted `--wall-input`. Raw bytes/receipts are retained, with a correction record. |
| `shell-resolved-full-replay-v2` |726545 | Corrected volume-only600-time replay completes:113 admitted/487 rejected. Its captured input and actual rest mass/thickness verify the volume-surface1 variant. |
| `shell-resolved-replay-audit-v1` |726543 | Complete nominal audit:114/600 admitted,486 rejected, zero PCG caps, all recorded fields finite. The second source is the mislabeled nominal duplicate, not an independent volume contrast. |
| `adaptive-input-v1/v2` |726523/726528 | First face-bound attempt admits neither mesh. The second admits h0.5mm/wall4:30290 vertices,145344 tets. h0.25mm exceeds the20000-face bound; no solid is admitted. |
| `adaptive-prescribed-pilot-v1` |726533 | Completes the requested1.16s in1072.4 intrusive seconds, with1160 physical observations and14 sparse full frames.696 PCG-cap steps and large force imbalance reject numerical qualification. This is a prefix pilot, not a full-history reference. |
| `expanded-outcomes-v1` |726534 | All ten groups complete. Full friction-half contrast, seven prescribed Shell geometry/cutoff/dt outcome variants and stock/volume case7 outcome contrasts are retained. |
| `reference-diagnostics-v2` |726532 | Complete23-source descriptive audit adds recorded equality-coefficient, gap and CCD statistics. Missing observations remain explicit. |
| `cost-consolidated-v2` |726552 |112 configuration records, six full profile tables and eight cold-cache records; the new matched Shell pair includes actual output bytes/physical-step scope. |
| `shell-resolved-replay-audit-v2` |726554 | Uses valid nominal plus corrected volume only. Both actual process receipts and wall mass/thickness are checked. The mislabeled nominal duplicate is excluded. |
| `adaptive-prescribed-audit-v1` |726557 | Reaction and history groups complete; common-column field comparison exits1 at the geometric domain-coverage assertion. It produces no field score. |
| `adaptive-metric-gap-v1` |726564 | Independent float64/80-digit Decimal query confirms the first uncovered sample lies inside an actual tetrahedron; diagnostic complete, field admission unchanged. |
| `qualification-figure-v3` / `qualification-export-qa-v1` |726561/726562 | Accepted complete-reference sensitivity/raw-peak/signed-work figure. Text bounds/export checks pass; actual PNG and PDF previews are inspected. Earlier clipped/overlapping drafts are retained and excluded. |

The Shell transfer maps contact and measured inertial loads separately to the
supporting midpoint triangles. Local force couples preserve the actual source
point's wrench and rigid virtual work **in the fixed undeformed Shell
configuration**. Pilot random-force/moment defects are about1e-13N/1e-15Nm,
and rigid-work discrepancy3.55e-15 in its synthetic units. A pure through-wall
couple control also passes. No footprint radius or stiffness is fitted.

After nonlinear relaxation, the forces act at deformed Shell vertices, so their
actual moment can drift. The full replay records net drift and the sum of local
drift norms, plus the separately reported mass-weighted six-mode balance
correction. It does not preserve arbitrary nonrigid3D virtual work. Pilot scope
clarifications retain this limitation without changing original fields/results.
The adapter uses original PR22 force/stiffness/PCG/integration kernels and the
existing prescribed-force right-hand-side input. These independent static
elastic-response tests do not replace complete contact/control histories or
qualify a physically equivalent3D traction field.

For the valid nominal600-time replay, absolute static residual has median532.94N
and maximum31919N. Net deformed contact moment drift has median15.173Nm and
maximum22.387Nm; the sum of local drift norms reaches22.412Nm. The all-finite
tangent discrepancy norm11.1837 includes rejected statics and is not an
accuracy-qualified score. Increasing six to24 outer iterations in the pilots
does not improve their3/7 admission. These failures confound load transfer,
deformed geometry and nonlinear relaxation; they are not a clean Shell
constitutive-error measurement. The accidental volume omission above is a
task-generation error, separately corrected without editing the original result.

Corrected volume replay has the actual mass0.006529848210401324kg and thickness
0.435712–0.483267mm; nominal replay is0.006946250126509785kg and0.5mm.
The volume replay's median/max absolute static residual is307.748/28054.684N,
maximum deformed contact net-moment drift19.1197Nm and diagnostic all-finite
tangent discrepancy6.20259. It also fails the strict static admission budget;
the smaller diagnostic discrepancy is not a qualified accuracy advantage.

The adaptive mask uses30 material rays carrying actual reference contact force
above1e-6N in the complete surface2/wall4 history, with2mm padding. Conforming
edge refinement interpolates the same outer/inner faceted domain; manifold,
orientation, exact volume/mass and continuum-inertia controls pass. It changes
contact resolution without imposing reference forces or fitting a footprint.
The unresolved finer mask and the short sparse pilot remain explicit limits.
The pilot's raw global VM reaches647.933MPa at1.13s. Maximum recorded net force
residual is38.8508N, or606.496 egg weights, so finite completion is not numerical
qualification. Its14-time reaction audit versus surface2/wall4 changes force/
moment57.62%/56.78% at1.13s.586 common candidate ticks are absent and the audit
correctly emits no whole-history norm. The separate field audit stops on a
common-column gap assertion; a subsequent independent float64/80-digit Decimal query finds a positive
barycentric point inside an actual tetrahedron near the outer boundary. This
indicates a query/interval-admission defect at that first failing point, not
proof of a physical hole. No tolerance or coordinate is changed and no field
score is admitted. Further coverage diagnosis remains open. Raw failed analysis and its source
remain available for diagnosis.

The halved-friction-smoothing complete-history tangent contrast is0.0188165;
finger force/moment contrasts are0.00682619/0.00682133. These small contrasts
do not resolve the larger surface/timestep dependence. The mapped equality
inertia-coefficient discrepancy reaches2.901e-7 in complete capped case7 and
5.389e-6 in the old case0 prefix, so this measured component does not justify
inventing a controller correction. The prescribed coarse CCD journal records
a maximum140 iterations and a minimum reported gap ratio0.0435535; these are
recorded diagnostics, not a certificate that every native CCD operation passed.

Actual mass-COM outcomes expose configuration dependence. Prescribed
volume-surface0/surface1/surface2 Shell maximum COM heights are
0.03048/0.13367/0.03047m; strict nominal dt1ms/0.5ms gives0.03037/0.14512m.
The latter still has0.12832m first-lift hand-relative COM drift. Complete capped
case7 reference/stock Shell maximum COM heights are0.02895/0.02890m, while
volume-surface2 Shell reaches0.14715m and diverges from that reference outcome.
Visual lift alone is not a measured stress-accuracy advantage. The authored
hand-frame/macroscopic drift analysis does not classify local contact slip.

No full-reference qualification, final source closure, refreshed final figure
set or goal completion is inferred from these results. The
[requirement audit](QIPC_REFERENCE_AUDIT.md) and
[working report](QIPC_REFERENCE_RESULTS_20261018.md) remain the completion record.
The [selected checkpointv2](../evidence/20261018-qipc-reference/checkpoint-v2/README.md)
preserves these already-derived values, exact captured sources/receipts,
profile tables and the inspected uncertainty figure without recalculating
scientific metrics on the login node.
