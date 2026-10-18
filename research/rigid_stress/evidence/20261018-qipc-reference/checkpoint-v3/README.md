# Complete capped comparisons and corrected reference admission

**The goal remains incomplete; no physical-reference accuracy ranking is
qualified.** This packet publishes already executed Slurm results, exact
execution sources and the inspected cost figure. It preserves the original
records, including older receipts whose completion flag missed a native
edge–triangle (ET) forced exit. See the
[working report](../../../docs/QIPC_REFERENCE_RESULTS_20261018.md),
[requirement audit](../../../docs/QIPC_REFERENCE_AUDIT.md) and
[full goal](../../../docs/QIPC_REFERENCE_COMPARISON_GOAL.md).

| Evidence | What it establishes |
| --- | --- |
| `summaries/case2-*`, `summaries/case7-stock.json`, `summaries/case7-volume-surface2.json` | All900 common observations over each complete9s capped history. Ratios use one common tensor/resultant metric and retain near-zero intervals. They are diagnostic discrepancies. |
| `summaries/case7-wall2-*`, `summaries/case7-strict-*` | Full9s contrasts: wall2 tangent1.37455; combined stricter stopping/budget tangent1.32335. Finger-force contrasts1.0/1.01008 retain absolute norms and the1mN floor. Neither reference control is qualified. |
| `summaries/material-query-controls.json`, `summaries/observation-grid-controls.json` | Actual inside-point fallback, outside rejection, distinct one-sided cells at the same boundary,21 unchanged legacy arrays, and refusal to emit a whole-history norm on a sparse/incomplete grid. |
| `summaries/ET-admission-controls.json`, `summaries/historical-ET-audit.json` |15 admission controls and retrospective native-log inspection. Historical4k/16k adaptive pilots have6/2 ET forced exits; their original raw receipts remain unchanged. |
| `summaries/adaptive-ET-stopped-run.json`, `summaries/ET-stop-and-strict-history.json`, `ET-failure/` | New runner stops at its first ET event,1.102s, with invalid-step flag and real static/native captures. The reported pair is preserved separately. |
| `summaries/ET-saved-geometry.json` |80-digit Decimal plane crossing lies outside the reported triangle; the copied float64 native predicate reports intersection via its approximate-coplanar branch. The saved native/FEM/contact-point geometry agrees exactly. This pair is an observed detector false positive, not proof of actual self-penetration. |
| `summaries/unloaded-bank-*`, `summaries/*-actual-poses.json` | Three complete unloaded9s robot schedules satisfy the declared joint/hand bands; actual recorded case2/7 public rigid poses agree with authored joint FK under immutable initial frame offsets. Loaded contact accuracy remains open. |
| `summaries/*Shell-B1-full-cost.json` | Three actual B1, FP64,10-substep full-field timing repeats for the exact stock/fine Shell configurations. B1 errors must not be silently paired with B32 cost because Shell batch equivalence fails. |
| `figure/` | Editable SVG/PDF,450dpi PNG, previews, actual plotted CSV and export QA for full/max, memory, cold activation and loaded kernel attribution. |
| `raw-results/*.json.gz` | Every selected full result, including per-time history and constitutive diagnostics, losslessly encoded. Decoded bytes equal the hashed original exactly. |
| `source/`, `source-variants/`, `receipts/` | One complete executed analysis capture plus every differing Python file for each selected campaign, full compressed inventories, patches, tasks, submissions and available true return codes. |

The corrected ET gate does not change any solver tolerance, field, contact law
or old evidence. A finite post-ET state is not an admitted solve even when the
ET report itself is a geometric false positive: the native Newton iteration
really terminated before its displacement criterion. For the new stopped run,
the last displacement is86.8micrometres versus the requested1pm stopping scale,
with no PCG/Newton cap. No continuation through this rejected state is claimed.
Only the actually saved reported pair receives the independent geometric
verdict; the other historical ET events were not geometrically classified.

The first forensic analysis (`ET-geometry-audit-v1`) fails an overly broad
owner-selection assertion because the native floor vertex was included with
FEM. Version2 records that discrepancy rather than hiding it. The accepted
`ET-and-case7-strict-v1` selects the single declared FEM body, excludes the
separate halfplane id and verifies exact equality against the saved FEM field
and active global contact points. Its24 crossing/outside controls cover scales,
axis permutations and reversed segment/triangle orientation; an exactly
coplanar case is explicitly unresolved by the independent linear solve.
This is not an exhaustive contact/CCD certificate or a detector repair.

At publication, the fresh case0 and the remaining complete case7 spatial,
timestep, band and friction controls are still running/pending. Their task
receipts are included for provenance, without promoting them to completed
results. The user's explicit temporary-home authorization and4GB persistent
output bound are recorded in [DATA_LOCATION.md](../../../../../DATA_LOCATION.md).
Original large fields, dumps, binaries and traces remain at their original
hashed paths. This packet is not final portable raw/source closure.

For reproduction use the campaign's exact captured sources: start with
`source/` and overlay its `source-variants/<campaign>/` files. Reconstruct the
pinned engine/patch/build/input identities from that campaign's receipts and
recorded worker inputs. Execute scientific analysis only within the established
Slurm `rtx-mid`, `GENESIS_IMAGE_VER=1_26 gs-srun` setup, with fresh explicit
output/cache paths. `capture.py` itself only selects existing values and
losslessly copies files; it performs no scientific calculation. Gzip-decoded
hashes are listed in `selection-provenance.json`; all published file hashes
are in `manifest.json`.

**Cost-figure caption.** Measured full-scene cost under distinct declared
workloads. (a) All three full/max repeats for DPR/SCR atB4096 and stock/fine
Shell atB32; markers identify output mode and batch labels are explicit.
Native numerical admission and diagnostic Shell finite completion remain
different scopes. Full observes the last physical-substep distribution, while
max covers all physical substeps. (b) Intrusively sampled process memory at
tested batch sizes; Shell S0 uses one substep and S1 ten. Joining lines do not
extrapolate a maximum capacity or imply equal accuracy. (c) Eight isolated
fresh-cache activations; each is one measured initialization/build/first-two-
steps total, with imports outside its timer. (d) Three largest individual
kernels per loaded trace, divided by that trace's summed kernel durations;
overlap means these fractions are not scene latency or an additive time model.
The [figure contract](../../../docs/QIPC_COST_FIGURE_CONTRACT.md), exact renderer,
plotted CSV, export QA and actual preview inspection accompany these exports.
