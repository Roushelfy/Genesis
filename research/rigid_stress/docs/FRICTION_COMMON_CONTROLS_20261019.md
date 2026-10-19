# Common-sampling controls for the nonlinear friction reference

All24 checks pass for coarse-reference time/contact controls evaluated with
9/25 angular samples and common tangent-plane peaks. This removes a sampling
definition mismatch, but does not qualify the pending fine spatial reference
or complete the [goal](QUALIFIED_STRESS_RANKING_GOAL.md).
The [packet](../evidence/20261019-qualified-ranking/friction-common-controls-v1/README.md)
retains88 verified payloads totaling1,012,437 bytes.

`stvk-time-contact-common-fields-v1` (728592) re-evaluates existing completed
trajectories with physical-volume columns, exact StVK Cauchy stress, depth4
and surface quadrature9/25. It changes no physical equation. All raw maxima
and common VM/tensile peaks are retained. The earlier depth2/4 control remains
separately identified at surface quadrature3.

`stvk-time-contact-common-audit-v1` (728617) verifies four parent histories:
the400-step original,800-step half-dt and both400-step contact-order6/10 runs.
Both windows and independent current-momentum audits pass. All eight endpoint
fields are rehashed and matched to their momentum and comparison receipts.
A different passing run cannot admit the analyzed fields.

Table entries are **percent changes**, with25 angular points. The field column
is the maximum over tangent/membrane/bending/outer/inner fields; the raw-peak
column is the larger VM/tensile change.

| Control | Endpoint | Largest field | Largest raw peak | Common VM | Common tension |
| --- | --- | ---: | ---: | ---: | ---: |
| dt1 to0.5ms | Normal | 4.418e-8 | 1.754e-8 | 8.250e-10 | 1.737e-8 |
| dt1 to0.5ms | Friction | 3.315e-5 | 1.042e-5 | 3.183e-6 | 1.167e-6 |
| Contact order6 to10 | Normal | 0.124304 | 0.0124295 | 0.0129106 | 0.0070210 |
| Contact order6 to10 | Friction | 0.124312 | 0.0124219 | 0.0129002 | 0.0071234 |

The unchanged control bands are0.5% fields and1% raw/common peaks. The common
peak values themselves change by up to0.429251% between9 and25 angular points;
this sampling effect must be retained separately from contact-order sensitivity.
Angular sampling changes field-contrast metrics by at most3.877e-7 in absolute
relative-error units. The prior depth2/4 difference is at most1.076e-14, at q3.
These are coarse one-wall controls, not measured fine-mesh time/contact effects
or certified continuum bounds. Fine spatial and final common-peak controls
remain required. Cache/response-reuse controls remain separately identified.

The completed500um linear-static/StVK-dynamic contrast only covers0.1N.
`solid-pad-stvk-normal-load-controls-v1` (728615) adds0.3/1N on the same500um,
two-wall mesh, matching E, nu, pad stiffness and q10 contact. It uses validated
direct correction, exact caching and response reuse. Its complete history and
momentum audit must pass before the captured q9/q25 field/peak comparisons run.
Predeclared reporting bands are0.5% fields and1% peaks; they do not modify
simulation acceptance or establish a finest-mesh model-error bound. Results
are pending; the packet includes only immutable submission/source receipts
and targets from this active campaign.

The retained `friction-final-field-plan-v1.json` fixes500-to250um surface,
250um wall,125um wall/surface/joint, and actual DPR/SCR/250um Shell field tasks.
These are unexecuted plans. They retain the complete domain,2mm native radius,
2% spatial-field/5% peak and0.5% control-field/1% peak targets. Method q1 exports
are display samples; q9/q25 provide metrics. Verify terminal scheduler status,
complete histories, both windows and momentum audits before dependent submission.
Output existence or a dependency environment variable is insufficient. Native
every-step failures remain separate from valid requested equilibrium fields.

Reproduce via captured tasks and the established rtx-mid / gs-srun workflow,
with fresh data-side paths. `audit_stvk_time_contact --data-root RUN_ROOT
--output NEW_DATA_PATH` reproduces the24 checks after the field campaign completes.
Large fields and active outputs stay on data. No runtime directory, environment
or compatibility link is migrated. The full goal remains in progress.
