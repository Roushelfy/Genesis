# Native pad histories, source sensitivity and reference refinement

This checkpoint preserves exact selected receipts and executed native adapter
sources in [manifest.json](manifest.json). Original full fields, journals,
source captures and logs remain at the listed data-side paths. It does not
close the qualified-ranking goal.

The native seven-velocity pad boundary condition passes five analytic controls.
The native semi-implicit position update then passes every step of normal and
sustained-friction histories; the default free-body midpoint negative control
fails82 pressure/friction steps despite passing its endpoint windows. The
original native external-wrench guard is preserved. An independent observer
model receives the complete measured surface packet at observation times.
DPR/SCR both accept the normal/shear fields and all five normal-cycle loads.

The complete-source audit checks force/moment compression, frozen-point moment
error and unchanged motion with observation. Nevertheless, the wrench-only
source compression fails the shear timestep field gate: a5.06e-7 change in
the actual force field switches a sparse support point and changes the sparse
load field17.33%, producing about2% tangent-field change. This is a failed
adapter configuration for qualified shear errors and costs. The explicit
full-source replacement is a separate ongoing control.

The `native-pad-field-sensitivity-v1/*/repeat` path is a historical launch name:
its inputs change friction0.3 to0, so its7.07% SCR difference is parameter
sensitivity, not a same-configuration repeat failure. The corrected normal
comparison in `native-shell-pad-normal-fields-v2` uses friction0 for native
and Shell. Its values remain provisional while joint reference uncertainty
and the final source representation are checked.

Surface250-to125um and wall2-to3 adjacent reference comparisons meet the
individual2% field/5% peak targets. Their joint effect and remaining refinement
trend are not inferred from those pairs alone. The published Shell cycle and
shear-return receipts add continuous mechanical and field stability evidence.
See [the progress report](../../../docs/QUALIFIED_STRESS_PROGRESS_20261019.md)
for exact values, failed launches and open work.

Use each captured `tasks.json` with the established Slurm submitter, pinned
engine/research sources and a fresh data-side output directory. Native
controls use the existing `genesis-shell-contact` environment; independent
audits/field comparisons use `rigid-stress-physical`. All scientific execution
uses `rtx-mid` and `GENESIS_IMAGE_VER=1_26 gs-srun`.
