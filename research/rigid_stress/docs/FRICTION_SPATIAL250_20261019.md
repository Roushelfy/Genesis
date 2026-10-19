# Nonlinear250um reference and plotting-data checkpoint

The250um, two-wall-layer nonlinear trajectory passes its complete mechanical
history and independent current-position momentum checks. Its500-to250um field
changes still exceed the predeclared2% target at both the normal and sustained
friction endpoints. Passing peaks do not override this failure. The final
friction reference and three-method ranking remain in progress.

The [packet](../evidence/20261019-qualified-ranking/friction-spatial250-v1/README.md)
contains97 verified metadata/source/log payloads totaling1,088,827 bytes.
All four analyzed reference fields were rehashed against their own momentum
and comparison receipts. A separate metadata audit rehashes45 unique inputs
and outputs, including the actual plotting arrays; large arrays remain on data.

## Complete trajectory and spatial changes

`solid-pad-stvk-spatial250-cache-v1` (728389) completes400 physical1ms steps,
with two51-sample equilibrium windows. It retains the declared free egg,
StVK material, finite-area physical pads,0.1N normal load and lagged smooth
friction law. No DOF is fixed and no velocity is cleared. Both independent
endpoint momentum checks pass: relative force balance is1.013e-9/5.182e-10
and current-position moment balance is4.578e-10/6.163e-10. The complete nodal,
normal/friction, point-velocity and omitted-boundary checks also pass.

The reference has1,004,190 displacement DOFs and134800 contact points.
Its setup takes320.361s and the dynamics7014.696s in this one execution.
This cached run and the preceding uncached500um run are different resolutions
and configurations, so their times are not a cache-speedup measurement.

`stvk-spatial250-fields-v1` (728678) evaluates both endpoints at9/25 surface
points and depth4, with physical-volume weighting and complete material
columns. Both meshes retain two wall layers; this is the surface refinement
control. The preceding one-wall coarse-to500um comparison changed both surface
and wall resolution and is not another isolated surface-refinement pair.

Entries below are **percent changes** at q25, normalized by the250um result.
The geometry, native footprint or acceptance targets were not retuned.

| Endpoint | Tangent | Membrane | Bending | Outer | Inner | Raw VM | Raw tension | Common VM | Common tension |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 2.74909 | 1.90859 | 2.39879 | 2.01110 | 2.28628 | 1.67512 | 2.52839 | 1.56204 | 2.50411 |
| Sustained friction | 2.74895 | 1.90844 | 2.39883 | 2.01113 | 2.28619 | 1.67470 | 2.51931 | 1.56123 | 2.51242 |

Tangent RMS differences are251.001/251.009Pa. Both complete-field gates fail;
raw/common peaks pass5%. The raw VM maximum remains at the lower outer pole
on both meshes. Common peaks retain their own positions and tangent-only
definition in the receipts; they are not interchangeable with raw full3D peaks.

Changing angular sampling from9 to25 changes the field-contrast metrics by
at most5.877e-5 in absolute relative-error units, or0.005877 percentage points.
The actual common peak values change by up to0.055729%. These sampling effects
are retained separately from the mesh changes. They cannot turn2.749% into
a passing2% result. Full3D-only normal and transverse-shear changes are about
38.21% and9.48%; their smaller absolute components remain separately reported,
with no corresponding independent Shell component rank.

## Audited figure inputs

`friction-plot-data500-control-v1` (728644) passes all five checks for actual
history and traction totals, including native DPR/SCR contact-data agreement.
It prepares1600 step rows and eight endpoint traction arrays. Both500um reference
fields and both250um Shell fields match their own momentum/field audit hashes.
Native endpoints require valid requested observations. This does not admit the
separately failed native every-step stress configuration.

The [figure contract](FRICTION_FIGURE_PLAN.md) fixes the four intended artifacts,
complete-domain coverage, shared scales and export/visual QA before rendering.
The reader keeps every nonzero point force, actual wrench application positions,
projected reference areas, clearance-corrected closure and each method's actual
body force. These are physical pad contact loads, not native reconstructed
boundary tractions. They cannot alone separate contact from reconstruction error.
The primary same-traction/source-reconstruction evidence still needs an explicit
completion audit before the final error decomposition is claimed.

No final friction figure is rendered here. The data-reader control uses the
completed500um reference and must be rerun against the eventually qualified
fine reference. Finite-mesh masses and continued frictional creep remain visible;
one nominal body-force curve must not hide their small actual load differences.

## Remaining work and reproduction

The250um three-wall reference (728464),125um two-/three-wall references
(728589), and500um0.3/1N normal material controls (728615) remain active at this
checkpoint. Their active outputs are not published or admitted. The existing
fixed wall/surface/joint and method field-task plans remain unchanged. Submit
them only after terminal scheduler status, complete histories and their own
momentum checks have been verified.

Use captured `tasks.json` with the established rtx-mid / gs-srun submission
workflow and fresh data-side destinations. The reader command is
`friction_plot_data --data-root RUN_ROOT --reference-run ADMITTED_REFERENCE_RUN
--output NEW_DATA_PATH`. The captured `meta/audit-friction-spatial250-v1.py`
rechecks scheduler/metadata/scalar results and hashes; it does not recompute
the scientific solves. Its output path is exclusive-create.

Source, small documentation and this selected evidence packet remain in the
workspace. All full fields, journals, CSV/NPZ intermediates and active outputs
stay under the established data root. No environment, live writer or
compatibility symlink is migrated. The overall goal is not complete.
