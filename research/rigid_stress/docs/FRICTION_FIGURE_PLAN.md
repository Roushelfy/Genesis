# Final friction figure contract

This supplements the [active goal](QUALIFIED_STRESS_RANKING_GOAL.md) and the
[fixed field-task plan](FRICTION_COMMON_CONTROLS_20261019.md). It is a figure
specification, not completed artwork or reference qualification. Follow the
existing academic figure workflow and the inspected normal-figure style.

Required materials are actual complete mechanical histories, all point forces
and projected reference areas, final common-material complete fields, reference
spatial uncertainty and each compared mode's own admission. Fine125um reference
qualification and the final common-field exports are still missing. Data-reader
validation can proceed on the completed500um reference; final comparative
rendering waits for the matching qualified reference and fields. No new user
data or style decision is required by the existing repository-supplement scope.

## Evidence and preparation

`friction_plot_data --data-root RUN_ROOT --reference-run REFERENCE_RUN
--output NEW_DATA_PATH` prepares data through the established Slurm workflow.
It requires complete400-step normal/friction histories, both windows, the Shell
own-run audit and the reference current-momentum audit. It rehashes audited
frames, verifies their actual reaction totals against the journals, retains all
nonzero point forces and checks DPR/SCR motion/source agreement before combining
their contact-mechanics legend. It cannot itself qualify reference resolution.

The native map uses `applied_points_world_m`, where the semi-implicit wrench
actually acted; final predicted positions remain separately retained. Shell and
P2 use their recorded current pad points. Load densities divide actual point
forces by the declared projected **reference** areas. They are not silently
interpreted as current-surface Cauchy tractions. No force threshold, pressure
smoothing, polar removal or fitted footprint radius is introduced.

These maps show physical pad contact loads supplied to the recovery methods.
They do not show the boundary traction reconstructed by DPR/SCR. In particular,
assembled nodal force divided by an arbitrary dual area is not a substitute for
the actual native reconstruction quadrature. Any later claim separating contact
error from reconstruction error needs its own same-traction replay and exact
reconstruction/source audit; these contact maps alone cannot establish it.
The later [native reconstruction capture](NATIVE_RECONSTRUCTION_MATERIAL_20261019.md)
now retains exact integration points, independent assembly/core checks and
measured plate-side redistribution. Keep those reconstructed loads distinct
from this figure's actual physical pad loads; final continuum error attribution
still requires the separate reference and same-traction qualifications.

The source CSV retains every recorded physical step, actual upper/lower and
tangential reactions, body force, plate-gap closure, mass, inertia norm and
kinetic energy. Closure subtracts both initial clearances and includes pad
compression. Body-force curves retain each method's actual mass. Native stress
is admitted only at its requested equilibrium observations; complete mechanical
curves do not admit the separately failed every-step stress mode.

## Panel architecture

| Artifact / claim anchor | Panels and distinct role | Evidence | Encoding |
| --- | --- | --- | --- |
| `friction-complete-surface-stress` | Two rows: outer/inner stress; four columns: DPR/SCR/repaired250um Shell/P2 reference. Show the complete material domain. | Final q1 surface exports, identical material positions/reference tensors; q9/q25 supply separate peaks/errors. | Angular full-surface maps; one untrimmed shared sequential stress scale, tangent Mises in kPa. |
| `friction-normal-traction` | Upper/lower plate rows and native/Shell/reference columns. Show the actual normal load footprint. | Final actual point forces and reference-area weights. | Point maps with one shared nonnegative density scale; identical x/y bounds include every nonzero-force point. |
| `friction-tangent-traction` | Same plate/method layout. Show actual tangential support and direction. | Both tangential force components at all actual points. | Shared tangential-density magnitude scale; direction key or vector marks must retain both components without renormalizing force. |
| `friction-force-response` | (a) actual normal force against clearance-corrected closure; (b) opposing tangential resultant and body force against time. Distinguish compliance from sustained shear support. | Complete histories; separate actual masses and physical loading ramps. | Continuous trajectories with method colours and redundant line/marker cues; body forcing is dashed and explicitly distinguished from reaction. |

The common-scale field plot supports spatial comparison; it must not supply
numerical maxima from its single-centroid display. Peak locations, raw maxima,
common peaks and uncertainty remain in the independent q9/q25 tables. Point
traction maps show actual integration samples rather than interpolated pressure
images. Bounds are determined from the union of actual support, with no excluded
small force. Traction maps may combine native DPR/SCR only after the data check
passes; stress maps retain them as separate methods.

## Style, output and QA

Use the captured `pad-spatial-figure-v2/group0/figure/nature_mpl_style.py` from
the ranking data root. Target English repository supplements,183mm width,
at most170mm height per figure,5–7pt readable text and8pt lowercase panel labels.
Use the existing blue/vermillion/green method roles, distinct markers/line styles,
shared legends, labelled units and colour bars. Magnitude maps use a perceptually
ordered sequential palette. Signed component alternatives require a centred
diverging scale; no rainbow map or colour-only categorical distinction.

Retain plotting source, material passports with exact input hashes, panel specs,
CSV/NPZ plotted data, captions, body-text interpretation and alt text. Export
editable SVG/PDF, a final-size PNG and a preview, retaining editable vector text.
Dense field primitives may be rasterized with disclosed resolution. Verify
dimensions, text/font embedding, data coverage and colour-scale consistency;
inspect every rendered preview for clipping/overlap and final-size readability.
Uninspected artifacts remain drafts. Do not claim statistical confidence from
refinement envelopes or consecutive warm timing blocks.

After the actual data pass, captions and body text must state the strongest and
weakest spatial regions, where load concentrates, any method separation or
uncertainty-supported ties, continued creep, source-model limits and unavailable
full3D Shell components. These findings are intentionally not filled in before
the final reference and field comparisons complete.

## Executed data-reader control

`friction-plot-data500-control-v1` (728644) passes all five own-history/contact
data checks. It prepares1600 actual history rows, eight endpoint traction arrays
and exact source/output hashes, without rendering or claiming spatial accuracy.
At the friction endpoint, native DPR/SCR have166 nonzero-force samples among
8992 points, Shell3098 among33472, and the500um reference19126 among35400.
Their complete active x/y extents are2.024,2.349 and2.371mm, respectively;
all force samples remain in the data regardless of magnitude.

DPR and SCR actual contact positions, force arrays and area weights agree
within the retained checks. Their total tangential force is about-0.00747279N,
versus-0.00748522N for Shell and-0.00748074N for this500um reference. These
differences track the actual finite-mesh masses and must not be hidden by
drawing one nominal body-force curve. Native application and final point
positions differ by at most6.867nm at this endpoint; both are retained.
The data-preparation pass admits neither a coarse continuum reference nor
native every-step stress. Final traction extraction must be rerun with the
eventually qualified fine reference and linked to that qualification receipt.
