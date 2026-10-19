# Complete-field and physical-load figure rendering

The figure plan remains [the fixed friction contract](FRICTION_FIGURE_PLAN.md).
The rendering controls pass15 checks and all four exported previews have been
inspected. The rendering implementation uses complete qualified normal fields and
mechanically admitted coarse friction data for separate controls. These controls
do not qualify a fine friction reference or grant a method-error ranking.

## Exact scientific coverage

`stvk_figure_render` retains every reference face in both outer/inner material
maps, with identical full tensor and material-position checks. One centroid
per reference face is display data; numerical errors and raw/common maxima
remain in the independent q9/q25 tables. Both angular poles and all azimuths
are displayed, without percentile clipping or polar exclusion.

Contact maps retain every sample with any nonzero force component, including
purely tangential force if present. Densities are independently recomputed
from each actual point force and its projected reference area. Native maps use
actual applied wrench points, with later predicted points separately retained.
The common support window includes every active point across all methods and
both plates. Separate normal/tangential shared scales use the actual maxima.
Net-force arrows use the same physical scale and retain both signed components;
they are direction keys for the resultant, not local vector fields. Original
force arrays remain available for each method, including SCR.

The response panels retain all1600 rows across four400-step trajectories.
DPR/SCR curves combine only after all recorded mechanics agree under the
established data-reader bounds. Initial approach and both loading levels remain
visible. Closure already subtracts both initial clearances; each body-force
curve retains its actual finite-mesh mass. Complete mechanics do not admit the
separately failed native every-step stress observations.

## Rendering and admission checks

`check_stvk_figure_rendering --data-root RUN_ROOT --style CAPTURED_STYLE
--output NEW_DATA_PATH` is an explicitly fixed renderer control. It checks the
existing normal qualification and original hashes, all eight coarse endpoint
traction arrays, complete histories, missing/repeated steps, nonfinite force
and altered native mass. It exports four clearly named control artifacts.
The normal surface reference is125um; coarse physical friction maps/curves
are labelled P2,500um and remain outside final fine-reference accuracy claims.

The first control729875 stopped before plotting because a force tolerance in N
had incorrectly been reused for derived Pa densities. The correction compares
original dimensional position/force/area primitives under the unchanged reader
bounds and independently recomputes each density; no physical gate was relaxed.
The original failure and captured source are retained. Subsequent export QA
retains all detected text issues before admission.

Every figure has183mm width and height below170mm, editable SVG text, embedded
TrueType PDF fonts,600dpi PNG,150dpi preview and450dpi dense data artists.
SVG dimensions, text presence, PDF embedding, PNG dimensions/DPI and rendered
text boxes are checked. Per-figure export QA is retained even when a later
check fails. Visual preview inspection is required separately before accepting
layout; programmatic success alone does not grant submission readiness.

## Guarded final entrypoint

`plot_stvk_pad_figures --data-root RUN_ROOT --reference-qualification QUALIFIED_JSON
--ranking RANKING_JSON --force-data FORCE_DATA_JSON --style CAPTURED_STYLE
--output NEW_DATA_PATH` requires explicit fine-reference qualification, all54
admitted error rows, eighteen conditional ranking records and the same source
frames, meshes and complete candidate histories in the actual force extraction.
All cited files are rehashed. The q1 display must match the q25 ranking frames;
traction endpoints must match the independent qualified/admitted fields.
There is no skip-qualification option. A qualified normal-only reference is
an intentional negative input: it must produce a failed receipt and no final
artwork. Final rendering and its positive admission remain pending until the
actual fine reference, candidate fields, ranking and matching force extraction
complete.

The output packet retains a claim anchor, material passports, panel roles,
complete plotted arrays/CSV, source/style, captions and export QA. Fine graphics
still need visual inspection, accessibility/alt-text and quantitative body-text
handoff based on their actual profiles and the uncertainty-aware error table.
No continuum bound, statistical confidence, universal method order or unmatched
production-cost/error pairing is inferred from the image colours.

## Terminal controls and queued final render

Control729932 passes all15 checks and all four183mm previews pass independent
visual inspection. Both initial geometry warnings are preserved:729877/729882
retain the time-axis tick warning at0.45s outside the observed0.4s history.
The fix explicitly declares0–0.4s ticks and retains every actual observation.
No QA issue is suppressed. Native/Shell/500um P2 traction maps retain166/3098/
19126 force-bearing samples across both plates, with actual upper/lower
resultants in their profiles. Categorical curves retain marker and line-style
cues, and every magnitude panel uses a labelled sequential scale.

Negative729930 rejects the normal-only qualification with no final figures.
Final729938 waits afterany on the54-row ranking729835 and matching fine-force
extractor729519, then runs its own independent gates. It cannot admit graphics
merely because its dependencies terminated. The three renderer/entrypoint/control
module hashes match the predeclared figure contract and the queued source
snapshot. Final fine artwork, its visual inspection and quantitative narrative
remain pending. The compact packet below retains source variants, terminal
failures, preview inspection and the immutable queued plan.


The [renderer packet](../evidence/20261019-qualified-ranking/stvk-figure-render-controls-v1/README.md) retains123 compact payloads/1,750,818 bytes, four inspected control previews and exact original/encoded hashes. Full plotted arrays and vector/high-resolution exports remain on data with external identities; live fine outputs and final scientific artwork are excluded.
