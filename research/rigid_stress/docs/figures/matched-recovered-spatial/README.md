# Matched physical-time spatial fields

Claim anchor: `unresolved-contact-field-after-resolving-more-nodes`.
This repository-supplement figure supports the matched32.5ms comparison in
`../../PHYSICAL_MATCHED_SPATIAL_20261015.md`. It compares480/3577/8433-tet
full hollow-egg states under fixed original physical inputs. It does not claim
a completed fine trajectory, converged contact patch or singularity proof.

The editable source is `../../../plot_recovered_spatial_state.py`, with exact
arguments and copied style in runtime `franka-matched-spatial-figure-v3`.
`figure_contract.json` records panel roles and common logarithmic colour limits;
`plotted_values.csv` retains the physical units and raw extrema. Figure data come
from `franka-matched-recovered-spatial-v2`; separate version3 additionally audits
objective nodal displacement and actual corner Green-Lagrange strain. Original
saved contact forces and precision-aware reconstructed forces are separately
retained in those comparisons.

SVG text is editable and PDF TrueType fonts are embedded. The final PNG is
3242x2746 pixels at450dpi,183x155mm. `preview.png` is the180dpi rendered preview.
`rendered_qa.json` records the completed visual/export review; the immutable
producer metadata retains its earlier unreviewed-export flag.

The top row shows unaveraged element corner-maximum envelopes on reference x=0
sections, with a common0.16mm hotspot window. These are envelopes, not exact
cut-plane stresses. The lower row uses a common tangential x-z projection of the
negative-y contact. That side is selected from the finest globally searched
peak. Marker size does not represent physical area, and force/lumped-area
density does not define an independently resolved physical contact boundary.
The Cividis logarithmic colour scales are shared within each quantity, with
distinct stress and traction-density bars. Statistical replicates, fitted orders
and confidence intervals are inapplicable to these three deterministic fields.

Body-text handoff: the finer contact has more active vertices, yet the final
tensor, energy and raw-peak changes remain large. The strongest concentration is
at the finest negative-y contact; common interior probes do not replace that
global search. Near-coincident small dots are projected contact vertices, not
smoothed pressure measurements. Use `caption.md` for the full caption and retain
the single-state/history and continuum-error limitations in the report.
