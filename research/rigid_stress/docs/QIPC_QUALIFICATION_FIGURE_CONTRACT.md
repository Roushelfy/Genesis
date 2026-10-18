# Reference-qualification figure contract

Claim anchor: `qipc-reference-uncertainty-20261018`.
Placement: the working report's reference uncertainty and numerical diagnostics.

Main claim: completed reference histories remain spatially and temporally
sensitive, even where a stopping/smoothing contrast or the net force residual is
small. No physical accuracy ranking is implied.

| Panel | Material passport / evidence | Distinct role |
| --- | --- | --- |
| a | Original `reference-pairs-v2`, fallback fixed-wall2 pair and friction pair; full600-time tangent norms | Compare complete field contrasts with the existing5% field target. Fixed-wall surface3/surface2 uses surface2 as denominator; other contrasts use coarse surface1/wall4. |
| b | Fallback validation actual per-finger reaction audit, full600 times | Show force/moment sensitivity without opposing-finger cancellation, with the existing2% reaction target. |
| c | Full raw physical-step journals for coarse, surface2/wall4, surface3/wall2 and half-dt | Preserve global VM peaks across all loading/release times. No display-grid maximum or phase cut replaces these recorded discrete peaks. |
| d | The same raw journals, signed egg-energy-minus-recorded-contact-work | Expose numerical dissipation. These ledgers are descriptive and do not establish a complete conservation law. |

All sources already exist and have completed Slurm receipts. Input identities,
actual plotted CSV values, exact renderer and QA accompany the outputs. Curves
are full recorded histories without smoothing, outlier deletion or clipping.
Contrasts are separate sensitivity measurements, not confidence intervals or an
additive error budget. The fixed-wall contrast has a different denominator and
is labeled explicitly. A small nonlinear-stopping contrast is documented in the
report rather than compressed into a misleading near-zero graphical mark.

Method: Matplotlib using the established academic-figure-workflow style and
Slurm `rtx-mid` execution. Target width183mm, height148mm, editable SVG/PDF,
450dpi PNG and150dpi preview. Check rendered text bounds, editable SVG text,
PDF font embedding, physical dimensions and the actual preview. Use separate
markers/line styles with accessible colors; retain raw values and units.

Open limits: the reference is unqualified; incomplete wall8/half-band histories,
the sparse adaptive pilot and capped histories are excluded from this complete
prescribed-history figure. No final-goal or damage-calibration claim is made.
