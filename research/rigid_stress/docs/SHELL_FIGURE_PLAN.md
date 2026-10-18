# Shell comparison figure argument

Repository report figures, English labels, 183 mm width, at most 170 mm high.
All required materials are executed immutable receipts/NPZ fields in the data
run root; no external artwork, generated images or additional user input is
needed. Plot sample positions directly; no interpolated field or fabricated
uncertainty. Three-repeat whiskers show the observed range, not confidence bounds.

| Figure | Claim and panels | Evidence | Limits carried into caption |
| --- | --- | --- | --- |
| mechanics | Shared finite traction; reference/shell compatible midsurface tensor maps; tensor/resultant discrepancy; analytic sphere error; prescribed-load cost and algebraic/model discrepancy | prepare-loads-v2, analyse-prescribed-v1, analytic-sphere-v1, static-cost-v1, static-inverse-tuning-v1, collect | Faceted meshes differ; compatible shell tangent norm and native volume norm differ; volume peak is not mesh converged. |
| contact | Coarse interior/edge/vertex geometry and impulses; dt sensitivity after linear convergence; measured residual; mechanical energy; frictional slide/release | diagnostic-controls-v2, contact-audit-v1, contact-ledger-v1, press-controls-v3 | Deformed poststep geometry; coupling impulses precede integration; coarse misses are distinct from flexible-sheet height; lagged reaction and energy splitting are declared. |
| consistency | Full raw application peaks without clipping; fresh restarts; public/private restore ablation; per-condition sampled peak | full-shell-history-v1, peak-traces-v2, fresh-restarts-v1, cache-ablation-v2 | Dynamic contact loads differ; invariants compare magnitudes, not tensor error; no shell accuracy-qualified rollout established; private restore is diagnostic only. |
| performance | Full-field aggregate rate; paired native increment; sampled process memory; grouped kernel cost | validated-native-v2, paired-normal-v1, paired-diagnostics-v1, profiles-v1 | Native rates pass numerical gates only; shell rates are unqualified diagnostics; matched hardware cohorts, not identical GPU UUID across cohorts; profiles/memory exclude headline timing. |

Use Wong colours plus distinct markers/line styles; sequential viridis for
nonnegative fields. Shared units/scales only for mechanically compatible maps.
Export editable text SVG, font-embedded PDF, 450 dpi PNG and 150 dpi preview.
Check bounding boxes, typography, font embedding, dimensions/DPI and all actual
PNG/PDF rasters. The report supplies captions, interpretation and provenance.
