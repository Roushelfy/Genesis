# Shell PR22 comparison evidence

Executed2026-10-18. Read the
[decision/report](../../../docs/SHELL_COMPARISON_RESULTS_20261018.md),
[reproduction guide](../../../docs/RUNNING_SHELL_COMPARISON.md),
[completion audit](../../../docs/SHELL_COMPARISON_AUDIT.md) and
[figure argument plan](../../../docs/SHELL_FIGURE_PLAN.md).

Native source: `7fe068267e09ff97c09ffac4904a2f47914596d9`.
Original external source: `fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab`.
No native engine change or external production repair is included. A separate
read-only diagnostic patch records residuals and coupling impulses. Original
normal timing adds the declared device full-field observer; private restore and
prescribed-load harnesses are separate diagnostics.

* `receipts/`: unmodified result JSONs, including source/import/environment and
  application scopes; `native-curved-v1` retains local-traction/source-shift audits.
* `raw/`: selected **byte-identical** meshes, loads, control fields, static fields,
  contact ledger and four dense application peak archives. Each peak archive
  retains the full field at every environment's raw sampled maximum, primitive,
  sample time/phase, material/world point and deformed vertices or link pose.
* `inputs/`: the exact accepted eight-condition bank, reused without changes.
* `derived/`: final collector tables/JSON and compact plotting inputs. Impulses
  in `plot-inputs` are summed over vertices, preserving environment/time/vector
  axes. These files are labeled **derived**, not original trajectories.
* `figures/`: measured mechanics/contact/consistency/performance panels, editable
  text SVG, embedded-font PDF,450dpi PNG and previews. Captions/placement and
  interpretation are in the report. Place these with their corresponding
  mechanics, contact, consistency and practical-cost sections.
* `figures/qa/`: actual PDF raster previews and export size/font/DPI checks;
  `rendered-text-qa.json` has no detected overlaps/clipping. All four final PDF
  rasters and the original/changed PNG previews were visually inspected. There
  is no Type3 font; TrueType fonts are embedded; width183mm, heights138–144mm.
* `reproduction.tar.gz`:43standard campaign captures plus the initial frozen
  tools snapshot and manual launchers, with deduplicated executed Python
  blobs, task JSONs, source/asset inventories, patches, original task
  and Slurm logs, and terminal accounting. `analysis-index.json` maps each exact
  snapshot path to its blob SHA256. Use `restore_snapshot.py` from the running
  guide; no unsafe archive extraction is needed to restore code.
* `source-closure.json`: source archive identity/counts.
* `data-artifacts.json`: hashes/sizes/paths of larger originals retained under
  the established data run root. GB-scale raw traces stay there; their complete
  compact kernel/API/copy summaries are in this packet.
* `manifest.json`: SHA256 and byte size of every other packet payload. Scoped
  Git attributes preserve CSV line endings and generated SVG/JSON bytes.

Primary full-output native rates come from `validated-native-v2`, with all
36stress repeats and warmup counters valid. Earlier native v1 rates omit exported
counters and remain attempted diagnostics. All shell rates are unqualified
physical-accuracy diagnostics. Memory/profile/capacity scopes are separate;
neither final finiteness nor linear convergence is substituted for contact
accuracy. Matched hardware cohorts have individual GPU UUIDs and same primary
software, not a fabricated shared allocation.

Preserved failures include the circular-import adapter, missing scikit-fem,
precision/OBJ/static attempts, original shell instabilities, OOM versus Jacobian
capacity guards and first figure layout clipping. Earlier `fell_through` JSON
fields are height markers; the final deformed-triangle audit supplies geometric
evidence. No failed/unresolved case is silently upgraded to a success.

The underlying continuum/contact peak is not mesh converged, the shell fiber
recovery is approximate and neither method is calibrated for physical damage.
These are report limitations, not omitted evidence or promises of a new sweep.
