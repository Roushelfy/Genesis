# Inspected diagnostic figure checkpoint

**The reference goal is incomplete.** These files are byte copies of the three
existing Slurm-rendered figures, not new experiments or refreshed final results.
The [captions and interpretation](../../../docs/QIPC_REFERENCE_FIGURE_CAPTIONS.md)
and [claim contracts](../../../docs/QIPC_REFERENCE_FIGURE_SPEC.md) explain scopes.
The cost figure predates some later-completed Shell/memory points. Reference
spatial/timestep uncertainty prevents using these plots as an accuracy ranking.

| Figure | Preview | Editable source and exports |
| --- | --- | --- |
| Full-field cost snapshot | [PNG](new-full-field-cost_preview.png) | [SVG](new-full-field-cost.svg), [PDF](new-full-field-cost.pdf), [450dpi PNG](new-full-field-cost.png) |
| Complete6s prescribed diagnostic | [PNG](complete-prescribed-diagnostic_preview.png) | [SVG](complete-prescribed-diagnostic.svg), [PDF](complete-prescribed-diagnostic.pdf), [450dpi PNG](complete-prescribed-diagnostic.png) |
| Shared outer tangent maps | [PNG](shared-outer-tangent-maps_preview.png) | [SVG](shared-outer-tangent-maps.svg), [PDF](shared-outer-tangent-maps.pdf), [450dpi PNG](shared-outer-tangent-maps.png) |

The source plots use one common colour range; the map upper bound is computed
from both inner and outer sampled surfaces at the selected times, while only
outer samples are displayed. Raw full3D/global peaks are separate observables.
No samples are clipped or smoothed. Relative time-series gaps retain the source
denominator/log-coordinate rules described in the captions.

`figure-data.npz` contains the plotted line/scatter arrays. `source/` contains
the exact captured Matplotlib renderer, style helper and support package files.
`original-tasks.json` and `original-submission.json` preserve actual execution;
`rendering-result.json` identifies the raw inputs. Original inputs still match
their recorded SHA256 values. Re-running the renderer requires those full raw
histories at the recorded data paths and a fresh writable output directory,
inside the existing Slurm/container workflow. This compact checkpoint does not
claim a self-contained final data/source closure.

`rendered-text-qa.json` and `export-qa.json` retain the original executed checks:
183mm width, editable SVG text, embedded non-Type3 PDF fonts and450dpi PNG.
All three existing previews were viewed again during publication; details and
limitations are in `publication-check.json`. No figure was rendered or changed
on the login node. `manifest.json` hashes every selected publication file.
