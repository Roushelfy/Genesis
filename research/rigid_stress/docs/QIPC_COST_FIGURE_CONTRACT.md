# Full/max, memory, activation and profile figure contract

Claim anchor: `qipc-cost-scopes-20261018`. Placement: the report's cost section.
Purpose: explain measured complete-scene output cost and practical setup/memory
limits without conflating numerical completion with physical stress accuracy.

Material passport: the completed `F/cost-consolidated-v2/group0/cost/result.json`,
its hashed original workers and six full kernel CSVs. All plotted values already
exist. The new renderer does not run an engine or infer a passing error band.

| Panel | Evidence and distinct message |
| --- | --- |
| a | Three-repeat full/max10-substep complete-scene rates: DPR/SCR atB4096 and the matched Shell surface0/surface2 pair atB32. Show all repeats and median/min/max; the useful measured batch differs and is explicit. Native valid-step counters and Shell diagnostic finite rates retain different admission scopes. |
| b | Intrusively sampled process peak memory versus actual tested batch size, with separate DPR/SCR and stock/fine Shell curves. Shell stock uses one physical substep, fine uses ten. Lines only join observations; no maximum-capacity or equal-accuracy extrapolation. |
| c | Eight isolated fresh-cache initialization/build/first-two-step totals, with exact method/batch labels. One activation per setting, no invented repeats/error bars. Imports precede the timer. |
| d | The three largest individual kernels in each of six loaded intrusive traces, as fractions of that trace's summed kernel duration. Full exact kernel names remain in source CSV. Durations may overlap and do not represent scene latency or throughput. |

Method: Matplotlib with the existing academic figure style. Width183mm,
height160mm, editable SVG/PDF,450dpi PNG and150dpi preview. Actual plotted CSV,
source identities, exact renderer and rendered/export/visual QA accompany it.
Use method colors plus marker/line differences; rank kernels with gray levels.
Keep legends clear of data. No smoothing, omitted failed settings, fitted
capacity curve, added scalar error budget or fastest-in-accuracy-band claim.

Limit: the unresolved physical-reference uncertainty is not a timing error bar.
The report and caption must state Shell's absent invalid-step counter and the
full-last-substep versus max-all-physical-substeps output contract. Previous
primary rates keep their original source/observer identities; the separately
paired read-only max observer does not silently replace them.
