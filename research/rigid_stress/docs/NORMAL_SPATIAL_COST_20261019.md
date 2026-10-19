# Normal spatial maps and completed single-environment cost matrix

This update adds inspected full-surface stress/actual-pressure maps and admitted
sparse full/max costs to the [common-peak checkpoint](COMMON_PEAK_BATCH_RESULTS_20261019.md).
The normal B1 cost matrix is complete. Sustained-friction spatial qualification,
the full B4 cost/memory matrix and the final cross-case comparison remain open;
the [goal](QUALIFIED_STRESS_RANKING_GOAL.md) is not complete.

The [evidence packet](../evidence/20261019-qualified-ranking/normal-spatial-cost-v1/README.md)
retains175 verified payloads totaling7,057,473 bytes. Full trajectory fields,
journals and larger plotting arrays remain on data at their manifest paths.

## Complete-surface and actual-contact maps

Job728341 exports the already evaluated outer/inner tangent tensors at common
material face centroids. No comparison formula, field or trajectory is changed.
The display uses all reference faces, both poles and all azimuths. Centroid
display samples do not replace the9/25-point common peaks or volume-weighted
complete-field metrics. The angular chart is explicitly not equal-area.

Job728380 produces the inspected
[stress map](../evidence/20261019-qualified-ranking/normal-spatial-cost-v1/pad-spatial-figure-v2/group0/figure/normal-complete-surface-stress.svg)
and [actual-pressure map](../evidence/20261019-qualified-ranking/normal-spatial-cost-v1/pad-spatial-figure-v2/group0/figure/normal-actual-traction.svg).
Each uses a common untrimmed color scale. Stress is displayed as constant
centroid values on the reference angular cells, with no smoothing. Dense data
artists are rasterized; axes/text remain editable in SVG/PDF. The original
layout-QA failure728378 is retained; version2 moves panel labels and disables
triangle-edge antialiasing artifacts without changing scalar data. Both final
previews, programmatic text checks, editable text and embedded PDF fonts pass.

At0.1N, actual pad-pressure samples give:

| Mechanics | Upper active points | Lower active points | Upper maximum kPa | Lower maximum kPa |
| --- | ---: | ---: | ---: | ---: |
| Native DPR/SCR |68|98|8.2307|7.3977|
| Shell, repaired |1314|1784|6.5722|6.2091|
| P2 reference |132826|180404|6.4721|6.1496|

Pressure is each point's absolute vertical force divided by its own projected
quadrature area. All active samples fit within the plotted window; all six
plate reactions match0.1N within1e-6N. No pressure averaging or fitted footprint
is used. Marker size is a display choice, not physical contact area. The native
2mm recovery radius is not the actual force-bearing support. A similar complete
field or peak does not imply identical physical pressure sampling. Native and
Shell points are in their actual world projection; reference points use its
rest material projection at this normal equilibrium.

The companion [response/common-peak figure](../evidence/20261019-qualified-ranking/common-peaks-batch-v1/pad-response-figure-v2/group0/figure/normal-response-peaks.svg)
already reports clearance-corrected force/closure and conditional peak ranges.
These normal figures do not establish a sustained-friction ranking.

## Primary sparse cost and matched interpretation

Audit728381 passes all six sparse runs from728245 groups2/3. Each exact run's
complete800-step history, mechanical audit, observations, endpoint fields,
disabled phase clocks and whole-interval accounting pass. After200 warmup
steps, three200-step blocks observe stress every20 steps, including endpoints.
The following are synchronized whole-workflow ms/step on RTX PRO6000 Blackwell,
including mechanics, requested recovery, transfers, acceptance and journal/
endpoint I/O; they are not production-only kernel latency.

| Method | Sparse full median | Full min--max | Sparse max median | Max min--max |
| --- | ---: | ---: | ---: | ---: |
| DPR |16.629|16.614--16.767|16.331|16.276--16.341|
| SCR |82.728|82.631--82.825|82.368|82.330--82.581|
| Shell, repaired |311.257|251.709--458.375|312.108|251.301--458.774|

For comparison, every-step full medians are76.916/1400.682/314.464ms for
DPR/SCR/Shell. Sparse observation materially reduces native recovery work;
Shell still integrates its elastic/contact state every step. Shell's repeated
blocks vary substantially, and small full/max or cadence differences are not
established speedups. These are consecutive warm blocks, not independent-process
statistical replicates. Complete repeats, setup time, observation counts,
transfers and hardware identities remain in the receipts. Intrusive memory
profiles remain separate and cannot substitute for these admitted runs.

## Measured reference-evaluation cost and remaining work

Profile728379 uses an admitted coarse sustained-friction displacement and
performs five calls per operation without changing simulation. Gradient
interpolation takes0.203--0.217s, complete force/energy0.339--0.378s, and direct
nonlinear correction0.378--0.418s. These intrusive CPU samples indicate repeated
gradient work is worth investigating before further long refinements. They do
not prove a trajectory speedup or authorize looser stopping criteria. No cache
or constitutive approximation was introduced by the profile.

The fine500um/two-wall direct-correction trajectory728256 remains live at this
checkpoint. Its unfinished history is not a qualified spatial control. B4
SCR/Shell primary costs and the remaining separate memory profiles are also
unfinished. Complete these, refine the sustained-friction reference as needed,
then compute its qualified fields/peaks and assemble the final per-case
error/uncertainty/cost/stability comparison.
