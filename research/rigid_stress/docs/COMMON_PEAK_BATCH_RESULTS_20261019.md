# Common peaks, batch conformance and primary cost checkpoint

The [later spatial/cost update](NORMAL_SPATIAL_COST_20261019.md) completes
normal field/traction maps and B1 sparse primary costs. Status below describes
this earlier checkpoint; friction spatial qualification and B4/memory remain open.

The normal physical-pad comparison now has qualified common-location peak
changes, actual four-environment motion controls and admitted single-environment
every-step full/max wall times. These extend the
[normal reference report](NORMAL_PAD_QUALIFICATION_20261019.md). The
[complete goal](QUALIFIED_STRESS_RANKING_GOAL.md) remains in progress: friction
spatial refinement, all matched sparse/batched costs, independent memory
profiles and common-scale stress/traction maps are still required.

The [checkpoint](../evidence/20261019-qualified-ranking/common-peaks-batch-v1/README.md)
retains executed sources, terminal receipts and the response/peak figure.
Full fields and journals remain under the data root recorded in DATA_LOCATION.md.

## Common-location peaks and full3D components

Job728274 (`pad-common-peak-qualification-v2`) passes the unchanged5% spatial
and1% contact/angular peak-change targets. Independent125um wall, three-wall
surface and joint mesh changes, contact orders6/10 and angular samples9/25
per face are retained. Both one-sided traces of every through-wall partition
and both wall boundaries contribute; extrema are neither smoothed nor trimmed.
The common invariant embeds the2D tangent tensor into3D with zero normal/shear
components. It is distinct from each method's raw unsmoothed full3D maximum.

| Load N | DPR Mises % | SCR Mises % | Shell Mises % | DPR tension % | SCR tension % | Shell tension % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|0.1|2.3388|2.3754|3.0774|5.5734|5.0365|14.8518|
|0.3|0.3097|0.4063|3.0503|0.7434|0.1179|14.9677|
|1.0|1.6127|1.7508|2.7966|1.7381|2.4031|15.6670|

The observed reference envelopes for Mises are0.4607/0.3408/0.2336%, and for
tension0.6084/0.4209/0.2451%. Method angular-sampling changes are propagated
separately. Conditional ranges assume these empirical changes bound unresolved
error; they are not certified continuum bounds or statistical intervals.
At0.1N all three Mises ranges overlap. At0.3/1N native Mises errors separate
from Shell, while DPR/SCR remain unresolved. Native tensile errors separate
from Shell at all three loads; DPR/SCR overlap at0.1/0.3N. At1N the fixed2mm
tensile intervals favor DPR, but the independently observed footprint-radius
sensitivity is larger, so this is not parameter-robust method superiority.
Exact peak positions and absolute peak values accompany the CSV/JSON.

True3D-only components are reported separately. Native normal-stress relative
errors are117.4--126.5% for DPR and103.0--113.8% for SCR; transverse-shear-vector
errors are38.0--38.7% for both. At0.1N their normal RMS errors are358.65/322.79Pa,
and shear RMS errors328.35/328.81Pa. The reference normal/transverse components
account for1.45--1.81% of total squared Frobenius stress. Small contribution to
the whole norm does not establish local component accuracy. These component
values have not received a separate spatial-uncertainty ranking. Shell has
no independent3D normal/transverse stress here and receives no such error rank.

The editable `pad-response-figure-v2` figure (728336) plots the force/closure
response and both conditional common-peak comparisons. The response uses actual
upper reactions and subtracts both initial22um clearances; closure includes
pad compression as well as egg deformation. Loading/unloading observations
are connected only as equilibrium guides. DPR/SCR share the same motion;
the reference starts at zero clearance. Text/layout checks and visual review
pass. Version1's off-axis tick-label QA failure is retained; exported visible
preview pixels are unchanged by the explicit tick fix.

## Actual batching and observer controls

`native-pad-batch-control-v1` (728252) first checks four distinct saved normal/
friction packets, permutation, an empty environment and full/max observers.
These are input controls only. The subsequent motion campaigns728259 and728269
actually integrate four independent eggs at0.1/0.3/1/0.1N for400 steps each.
Every motion history, equilibrium window and requested stress observation passes.
The native adapter solves each environment's pad mechanics separately, then
executes the existing batched native integration and recovery. Its serial pad
adapter cost remains included; this is not a pure GPU-kernel scaling claim.

Independent audits728272/728278/728281 compare every endpoint with the admitted
single-environment fields at the matched loads. Maximum complete tensor relative
differences are1.4452e-11 DPR,4.2170e-9 SCR and5.7009e-8 Shell. Native source
packets agree with their own motion-frame forces and moments. Shell also passes
the full-history current-position moment criterion. No fields are broadcast
from one environment to simulate a batch.

Shell's same-frozen-state paired full/max check (728220) passes all400 steps,
maximum peak discrepancy3.153e-16. The earlier independent-trajectory strict
1e-8 peak audit failure is retained; it is not reclassified or loosened.
Sparse preflight audit728229 passes but its segmented times are not primary
whole-workflow rates.

## Admitted primary wall times

Job728337 audits six completed runs from groups0/1 of `pad-primary-cost-v1`
(728245): each has800 actual steps,200 warmup and three consecutive200-step
measurements. Its own mechanical audit, full history, observation cadence,
complete-field/peak conformance and whole-interval accounting pass. Hardware is
RTX PRO6000 Blackwell Server Edition. Per-phase clocks are disabled; each
synchronized interval includes mechanics, recovery, host transfers, mechanical
checks, journal I/O and endpoint field/certificate writing.

| Method | Full median ms/step | Full min--max | Max median ms/step | Max min--max |
| --- | ---: | ---: | ---: | ---: |
| DPR |76.916|76.801--77.142|68.331|68.240--68.569|
| SCR |1400.682|1400.353--1403.240|1384.356|1384.193--1385.525|
| Shell, repaired |314.464|255.251--462.881|309.917|249.458--456.545|

These are consecutive warm blocks of the declared research workflow, not
independent-process statistical replicates or production-only solver latency.
Shell's repeat range is substantial and must remain visible. Max output retains
all the declared numerical/mechanical acceptance work; eliminating field output
does not eliminate that work. Initial setup and individual block data are in
the receipts. Sparse cadence20 and homogeneous B4 full/max primary campaigns
are separate and ongoing; no complete error-versus-cost table is claimed yet.

`pad_memory_profile` starts the existing process sampler before importing the
simulator and runs each B1/B4 full/max workload in the same PID. The dedicated
728294 campaign includes build and complete normal history. Its cost files
explicitly set `primary_timing=false`; sampled GPU process peaks and Linux
process RSS are separate from PyTorch endpoint allocations. Sampling can miss
short-lived peaks. Pending results are not accepted memory measurements.

The first terminal B1 DPR full/max profiles both sample2952MiB GPU process
peaks, with2074/2034 observations. Their own200-step histories and subsequent
source/field audits pass. CPU max RSS is2,946,724/2,951,392KiB. These are
single separate process profiles; other methods and B4 remain outstanding.
The first admitted homogeneous B4 DPR full/max primary controls give median
batch-step times129.058/99.192ms across three warm blocks. Other B4 methods
remain pending, so these partial numbers do not establish the full batch table.

## Transient reference refinement

Half-timestep trajectory728209 and contact-order6/10 trajectories728226 pass
their complete400/800-step histories and current-position moment checks.
Field controls728261 give maximum half-dt tangent/membrane/bending changes
3.314e-7 relative, and contact-quadrature changes0.1248%; raw-peak changes are
at most1.042e-7 and0.0125%, respectively. Depth2/4 integration differences are
negligible relative to the predeclared0.5% field/1% peak solver/time targets.
Spatial convergence of sustained friction is still unqualified.

The500um/two-wall StVK run728228 failed at step45: subtracting complete
`K u - f(u)` left a1.027e-9 correction change at the unchanged1e-9 target after
20 rounds. The explicit `--constitutive-correction direct` variant instead
assembles the algebraically identical nonlinear remainder locally. It leaves
the constitutive law, contact solve,20-round cap and1e-9 target unchanged;
acceptance still independently reassembles the full Piola force.

Analytic rotation/covariance/energy/direct-correction tests and the400-step
direct trajectory728255 pass. Full-field comparison728297 against the original
coarse modal trajectory gives at most7.423e-10 tangent/membrane/bending change,
and5.462e-10 common-peak change across normal and sustained-friction endpoints.
The new fine run728256 has passed the former step45 failure boundary, but its
unfinished history is not admitted. Further spatial controls, qualified
friction errors and common-scale field/traction maps remain required.
