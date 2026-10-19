# Native observation cadence and parent-admission controls

B8/B128 complete conformance729467 and paired primary timing729470 finish all
26 cases each. Independent audit729473 passes521 checks and rehashes858 source
files. Sparse sampling preserves the measured motion and requested stress
observations while reducing scene cost. Physical collider errors and diverse
SCR reliability remain separate from this repeated eight-condition bank.

## Observation contract

The immutable measured engine is1ff39b077. Geometry, material, source model,
precision, original runtime acceptance and moving eight-condition bank match
the [canonical native configuration](NATIVE_CONTACT_CONSISTENCY_20261019.md).
All scenes use dt0.01s, one physical substep and elastic history0.

| Stride in scene steps | Observation interval | Native update mode |
| --- | --- | --- |
| 1 | 0.01s | every_step, unchanged baseline |
| 10 | 0.1s | manual, request after the step's inputs |
| 50 | 0.5s | manual, request after the step's inputs |

The helper registers a public pre-substep callback, after FrankaEgg applies its
controls and any partial reset. The callback requests recovery before the first
physical substep at the declared ticks. Request dispatch contributes to primary
wall time. No radius, force, friction cone, environment or failing step is
changed to obtain a result.

Sparse getters retain the latest sampled values until another observation or
state invalidation. `step_valid` alone does not mean a fresh observation; reset
states also set it true. The control records an explicit freshness mask. A
sampled maximum covers that sampled scene step, not unobserved peaks throughout
the preceding interval. Full-output buffers remain on GPU. No policy transition
or required host readback is measured by these primary scene windows.

Pilot DPR/full/stride10 issues210 requests across900 warmup and1200 recorded
steps; its1080 unsampled full-field states pass exact holding/reset checks.
SCR/full/stride50 issues42 requests and passes1176 such checks. Both independently
recompute von Mises stress from all saved tensor components and compare the
global reduction with the saved maxima. The completed campaign additionally compares all
requested peaks and twelve common complete fields with matched every-step runs.

The predeclared complete-campaign checks retain the existing1e-10 position/
configuration and1e-9 velocity motion budgets, exact contact counts, max/full
1e-8 relative plus1e-3Pa budget, and standalone-replay cadence consistency
1e-4 relative plus1e-3Pa. Every sampled peak and twelve common complete-field
snapshots are compared. All1200 motion/error/peak records, authored partial
resets, request counts and all unsampled full-output states are checked. Batches
repeat the frozen eight conditions; this is not a diverse-SCR reliability test.

Production timing excludes recording and independent audits. Each batch's pure
rigid, DPR/SCR, max/full and three cadences run on one GPU with900 warmup and
three fixed1200-step intervals. Warm repeats are within process. Required native
acceptance and periodic error handling remain active. The original every-step
friction failures and independently certified diverse-SCR failures remain
separate. Collider physical errors cannot inherit the2mm full-centroid pad errors.

## Fail-closed fine-reference postprocessing

`audit_solid_pad` records a failed physical admission without necessarily
returning a nonzero process status. Scheduler `afterok` alone is insufficient.
The new `require_solid_pad_admission` preflight checks complete sequential
histories, every original mechanical gate, both equilibrium windows, independent
current momentum, matching result receipts and both endpoint field hashes.

Control729476_0 passes all ten checks for the admitted400-step coarse reference.
Control729476_1 intentionally exits1 for the old125um history: the scheduler
success receipt exists, but history, momentum and every-step physical admission
fail. The positive and rejected originals remain unchanged.

Pending analysis-only jobs729474/729475 were cancelled before any worker ran.
Time/contact fields depend on729478, the explicit six-parent preflight waiting
for729312/729314. Spatial fields use only four of those parents; a separately
pinned preflight729487 waits for729312 and verifies both fine q10 histories plus
both250um baselines. Pending spatial job729479 now depends on729487, with the
exact scheduler update recorded in meta/spatial-parent-dependency-split-v1.json.
No field worker or captured source changes. Final qualification still requires
the separate time/contact controls.
Slurm dependency metadata, worker code and immutable task inputs are retained
under the data-side ranking run root. Queued jobs are not completed evidence;
fine spatial/time/contact qualification and final friction delivery remain open.

The [pilot packet](../evidence/20261019-qualified-ranking/native-observation-cadence-pilot-v1/README.md) retains66 payloads/390,160 bytes. The [parent-gate packet](../evidence/20261019-qualified-ranking/fine-parent-admission-guard-v1/README.md) retains84 payloads/502,986 bytes, including failed/cancelled receipts and guarded immutable follow-up plans. Live campaign outputs are excluded. Spatial729479 depends on four-parent preflight729487; time/contact729480 retains six-parent preflight729478. Archived earlier plans retain their original dependency and publication-time scope.


## Completed scene results

Each batch's thirteen primary modes run on one NVIDIA RTX PRO 6000 Blackwell
Server Edition GPU; the batches use different GPU UUIDs of that same model.
Motion is bitwise identical to pure rigid for all1200 recorded steps in every
stress mode. The largest sampled peak/full-tensor differences against every-step
are0.000378Pa/0.000246Pa, within the predeclared budgets. Max/full output, complete
contact counts, condition-matched batches and all held/reset states pass.

The table shows median aggregate env-scene-step/s at B128. Parentheses show
added scene latency relative to the paired pure-rigid median44385.6; they are
not overhead per stress update. Intervals use three warm repeats within process.
The complete CSV retains all26 rows and minimum/maximum repeat rates.

| GPU output | Every step | Every10 steps | Every50 steps |
| --- | ---: | ---: | ---: |
| DPR max | 32108 (+38.2%) | 42255 (+5.0%) | 43562 (+1.9%) |
| DPR full | 31928 (+39.0%) | 42056 (+5.5%) | 43566 (+1.9%) |
| SCR max | 10833 (+309.7%) | 33013 (+34.4%) | 41164 (+7.8%) |
| SCR full | 10873 (+308.2%) | 32867 (+35.0%) | 40968 (+8.3%) |

At B8, pure rigid is2932.2 env-scene-step/s. DPR max rates are2207.5/2811.1/2895.2
and SCR max721.7/2185.7/2727.2 for strides1/10/50. These rates count physical scene
steps; requested stress observations occur at100/10/2Hz per environment.
Max/full differences within these runs do not establish a general full-output
speed advantage. Memory profiling and module replay retain their separately
measured scopes; this experiment adds no policy or host-observation accounting.

The [terminal packet](../evidence/20261019-qualified-ranking/native-observation-cadence-v1/README.md)
retains312 payloads/879,107 bytes of exact primary/control/audit receipts, CSV, raw-log bytes, worker variants,
source identities and the prospective spatial-dependency update. Large complete
trajectories remain on data with hashes. This closes the measured sparse-output
configuration comparison for this bank. It does not replace every-step friction
failures, certify unobserved peaks or qualify physical errors for collider inputs.
