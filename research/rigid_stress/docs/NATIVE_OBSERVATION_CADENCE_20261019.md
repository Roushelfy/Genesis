# Native observation cadence and parent-admission controls

The sparse-output pilot729460 passes two complete1200-step output controls and
one separate three-repeat primary-timing control. B8/B128 paired campaigns
729467/729470 and their independent audit729473 remain pending. No sparse
speedup or physical-error claim is inferred from the unpaired pilot.

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
global reduction with the saved maxima. These pilots do not yet compare all
requested samples with a matched every-step run.

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
Replacement fields depend on729478, the explicit six-parent admission preflight,
which itself waits for729312/729314 to complete. Both250um parent baselines and
all four new fine trajectories must pass before any dependent comparison runs.
Slurm dependency metadata, worker code and immutable task inputs are retained
under the data-side ranking run root. Queued jobs are not completed evidence;
fine spatial/time/contact qualification and final friction delivery remain open.

The [pilot packet](../evidence/20261019-qualified-ranking/native-observation-cadence-pilot-v1/README.md) retains66 payloads/390,160 bytes. The [parent-gate packet](../evidence/20261019-qualified-ranking/fine-parent-admission-guard-v1/README.md) retains84 payloads/502,986 bytes, including failed/cancelled receipts and guarded immutable follow-up plans. Live campaign outputs are excluded. Replacement spatial729479 and time/contact729480 depend on preflight729478.
