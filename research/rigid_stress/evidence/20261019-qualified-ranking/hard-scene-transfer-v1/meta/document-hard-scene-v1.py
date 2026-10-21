"""Record the executed scene costs and remaining physical limitations."""
from pathlib import Path
import json

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
summary = json.loads((N / 'meta/hard-scene-terminal-v1.json').read_text())
packet = json.loads((N / 'meta/hard-scene-publication-v1.json').read_text())
bridge_error = max(error for backend in summary['controls'] for control in backend['controls'] for error in control['independent_field_relative_errors'])
preparation = summary['preparation']
profile = {row['phase']: row['median_seconds'] for row in json.loads((N / 'hard-scene-stable-v3/group4/max/result.json').read_text())['phase_profile']}
tables = []
for batch, result in zip([4, 64], summary['comparisons'], strict=True):
    t, c = result['timing'], result['counts']
    tables.append(f"|{batch}|{1000*t['rigid']['median_seconds_per_step']:.3f}|{1000*t['max']['median_seconds_per_step']:.3f}|{1000*t['full']['median_seconds_per_step']:.3f}|{1000*t['max']['added_seconds_per_step']:.3f}|{1000*t['full']['added_seconds_per_step']:.3f}|{c['admitted_loaded_environment_steps']}/{c['loaded_environment_steps']}|")
report = W / 'research/rigid_stress/docs/HARD_CONTACT_ESTIMATES_20261020.md'
text = '''

## Moving-scene transfer and coarse max/full output

The [scene adapter](../qipc_comparison/live_hertz_scene.py) now connects the
selected native input path to witnessed 96-point Hertz transfer, the existing
mass-relieved P2 elasticity and max/full output in the actual moving Franka scene.
It carries each original region force and moment into material coordinates,
walks from the witnessed reference face to the region center, and retains the
existing force/moment budgets. A separate gate checks compression and the
minimum source friction coefficient on every sampled actual material facet.
One failed source/region rejects the complete environment load; its peak and
full tensor are NaN. The research rollout continues rigid motion to retain
failures; this is a diagnostic adapter, not a new public production failure policy.

Three environments on each CPU/GPU backend pass normal and two-direction
proportional-shear controls in different frames, original resultants, max/full,
release, source/moment/cone failure isolation, partial reset and recovery.
Independent scikit-fem same-load complete P2 fields agree within BRIDGE_ERROR
relative norm. This verifies the bridge and elasticity; it does not independently
validate Hertz pressure, a continuum maximum, or friction history.

| Batch | Rigid ms/step | Max ms/step | Full ms/step | Max added | Full added | Admitted loaded / loaded env-steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
''' + '\n'.join(tables) + '''

Each batch uses one allocation for its paired pure-rigid/max/full modes,
720 steps and three warmed repeats, float64, dt 0.01/substeps 1, level 1 elastic
mesh and unchanged independently evolving conditions. Hardware is NVIDIA RTX
PRO 6000 Blackwell Server Edition with container `1_26`. Timings include native
controller/dynamics, geometry, boundary resolution, transfer, elastic solve,
acceptance and resident output. Audits, host field snapshots and phase
synchronization are confined to a separate audited rollout. First construction
and compilation are reported separately. The small B4 full/max difference is
not evidence that requesting full fields accelerates the computation.

Every audited step preserves input-stage validity/source identity/region counts
and the matched rigid trajectory. The selected fixed-preparation variant passes
both complete max/full comparisons. The original B64 run retains one failed peak assertion: step 554/env 19,
286.4669453376328 versus 286.46694534162646 Pa, a 3.994e-9 Pa difference against the
unchanged 1e-9 Pa + 1e-11-relative bound. That original configuration remains failed.
The diagnostic comparison records all remaining checks, then still exits 1
for this failure. Saved complete fields
satisfy an independent von-Mises reduction and invalid fields remain NaN.
These are source-matched numerical checks. Output copies to
a downstream policy and rendering are outside the timed workload.

The B4 run has 1427 geometry-rejected, 499 transfer-rejected and 240 facet-cone-
rejected environment steps; 16 overlap the last two categories. Witness mapping
and elastic solves add no rejections. Only 658/2808 loaded steps pass the complete
discrete gate, versus 1381/2808 for inputs alone. B4 steps 240-599 have zero admitted
loaded updates; fixed step windows do not identify individual finger contacts.
Rejected steps remain part of wall time and cannot count as useful stress updates.

The selected separately synchronized B4 max profile has median boundary-source resolution
BOUNDARY_MS ms, elastic recovery plus peak ELASTIC_MS ms, planar geometry GEOMETRY_MS ms, patch preparation
PATCH_MS ms, transfer TRANSFER_MS ms and load acceptance ACCEPT_MS ms. These intrusive phase timings
include dispatch/synchronization and cannot be summed into the uninstrumented
total. The receipts retain all phases and corresponding B64 measurements.

Fixed-input diagnosis isolates a preparation sensitivity: with identical
operators, 12 repeated max/full recoveries and packed/ordered boundary application
return identical peaks. Independently built original force-response and full
inverse operators differ by 1.32e-11 and 1.62e-11 in relative norm. Recaptured
source forces/region wrenches/velocities are identical; nodal-force differences
are 1.30e-16 relative. The original regression fails its exact operator comparison.

The native repair groups element contributions by matrix entry and sums them
in a fixed element order, using shared connectivity preprocessing. It also
orders one-time mass/center, rigid-mode Gram, centrifugal-wrench and boundary
unit-load reductions. Runtime solve/transfer models and convergence thresholds
are unchanged. The revised force-response, centrifugal-response and full inverse
operators agree exactly across independent builds. CPU/GPU each pass
nine focused checks, including independent U-shaped/cavity uniform-strain and
bending controls with direct and PCG elasticity. Both actual 720-step B4/B64
scene comparisons pass the original peak threshold after this repair.

PREPARATION_SUMMARY

The pressure model still uses proportional shear without unloading memory,
and the coarse P2 field lacks the Hertz near-field/boundary enrichment shown
necessary by the earlier controls. Thus discrete admission and the raw scene
rate do not establish physically useful shell-stress throughput. The next model
work must address finite-face/source moments, actual-normal friction and finite
shell/near-field errors; enlarging the old tolerances would not resolve these.
Large-batch input benchmarks remain valid within their earlier input-only scope.

Executed campaigns: `hard-scene-pilot-v1`/734362 (B4 paired scenes),
`hard-scene-check-v1`/734381 (CPU/GPU controls and complete-history comparison),
and `hard-scene-scaling-v1`/734385 (B64 scenes complete, comparison exits 1).
`hard-scene-replay-capture-v1`/734395 retains full 720-step diagnostic reruns with
the exact step 554 load/velocity/RHS; `hard-scene-replay-v1`/734405 retains the
original failed assertion and a fixed-input recovery/preparation study.
`hard-scene-stable-v3`/734421 supplies the old-engine negative regression,
repaired CPU/GPU checks, fixed-input replay and both complete paired scene
reruns. `hard-scene-preparation-v1`/734432 measures old/new construction and
logical shared storage. Two earlier sparse-checkout capture failures submitted
no jobs and remain recorded. Old scientific failures remain immutable.
The [scene packet](../evidence/20261019-qualified-ranking/hard-scene-transfer-v1/README.md)
retains task commands, captured workers, engine identity, full histories,
terminal accounting and hashes for external complete fields/trajectories.
Replay its `tasks.json` with the existing Slurm submitter and a fresh data-side
output directory; retain the captured worker versions and original conditions.
'''
text = text.replace('BRIDGE_ERROR', f'{bridge_error:.3g}')
for token, phase in [('BOUNDARY_MS', 'boundary_sources'), ('ELASTIC_MS', 'elastic_recovery_and_peak'),
                     ('GEOMETRY_MS', 'geometry_planar'), ('PATCH_MS', 'boundary_patches'),
                     ('TRANSFER_MS', 'traction_transfer'), ('ACCEPT_MS', 'load_acceptance')]:
    text = text.replace(token, f'{profile[phase]*1000:.3f}')
text = text.replace('PREPARATION_SUMMARY',
    f"On the same allocation, warmed shared-model construction changes from {preparation['baseline']['warm_median_seconds']:.3f} s "
    f"to {preparation['stable']['warm_median_seconds']:.3f} s (median of three constructions after one cold build). "
    f"The operator stage changes from {preparation['baseline']['warm_phase_median_seconds']['operators_seconds']*1000:.3f} ms "
    f"to {preparation['stable']['warm_phase_median_seconds']['operators_seconds']*1000:.3f} ms. "
    f"Declared shared tensor storage changes from {preparation['baseline']['shared_logical_bytes']:,} to {preparation['stable']['shared_logical_bytes']:,} bytes; "
    "the additional per-entry offset array is shared across environments. Cold/phase measurements and per-field logical sizes are retained. "
    "Temporary construction state, allocator overhead and process peak memory are outside these byte counts.")
assert '## Moving-scene transfer and coarse max/full output' not in report.read_text()
report.write_text(report.read_text() + text)
readme = W / 'research/rigid_stress/README.md'
needle = 'The [first hard-contact controls]'
addition = '''The [moving-scene max/full pilot](docs/HARD_CONTACT_ESTIMATES_20261020.md#moving-scene-transfer-and-coarse-maxfull-output)
now connects native Hertz transfer and coarse elasticity for paired B4/B64
Franka runs. Independent CPU/GPU bridge controls and both max/full comparisons
pass after fixed-order operator preparation repairs an observed B64 cross-build
peak failure; the original failure and negative regression remain retained.
Actual boundary/source-moment and friction rejection remains
substantial; B4 admits 658/2808 loaded steps. Proportional shear, unresolved
near-field/shell accuracy and failed intervals prevent public estimator promotion.

'''
assert needle in readme.read_text()
readme.write_text(readme.read_text().replace(needle, addition + needle, 1))
checklist = W / 'research/rigid_stress/docs/QUALIFIED_STRESS_COMPLETION_CHECKLIST.md'
value = checklist.read_text().replace('complete max/full scene integration remain open.', 'qualified public max/full scene integration remain open.')
needle = 'The [current-curvature/live-input controls]'
addition = '''The [moving-scene max/full pilot](HARD_CONTACT_ESTIMATES_20261020.md#moving-scene-transfer-and-coarse-maxfull-output)
executes paired B4/B64 geometry/transfer/coarse-elasticity scenes, independent
CPU/GPU bridge controls and complete-history output checks. A B64 peak failure
is repaired by fixed-order shared-operator preparation, with unchanged thresholds,
an old-engine negative regression and eighteen passing CPU/GPU checks. B4 admits only
658/2808 loaded steps, with no admitted loaded update during steps 240-599.
This closes the missing composed timing measurement, while finite-face/source
moments, friction memory and shell/near-field physical accuracy remain open.

'''
assert needle in value
checklist.write_text(value.replace(needle, addition + needle, 1))
location = W / 'DATA_LOCATION.md'
value = location.read_text()
addition = f'''The `hard-scene-transfer-v1` packet retains {packet['payloads']} compact payloads/{packet['bytes']:,} bytes
for B4/B64 paired moving-scene max/full transfer and coarse elasticity, CPU/GPU
independent bridge controls and complete-history comparisons. Its {packet['external_arrays']} external
artifacts stay hashed under the established `20261019-qualified-ranking` data root
in `hard-scene-pilot-v1`, `hard-scene-check-v1`, `hard-scene-scaling-v1` and
`hard-scene-replay-capture-v1`/`hard-scene-replay-v1`, `hard-scene-stable-v3` and
`hard-scene-preparation-v1`. Original B64/regression failures and two unsubmitted
sparse-checkout capture failures remain retained beside the passing repair.
Full fields, trajectories, generated meshes and job logs use data; private
compiler caches were node-local. No active environment, existing data, source
checkout or compatibility link was migrated. Source/docs remain in workspace.

'''
location.write_text(value.replace('\n\n', '\n\n' + addition, 1))
print('Updated the report, README, completion checklist and storage record.')
