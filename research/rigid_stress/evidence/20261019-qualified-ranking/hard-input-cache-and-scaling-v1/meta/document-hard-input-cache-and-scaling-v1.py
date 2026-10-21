"""Document measured cache ablations and the selected nearest-surface bounds."""

import json
from pathlib import Path

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
read = lambda path: json.loads(path.read_text())
summary = read(N / 'meta/hard-input-scaling-summary-v1.json')
trace = read(N / 'meta/hard-input-bounds-trace-summary-v1.json')['rows']
paired = [read(N / f'hard-input-bounds-paired-v1/group0/{name}_comparison/result.json') for name in ['baseline', 'bounds']]
labels = {'published': 'Published precomputation', 'cache_v1': 'Bounds + temporal hints',
          'cache_v2': 'Squared bounds + temporal hints', 'closest_bounds': 'Nearest-surface bounds only'}
table = ['| Variant | B | Input scene ms/step | Pure rigid ms/step | Added ms/step | Raw aggregate env-step/s |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for row in summary['rows']:
    table.append(f"| {labels[row['variant']]} | {row['envs']} | {row['input_seconds_per_step']*1000:.3f} | {row['rigid_seconds_per_step']*1000:.3f} | {row['added_seconds_per_step']*1000:.3f} | {row['raw_aggregate_env_steps_per_second']:.0f} |")
counts = ['| B | Temporal hint / active source | Selected root matches temporal hint | Mean closest / root candidate calculations |',
          '| ---: | ---: | ---: | ---: |']
for row in summary['cache_rows']:
    if row['variant'] == 'cache_v2':
        counts.append(f"| {row['envs']} | {row['temporal_hint_fraction']:.2%} | {row['temporal_hit_rate']:.2%} | {row['closest_full_candidate_evaluations']['mean']:.2f} / {row['normal_root_full_candidate_evaluations']['mean']:.2f} |")
gpu = ['| Native phase | CUDA tasks/step | B4 baseline / selected us | B1024 baseline / selected us |',
       '| --- | ---: | ---: | ---: |']
for phase in ['begin_step', 'association', 'provenance', 'geometry_roots', 'geometry_planar', 'grouping_wrench', 'hertz_inputs', 'hertz_law']:
    entries = [next(row for row in item['scopes'] if row['scope'] == 'HertzInputs.' + phase) for item in trace]
    gpu.append(f"| {phase} | {entries[0]['kernel_tasks_per_step']:.0f} | {entries[0]['sum_kernel_duration_us_per_step']:.2f} / {entries[1]['sum_kernel_duration_us_per_step']:.2f} | {entries[2]['sum_kernel_duration_us_per_step']:.2f} / {entries[3]['sum_kernel_duration_us_per_step']:.2f} |")
valid = ', '.join(f"B{row['envs']}: {row['valid_loaded_environment_steps']}/{row['loaded_environment_steps']}" for row in summary['rows'] if row['variant']=='closest_bounds')
paired_text = '; '.join(f"{name}: {row['timing']['inputs']['median_scene_step_seconds']*1000:.3f} ms input, {row['timing']['rigid']['median_scene_step_seconds']*1000:.3f} ms rigid, {row['incremental_scene_step_seconds']*1000:.3f} ms added" for name,row in zip(['baseline','selected'],paired))
section = '''

## Cache ablations and batch scaling

The follow-up evaluates temporal normal-root hints, source-vertex hints and
conservative face bounds, then selects **nearest-surface bounds only**. The
complete cache variants pass numerical controls but add state and search work
without a consistent inclusive advantage over the smaller change. Their exact
sources, failures and measurements remain in the
[cache/scaling packet](../evidence/20261019-qualified-ranking/hard-input-cache-and-scaling-v1/README.md).
They are not the active engine path.

The selected implementation precomputes local face boxes and their squared
scales once. The nearest-surface loop retains ascending face order and its
original distance tolerance; conservative squared lower bounds avoid expensive
triangle candidates that cannot improve that ordered selection. All components
remain eligible. The exhaustive normal-root implementation, general curvature
arithmetic, grouping, force/moment accounting and live adapter API are unchanged.
This adds 56 logical bytes per face: 152,656 bytes for this scene, shared across
all environments. It adds no temporal state or reset obligation.

All rows use the previous 720-step, three-warmed-repeat, float64, 0.01 s/one-substep
moving Franka setup on RTX PRO 6000 Blackwell Server Edition GPUs, container 1_26.
Initial conditions match within each batch/variant comparison. Environments
have independently varied initial poses, friction and controller phase; these
are not homogeneous replay rates. Required validity runs on device. Research
histories, probes, grouping oracle and synchronized phase profiles are excluded
from the primary repeats. The research callback remains included.

''' + '\n'.join(table) + '''

The added cost is a difference of independently timed matched medians. The B256
cross-job subtraction is mixed, even though its absolute input time decreases;
it must not be presented as a uniform overhead reduction. A focused repeat runs
both variants and their rigid partners sequentially in the same allocation:
''' + paired_text + '''. Its complete histories and sampled arrays also pass
the unchanged implementation comparison. No statistical or universal speedup
bound is claimed from three repeats.

All variants retain the same loaded/valid input counts: ''' + valid + '''.
Raw aggregate rates include rejected environments. Geometric validity alone
does not certify a physical pressure or a useful stress update. B1024 is the
largest tested batch here, not a practical memory limit or proven throughput
optimum. Complete max/full elastic output remains absent.

Each final variant passes 2406 nearest-surface/curvature queries per CPU/GPU
backend with unchanged face choices, plus the existing 12 region/source controls.
Every audited moving step compares grouping, counts, forces and moments exactly
against the original serial oracle. Baseline/variant conditions and trajectories,
720-step source identities/validity and retained input arrays pass the original
numerical limits. B4's environment prefix also agrees with B64/B256/B1024,
including partial resets; full-batch reset totals naturally differ. Only the
diagnostic candidate-evaluation counter is excluded from tensor conformance.

The cache candidates additionally pass 11 lifecycle/identity cases per backend:
cold/warm contact, wrong disconnected-component hint, release/reload, partial
reset, source permutation, out-of-range hint, malformed pair/recovery and a
discontinuous source position. Roots match the frozen exhaustive implementation
exactly in face index, position and residual for those controls. A hint is
accepted only for the current geometry/pair, and all potentially closer faces
remain searchable. Failed finite-face inputs still reject and clear the cached
face. These checks concern geometry, not tangential-memory invalidation.

''' + '\n'.join(counts) + '''

The hit percentage is conditional on providing a temporal hint. Even a hit still
executes conservative bounds over other faces; it is not a search-free update.
Candidate counts exclude box-bound visits. Every actual moving-scene vertex-hint
count is zero: the synthetic provenance control works, but this collision path
does not supply a usable vertex feature. Broad search and finite-face rejection
therefore remain significant. The full cache adds 29 logical bytes per contact
slot, including diagnostics: 87,000 / 1,392,000 / 5,568,000 / 22,272,000 bytes for
B4/B64/B256/B1024, plus 158,308 shared geometry bytes in the squared variant.
This extra state is not retained in the selected implementation.

Separate 60-step CUDA traces follow 480 warmup steps without per-phase
synchronization or research audits. The table reports **summed device task
durations**, not inclusive wall time; host scope timing remains separate.

''' + '\n'.join(gpu) + '''

The input callback launches 21 CUDA tasks per scene step in these windows and
performs no traced device-to-host copy. One Quadrants call can launch several
tasks. Planar-query device cost falls from about 502 to 411 us at B4 and 573 to
452 us at B1024. Summed task times may overlap and must not be added to host
dispatch or independently synchronized phase medians. Some rigid GPU events
lack a recorded matching driver event; the summary keeps them unattributed.
Profiler occupancy metadata is an estimate, not a measured utilization result.

The trace records setup/first-compile time, logical field bytes, torch allocator
values, device free-memory snapshots and host maximum RSS. Device snapshots are
not process peaks, torch counters exclude Quadrants allocations, and logical
bytes exclude allocator rounding and other state. The initial trace writer
exported its traces, then failed while treating a Quadrants internal dictionary
as a tensor; the corrected writer uses declared fields and the final utility
uses `iter_data`. Both failed runs are preserved.

Reproduce each variant using its captured tasks, worker import cone and source
patch. The cache workers intentionally differ from the active baseline-compatible
adapter; use their captured snapshots. Large compressed histories, raw traces,
expanded snapshot summaries and arrays remain hashed on data. The selected
engine change and active trace/batch-comparison utilities match executed sources.
Further work remains on useful collision provenance, genuinely local updates,
fusion, homogeneous and larger batches, finite-face/source-moment compatibility,
friction memory, finite-shell accuracy and actual complete max/full scene cost.
'''
path = W / 'research/rigid_stress/docs/HARD_CONTACT_ESTIMATES_20261020.md'
assert '## Cache ablations and batch scaling' not in path.read_text()
path.write_text(path.read_text() + section)
path = W / 'research/rigid_stress/README.md'
text = path.read_text()
anchor = 'The [first hard-contact controls]'
assert anchor in text
text = text.replace(anchor, '''The [cache ablations and batch scaling](docs/HARD_CONTACT_ESTIMATES_20261020.md#cache-ablations-and-batch-scaling)
now cover B4/B64/B256/B1024, CUDA task attribution, logical memory and exact
shared-environment isolation. The selected nearest-surface bounds reduce the
measured planar device query by about18-21%; inclusive scene gains are smaller
and paired subtraction varies. Full temporal caches remain captured experiments.
All original geometric rejections remain visible; complete stress/history and
physical shell qualification are still open.

''' + anchor, 1)
path.write_text(text)
path = W / 'research/rigid_stress/docs/QUALIFIED_STRESS_COMPLETION_CHECKLIST.md'
text = path.read_text()
old = '''implementation comparisons pass. It remains input-only: temporal caching,
collision-feature reuse, larger-batch/memory tuning and useful complete max/full
scene integration still require execution.'''
assert old in text
text = text.replace(old, '''implementation comparisons pass. The [cache/scaling follow-up](HARD_CONTACT_ESTIMATES_20261020.md#cache-ablations-and-batch-scaling)
adds CPU/GPU cache lifecycle controls, four batch sizes through1024, shared-prefix
isolation, device traces and logical memory. Only conservative nearest-surface
bounds are selected; full temporal-cache variants remain experiments. Genuine
local updates, useful collision-feature reuse, homogeneous/capacity tuning and
complete max/full scene integration remain open.''')
path.write_text(text)
path = W / 'DATA_LOCATION.md'
path.write_text(path.read_text() + '''

Contact-input cache/scaling ablations use `hard-input-cache-v1/v2`,
`hard-input-trace-v1/v2`, `hard-input-scaling-v1`,
`hard-input-scaling-baseline-v1`, `hard-input-scaling-check-v1`,
`hard-input-bounds-v1`, `hard-input-bounds-trace-v1` and
`hard-input-bounds-paired-v1` under the established20261019 ranking run.
Complete trajectories, large compressed histories, expanded snapshot summaries,
raw CUDA traces and generated meshes stay there with external hashes. The
`hard-input-cache-and-scaling-v1` packet keeps compact terminal/source evidence,
including both initial trace-writer failures. Candidate source worktrees remain
under the existing workspace: `hard_contact_optimized` holds the cache experiment
and `hard_contact_bounds` holds the selected small change. Private compilation
caches are node-local; the data-side physical environment is reused. No existing
runtime data, active environment, compatibility link or preserved unfinished
friction-history source is migrated.
''')
