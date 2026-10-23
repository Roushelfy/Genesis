"""Document measured terminal batch/memory results without changing older evidence."""
from pathlib import Path
import json

n = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
w = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
r = json.loads((n / 'meta/batch-scaling-terminal-v1.json').read_text())
p = json.loads((n / 'meta/batch-scaling-publication-v1.json').read_text())
assert r['passed']
rows = {(row['candidate']['batch'], row['candidate']['mode']): row for row in r['paired_timings']}
mem = {(row['batch'], row['mode']): row for row in r['fresh_process_memory']}
for batch in [256, 1024]:
    for mode in ['max', 'full']:
        for kind in ['baseline', 'candidate']:
            row = rows[batch, mode][kind]
            assert row['loaded_timed_environment_steps'] == 90 * batch
            assert row['sharp_timed_environment_steps'] == 18 * batch
table = ['| Batch | Engine | Pure rigid ms/step | Max ms/step | Full ms/step | Max/full env-step/s |',
         '| --- | --- | ---: | ---: | ---: | ---: |']
for batch in [256, 1024]:
    for kind in ['baseline', 'candidate']:
        a, b, c = [rows[batch, mode][kind] for mode in ['rigid', 'max', 'full']]
        table.append(f"| {batch} | {kind} | {a['median_seconds'] * 1000:.3f} | {b['median_seconds'] * 1000:.3f} | {c['median_seconds'] * 1000:.3f} | {b['environment_steps_per_second']:.2f} / {c['environment_steps_per_second']:.2f} |")
memory_table = ['| Batch | Mode | Sampled process GPU peak (MiB) | Process RSS high-water (GiB) | Longest observed query gap (s) |',
                '| --- | --- | ---: | ---: | ---: |']
for batch in [256, 1024]:
    for mode in ['rigid', 'max', 'full']:
        m = mem[batch, mode]
        memory_table.append(f"| {batch} | {mode} | {m['sampled_gpu_peak_mib']:,} | {m['process_max_rss_kib'] / 1024**2:.3f} | {m['polling_audit']['maximum_observed_query_gap_seconds']:.3f} |")
changes = []
for batch in [256, 1024]:
    a, b = [rows[batch, mode] for mode in ['max', 'full']]
    changes.append(f"At B{batch}, max/full median scene-cost reductions are {100*a['scene_cost_reduction_fraction']:.3f}% / {100*b['scene_cost_reduction_fraction']:.3f}%; candidate inclusive added costs over its paired pure-rigid mode are {1000*a['candidate']['added_seconds_per_scene_step']:.3f} / {1000*b['candidate']['added_seconds_per_scene_step']:.3f} ms.")
separated = [f"B{batch} {mode}" for batch in [256, 1024] for mode in ['max', 'full']
             if rows[batch, mode]['candidate_repeat_range_below_baseline']]
overlap = [f"B{batch} {mode}" for batch in [256, 1024] for mode in ['max', 'full']
           if not rows[batch, mode]['candidate_repeat_range_below_baseline']]
repeat_text = 'Candidate repeat ranges lie entirely below their paired baseline ranges for ' + ', '.join(separated) + '.' if separated else 'No candidate repeat range lies entirely below its paired baseline range.'
if overlap:
    repeat_text += ' This separation does not hold for ' + ', '.join(overlap) + '.'
for mode in ['max', 'full']:
    pair = rows[1024, mode]
    samples = [pair[kind]['seconds_per_scene_step'] for kind in ['baseline', 'candidate']]
    assert max(min(values) for values in samples) <= min(max(values) for values in samples)
repeat_text += ' At B1024 both stress-mode repeat ranges overlap; these measurements do not establish a useful scene-level benefit from the source-search change at that batch.'
scaling_text = 'Relative to B256, B1024 changes candidate max/full aggregate throughput by '
scaling_text += ' / '.join(f"{100 * (rows[1024, mode]['candidate']['environment_steps_per_second'] / rows[256, mode]['candidate']['environment_steps_per_second'] - 1):+.2f}%" for mode in ['max', 'full'])
scaling_text += '. Its fresh-process sampled max/full GPU peaks are '
scaling_text += ' / '.join(f"{mem[1024, mode]['sampled_gpu_peak_mib'] / mem[256, mode]['sampled_gpu_peak_mib']:.2f} times" for mode in ['max', 'full'])
scaling_text += ' those at B256. For this tiled window, B256 is the lower-memory measured choice; use the measured B1024 throughput/memory tradeoff when choosing a larger simulation batch.'
section = '''
## Larger-batch throughput and process memory

The unchanged cooperative-source engine `f6c4ff2b0` now has measured B256 and
B1024 runs against serial-source baseline `1c8720391`. Each batch runs both
sources on one RTX PRO 6000 Blackwell Server GPU, with source-private compiler
caches, 64-bit precision and surface order 16. The moving scene tiles the four
starting conditions 64 or 256 times. It preserves the 0--129 continuous audit
and three timed 100--129 windows, including runtime acceptance, the controller,
dispatch and final synchronization. Reset, pre-window history and audit/export
remain outside timing. No engine or numerical acceptance threshold changes.

''' + '\n'.join(table) + '\n\n' + '\n\n'.join(changes) + '\n\n' + repeat_text + '''
The three repeat blocks describe these allocations, not statistical confidence
intervals or a portable speedup. Separate batches use different allocations;
their throughput trend does not isolate GPU-to-GPU variation. The small source
search improvement at large batches does not attribute their remaining scene
cost to a particular kernel.

Every primary stress mode admits 23,040 timed environment steps at B256 and
92,160 at B1024, with 33,280 / 133,120 separately audited environment steps.
All timed environments are loaded. Sharp-edge counts are 4,608 / 18,432 across
the three windows. The complete 130-frame public observations pass both the
baseline/candidate comparison and each fresh-process memory/primary comparison:
positions use absolute 1e-12 m; stresses use relative 1e-7 / absolute 1e-6 Pa;
validity, load/edge/status masks and infinite maxima retain exact checks.
These compare public observations, not every internal full-field array. The
preceding strict raw-field differences and original frame428 refusal remain
preserved; this batch study changes neither their criteria nor their outcomes.

The timer now accepts a single mode so memory can be measured from a fresh
process. An independent NVIDIA polling process starts before scientific imports
and stops after build, the 130-frame audit and one repeated timing window.
It filters the driver PID, records all polls, and verifies the GPU identity,
collector exit and observation agreement. Each mode's polling process is reaped
before the next task. Memory-run rates are intrusive diagnostics and excluded
from the primary throughput table.

''' + '\n'.join(memory_table) + '\n\n' + scaling_text + '''

These GPU samples include non-Torch allocations. Requested 0.1 s polling plus
query latency can miss short peaks; the observed maxima are not continuous
peaks or memory-capacity bounds. RSS includes the whole process through build
and the audit. Device-wide CUDA free memory, Torch allocator counters and job
MaxRSS are retained separately and must not be substituted for process GPU
memory. The primary timer also records its final memory counters outside the
timed window; earlier modes can retain allocations in that shared process.

Full-mode memory runs first and warms the candidate compiler caches before
primary timing. Preparation/build durations are retained but are not a matched
cold-start comparison. Both modes fit at B1024 on the measured hardware. These
tiled starting conditions do not establish throughput for independently chosen
large populations, full-cycle admission, the best batch or the memory limit.
Use B256/B1024 as measured points for this bounded workload; further tuning
must preserve admission and measure its own memory and trajectory diversity.

The worker also corrects a reporting bug: old `loaded_timed_environment_steps`
and `sharp_timed_environment_steps` described one audited window even when
three windows were timed. New `*_window_environment_steps` retain those
single-window values, and `*_timed_environment_steps` include all repeats.
Both comparison tasks verify the counts against saved audit arrays. Older
archives remain immutable; their timing values and admitted-step totals were
already correct.

The [batch-scaling packet](../evidence/20261019-qualified-ranking/hard-batch-scaling-v1/README.md)
retains both completed seven-task campaigns, raw timing samples, independent
process polling, driver/audit outputs, source inventories, exact captured worker
bindings and the B256-to-B1024 resource decision. Its captured task plans and
submission/workload scripts record the executed arguments and source/cache
selection. Reproduction requires fresh output directories and corresponding
task-path rebinding; archived output directories must remain immutable.
All 54 frozen engine heads and 12 original unfinished files are preserved.
Runtime arrays, logs and metadata remain under `/mnt/data/zhaofeng`; source,
documentation and compact publication evidence stay in workspace.
'''
doc = w / 'research/rigid_stress/docs/SHARP_EDGE_PRESSURE_20261022.md'
text = doc.read_text()
assert '## Larger-batch throughput and process memory' not in text
old_status = ('This supplies the sharp-edge pressure component;\n'
              'actual contact binding, complete curved-shell recovery and accepted full-grasp\n'
              'max/full throughput remain open.')
assert old_status in text
text = text.replace(old_status,
    'Subsequent sections measure native source/edge binding, curved-shell recovery\n'
    'and admitted max/full throughput over frames100--129. Independent complete\n'
    'curved-shell physical stress errors and full-cycle admission remain open;\n'
    'the original frame428 still refuses.', 1)
text = text.replace('Larger useful\nbatches and their memory requirements still need measurement.',
                    'The following batch study extends measured throughput and memory to B256/B1024.')
doc.write_text(text.rstrip() + '\n' + section)
a, b = [rows[1024, mode]['candidate'] for mode in ['max', 'full']]
update = f'''The [larger-batch study](docs/SHARP_EDGE_PRESSURE_20261022.md#larger-batch-throughput-and-process-memory)
measures B256/B1024 with separate process-memory runs and passing public output
comparisons. B1024 max/full reach {a['environment_steps_per_second']:.0f} / {b['environment_steps_per_second']:.0f} env-step/s;
fresh-process sampled GPU peaks are {mem[1024, 'max']['sampled_gpu_peak_mib']/1024:.2f} / {mem[1024, 'full']['sampled_gpu_peak_mib']/1024:.2f} GiB.
The batches tile four starting conditions and cover the admitted 100--129 window;
they do not establish the best batch, diverse-population or full-cycle throughput.
Loaded/sharp timed counters now include all repeats; older timing and admission
values remain valid and archived unchanged. Engine code and physical limits are unchanged.

'''
readme = w / 'research/rigid_stress/README.md'
text = readme.read_text()
anchor = 'The preceding [contact-application study]'
assert anchor in text
readme.write_text(text.replace(anchor, update + anchor, 1))
data = w / 'DATA_LOCATION.md'
text = data.read_text()
header = '# Rigid-stress data and artifact locations\n\n'
assert text.startswith(header)
data.write_text(header + f'''The `hard-batch-scaling-v1` packet retains {p['payloads']} compact payloads
({p['bytes']:,} bytes) and {p['external_arrays']} external receipts from two completed campaigns.
Primary B256/B1024 timings, fresh-process GPU polls, audits and logs remain in
the restored data root below under `hard-batch-scaling256-v1` and
`hard-batch-scaling1024-v1`. Scientific execution used Slurm `rtx-mid`, image
`1_26`, the existing data-side environment and source-private node caches.
No engine checkout moved or changed; all 54 frozen source heads and 12 original
unfinished files remain preserved. Source/docs and compact evidence stay in
workspace. No new home fallback output, data migration or environment change occurred.

''' + text[len(header):])
print(json.dumps({'documented': True, 'packet': p, 'range_separation': separated, 'range_not_separated': overlap}))
