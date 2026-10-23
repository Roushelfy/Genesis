"""Freeze larger-batch primary timing and separate fresh-process memory evidence."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_source_warps'
ALIAS = 'f6c4ff2b017d44cf45bb37b97aacd8ad9c9de5c9'
BASELINE = '1c872039165fad4dc4b3c7e642ca30126050dae8'
names = [f'hard-batch-scaling{batch}-v1' for batch in [256, 1024]]
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    root = N / name / 'group0'
    assert observed[f'{job}_0'] == ['COMPLETED', '0:0'], (name, observed.get(f'{job}_0'))
    assert (root / 'returncode.txt').read_text().strip() == '0'
    tasks = read(N / name / 'tasks.json')
    assert len(tasks) == 1 and len(tasks[0]) == 7
    for task in tasks[0]:
        assert read(root / task['name'] / 'result.json')['passed']
    states[name] = {'job': f'{job}_0', 'state': 'COMPLETED', 'exit_code': '0:0'}
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
assert {row['recorded_revision'] for row in replayed} == {ALIAS, BASELINE}
heads = read(N / 'meta/source-warps-terminal-v1.json')['frozen_heads']
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1

timing, memory, comparisons, bindings, caches = [], [], [], [], []
for name, batch_size in zip(names, [256, 1024]):
    root = N / name / 'group0'
    before, after = [read(root / kind / 'result.json') for kind in ['baseline', 'candidate']]
    assert len(before['rows']) == len(after['rows']) == 3
    identities = [read(root / kind / 'start.json')['identity'] for kind in ['baseline', 'candidate']]
    assert identities[0] == identities[1]
    for left, right in zip(before['rows'], after['rows']):
        mode = right['mode']
        assert left['mode'] == mode and left['batch'] == right['batch'] == batch_size
        assert left['audit_environment_steps'] == right['audit_environment_steps'] == 130 * batch_size
        assert len(left['seconds_per_scene_step']) == len(right['seconds_per_scene_step']) == 3
        if mode != 'rigid':
            assert left['admitted_timed_environment_steps'] == right['admitted_timed_environment_steps'] == 90 * batch_size
            assert left['surface_quadrature'] == right['surface_quadrature'] == 16
            for prefix in ['loaded', 'sharp']:
                assert left[prefix + '_timed_environment_steps'] == right[prefix + '_timed_environment_steps']
                for row in [left, right]:
                    assert row[prefix + '_timed_environment_steps'] == 3 * row[prefix + '_window_environment_steps']
        timing.append({'baseline': left, 'candidate': right,
            'scene_cost_reduction_fraction': 1 - right['median_seconds'] / left['median_seconds'],
            'candidate_repeat_range_below_baseline': max(right['seconds_per_scene_step']) < min(left['seconds_per_scene_step'])})
        mem = read(root / f'memory-{mode}/result.json')
        assert mem['passed'] and not mem['primary_timing'] and all(mem['checks'].values())
        assert mem['driver_error'] is None and mem['memory_error'] is None
        assert mem['identity'] == identities[0]
        audit = mem['polling_audit']
        assert audit['collector_returncode'] == 0 and audit['target_pid'] == mem['profile_pid']
        assert audit['poll_count'] >= audit['visible_poll_count'] > 0
        # Sampled peaks are observations, never continuous-peak capacity bounds.
        samples = mem['memory']['samples']
        memory.append({'batch': batch_size, 'mode': mode,
            'sampled_gpu_peak_mib': max(sample['process_mib'] for sample in samples),
            'process_max_rss_kib': mem['process_max_rss_kib'], 'polling_audit': audit,
            'result': identity(root / f'memory-{mode}/result.json'),
            'driver_row': read(root / f'memory-{mode}/driver/result.json')['rows'][0]})
    for kind in ['observations', 'memory-observations']:
        result = read(root / kind / 'result.json')
        assert result['passed'] and len(result['rows']) == 3
        for row in result['rows']:
            assert row['frames'] == 130 and row['batch'] == batch_size
            assert [audit['repeats'] for audit in row['repeated_count_audits']] == ([3, 3] if kind == 'observations' else [3, 1])
        comparisons.append({'batch': batch_size, 'kind': kind, 'result': result})
    for stem in ['time_hard_franka', 'compare_hard_observations', 'memory_hard_franka']:
        relative = Path(f'research/rigid_stress/qipc_comparison/{stem}.py')
        saved = N / name / 'analysis-source' / relative
        assert saved.read_bytes() == (WORK / relative).read_bytes()
        bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
    script = N / name / 'workload-0.sh'
    text = script.read_text()
    keys = [hashlib.sha256(str(WORK.parent / source).encode()).hexdigest()[:16]
            for source in ['hard_contact_edge_projection', 'hard_contact_source_warps']]
    for key in keys:
        assert f'GS_CACHE_FILE_PATH="$task_cache/{key}/genesis"' in text
        assert f'QD_OFFLINE_CACHE_FILE_PATH="$task_cache/{key}/quadrants"' in text
    caches.append({'campaign': name, 'source_keys': keys, 'script': identity(script)})

scope = ('Actual B256/B1024 moving Franka timings with the four-condition bank tiled64/256 times, '
    '64-bit precision, surface order16,130-frame admission and three100..129 repeats. '
    'Frozen serial/warp source engines use the same GPU per batch; independent fresh-process '
    'rigid/max/full GPU memory polls run separately and are not primary timing. '
    'Public observations and repeated-step counts must agree across engines and memory runs. '
    'Sampled GPU peaks include non-Torch allocations but are not continuous peaks or capacity bounds. '
    'This is neither independently diverse large-batch qualification nor a best-batch or full-cycle claim. '
    'The original frame428 refusal and independent physical-stress limits remain unchanged.')
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'baseline_engine': BASELINE,
    'frozen_heads': heads, 'preserved_files': preserved, 'paired_timings': timing,
    'fresh_process_memory': memory, 'public_observations': comparisons, 'source_bindings': bindings,
    'configured_source_private_caches': caches,
    'count_convention': 'Loaded/sharp timed counts now include all repeats; new *_window_environment_steps fields retain one-window counts. Old immutable archives used one-window counts under the timed label. Timing values and admission totals are unchanged.',
    'scope': scope, 'whole_goal_complete': False}
p = META / 'batch-scaling-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect(names)
files += [p, HELPER_SOURCE, REPLAY_HELPER]
files += [META / stem for stem in ['batch-scaling-storage-probe-v1.txt', 'batch-scaling-start-v1.json',
    'batch-scaling1024-capacity-v1.json', 'prepare-batch-scaling-v1.py', 'document-batch-scaling-v1.py']]
files += list(META.glob('hard-batch-scaling*-v1-tasks.json'))
files += list(META.glob('hard-batch-scaling*-v1-memory-*-task.json'))

def input_receipts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            original = Path(value['path'])
            actual = identity(original)
            assert all(actual[key] == value[key] for key in ['bytes', 'sha256']), str(original)
            # Current-worker bindings are verified now; immutable captured copies
            # supply their archived bytes, not future mutable workspace files.
            if not original.resolve().is_relative_to(WORK):
                external.add(original.resolve())
        else:
            for item in value.values(): input_receipts(item)
    elif isinstance(value, list):
        for item in value: input_receipts(item)

input_receipts(receipt)
input_receipts(read(META / 'batch-scaling1024-capacity-v1.json'))
for name in names:
    files += list((N / name).glob('extra-source*.json')) + list((N / name).glob('extra-source*.patch'))
    for start in (N / name).glob('group*/**/start.json'): input_receipts(read(start))
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('hard-batch-scaling-v1', files, sorted(external), scope)
(DEST / 'hard-batch-scaling-v1/README.md').write_text('# Larger-batch throughput and process memory\n\n' + scope +
    '\n\nSee [the batch and memory measurements](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#larger-batch-throughput-and-process-memory). '
    'Both scientific campaigns are terminal. All54 source heads and12 original unfinished files remain preserved. '
    'Scientific work used Slurm rtx-mid/image1_26, existing data-side Python and source-private node caches. '
    'The full-mode memory task precedes primary timing and warms candidate compiler caches; preparation values are not a cold-start comparison.\n')
with (META / 'batch-scaling-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
