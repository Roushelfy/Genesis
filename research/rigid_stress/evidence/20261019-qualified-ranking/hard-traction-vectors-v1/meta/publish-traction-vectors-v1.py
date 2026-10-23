"""Freeze boundary vector contraction controls and matched native scene costs."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_traction_vectors'
ALIAS = 'e2979aed2a13125a5a3b092c9cb4dddb43de9291'
BASELINE = 'f6c4ff2b017d44cf45bb37b97aacd8ad9c9de5c9'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-large-profile-v1', 'hard-traction-controls-v2', 'hard-traction-profile-v1', 'hard-traction-timing-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0'], (name, group, observed.get(f'{job}_{group}'))
        assert (root / 'returncode.txt').read_text().strip() == '0'
        for task in tasks:
            assert read(root / task['name'] / 'result.json')['passed']
        states[name].append({'job': f'{job}_{group}', 'state': 'COMPLETED', 'exit_code': '0:0'})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
assert {row['recorded_revision'] for row in replayed} == {BASELINE, ALIAS}
heads = read(N / 'meta/batch-scaling-terminal-v1.json')['frozen_heads']
heads[C3.name] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1

controls = []
for group, backend in enumerate(['cpu', 'gpu']):
    root = N / names[1] / f'group{group}'
    row = {'backend': backend}
    for kind in ['baseline', 'candidate']:
        normal = read(root / f'{kind}-normal/result.json')
        assert normal['passed'] and len(normal['rows']) == 2
        for geometry in normal['rows']:
            assert geometry['rejection_isolation_passed'] and geometry['release_and_recovery_passed']
            assert max(geometry['corner_relative'], geometry['probe_relative']) < normal['targets']['reference_field']
        hard = read(root / f'{kind}-hard/result.json')
        assert hard['passed'] and hard['max_output']['agrees_with_full'] and hard['moving_probes']
        assert all(value == 0 for value in hard['status'] + hard['errno'])
        row[kind] = {'normal': normal, 'hard': hard}
    comparison = read(root / 'comparison/result.json')
    assert comparison['passed'] and len(comparison['rows']) == 5
    row['complete_output_comparison'] = comparison
    controls.append(row)

profiles = []
roots = [(1024, 'initial-baseline', N / names[0] / 'group0/profile')]
roots += [(batch_size, kind, N / names[2] / f'group{group}' / kind)
          for group, batch_size in enumerate([4, 1024]) for kind in ['baseline', 'candidate']]
for batch_size, kind, root in roots:
    result = read(root / 'result.json')
    assert result['passed'] and not result['primary_timing'] and result['batch'] == batch_size
    source = 'hard_contact_traction_vectors' if kind == 'candidate' else 'hard_contact_source_warps'
    imported = str(WORK.parent / source / 'genesis/engine/solvers/rigid/stress/hertz_recovery.py')
    assert any(row['file'] == imported and row['calls'] > 0 for row in result['host_calls'])
    means = {}
    for line in (root / 'kernels.txt').read_text().splitlines():
        match = re.search(r'\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+ms\]\s+(\S+)', line)
        if match:
            for key in ['kernel_hertz_residual_traction', 'kernel_resolve_sources', 'kernel_hertz_combined_field', 'kernel_hertz_planar']:
                if match[4].startswith(key + '_') and match[4].endswith('_range_for'):
                    means.setdefault(key, []).append(float(match[2]))
    profiles.append({'batch': batch_size, 'kind': kind, 'result': result, 'kernel_means_ms': means})
    if kind == 'candidate':
        assert read(root / 'start.json')['identity'] == read(root.parent / 'baseline/start.json')['identity']

timing, observations, caches = [], [], []
for group, batch_size in enumerate([4, 1024]):
    root = N / names[3] / f'group{group}'
    before, after = [read(root / kind / 'result.json') for kind in ['baseline', 'candidate']]
    assert len(before['rows']) == len(after['rows']) == 3
    assert read(root / 'baseline/start.json')['identity'] == read(root / 'candidate/start.json')['identity']
    for left, right in zip(before['rows'], after['rows']):
        assert left['mode'] == right['mode'] and left['batch'] == right['batch'] == batch_size
        assert len(left['seconds_per_scene_step']) == len(right['seconds_per_scene_step']) == 3
        if right['mode'] != 'rigid':
            assert left['admitted_timed_environment_steps'] == right['admitted_timed_environment_steps'] == 90 * batch_size
            assert left['surface_quadrature'] == right['surface_quadrature'] == 16
        timing.append({'baseline': left, 'candidate': right,
            'scene_cost_reduction_fraction': 1 - right['median_seconds'] / left['median_seconds'],
            'candidate_repeat_range_below_baseline': max(right['seconds_per_scene_step']) < min(left['seconds_per_scene_step'])})
    result = read(root / 'observations/result.json')
    assert result['passed'] and len(result['rows']) == 3
    for row in result['rows']:
        assert row['frames'] == 130 and [audit['repeats'] for audit in row['repeated_count_audits']] == [3, 3]
    observations.append(result)
    script = N / names[3] / f'workload-{group}.sh'
    keys = [hashlib.sha256(str(WORK.parent / source).encode()).hexdigest()[:16]
            for source in ['hard_contact_source_warps', 'hard_contact_traction_vectors']]
    for key in keys:
        assert f'GS_CACHE_FILE_PATH="$task_cache/{key}/genesis"' in script.read_text()
        assert f'QD_OFFLINE_CACHE_FILE_PATH="$task_cache/{key}/quadrants"' in script.read_text()
    caches.append({'group': group, 'source_keys': keys, 'script': identity(script)})
scene = read(N / names[3] / 'group2/full/result.json')
assert scene['passed'] and len(scene['rows']) == 130 and all(all(row['accepted']) for row in scene['rows'])
differences = read(N / names[3] / 'group2/differences/result.json')
assert differences['passed'] and differences['diagnostic_only'] and len(differences['rows']) == 130
motion = read(N / names[3] / 'group3/motion/result.json')
assert motion['passed'] and motion['pre_failure_frames_accepted'] == 428 and motion['original_frame428_refusal_reproduced']
bindings = []
for stem, campaign in [('profile_hard_franka', names[0]), ('compare_traction_recovery', names[1])]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{stem}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preflight = read(META / 'traction-vector-preflight-v1.json')
assert not preflight['submitted_on_failure'] and preflight['partial_output_directory_preserved']
assert not (N / 'hard-traction-controls-v1/submission.json').exists()
scope = (META / 'traction-vector-scope-v1.txt').read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'baseline_engine': BASELINE,
    'frozen_heads': heads, 'preserved_files': preserved, 'normal_and_hard_controls': controls,
    'profiles': profiles, 'paired_timings': timing, 'public_observations': observations,
    'full_field_diagnostic': differences, 'original_refusal_capture': motion,
    'source_bindings': bindings, 'configured_source_private_caches': caches,
    'unsubmitted_preflight': preflight, 'scope': scope, 'whole_goal_complete': False}
p = META / 'traction-vector-terminal-v1.json'
with p.open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect(names)
files += [p, HELPER_SOURCE, REPLAY_HELPER]
files += [META / stem for stem in ['traction-vector-scope-v1.txt', 'traction-vector-storage-probe-v1.txt',
    'traction-vector-start-v1.json', 'traction-vector-launch-v1.json', 'traction-vector-preflight-v1.json',
    'traction-vector-worktree-v1.log', 'traction-vector-engine-commit-v1.log', 'hard-traction-controls-v1-submit-v1.log',
    'hard-traction-controls-v1-tasks.json', 'document-traction-vectors-v1.py']]
files += [META / f'{name}-tasks.json' for name in names]
files += [N / 'hard-traction-controls-v1' / path for path in ['analysis-files.json', 'tasks.json']]

def input_receipts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            original = Path(value['path'])
            actual = identity(original)
            assert all(actual[key] == value[key] for key in ['bytes', 'sha256']), str(original)
            external.add(original.resolve())
        else:
            for item in value.values(): input_receipts(item)
    elif isinstance(value, list):
        for item in value: input_receipts(item)

for name in names:
    files += list((N / name).glob('extra-source*.json')) + list((N / name).glob('extra-source*.patch'))
    files += list((N / name).glob('group*/*/*.pstats'))
    for start in (N / name).glob('group*/**/start.json'): input_receipts(read(start))
for control in controls: input_receipts(control['complete_output_comparison'])
for row in differences['rows']: external.add(Path(row['baseline']['path']).resolve())
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('hard-traction-vectors-v1', files, sorted(external), scope)
(DEST / 'hard-traction-vectors-v1/README.md').write_text('# Boundary traction through vector contractions\n\n' + scope +
    '\n\nSee [the traction measurements](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#boundary-traction-vector-contractions). '
    'Four scientific campaigns are terminal. An initial incomplete sparse checkout failed before submission and its partial capture is preserved. '
    'All55 source heads and12 original unfinished files remain preserved. Scientific work used Slurm rtx-mid/image1_26, existing data-side Python and source-private node caches. '
    'The measured native source is pinned separately from the original task engine baseline.\n')
with (META / 'traction-vector-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
