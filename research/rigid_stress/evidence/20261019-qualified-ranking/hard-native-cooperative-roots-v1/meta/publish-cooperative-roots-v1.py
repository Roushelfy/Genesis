"""Retain planar-stage attribution and source-paired cooperative root measurements."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_cooperative_roots'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS == '03ee8b02aeeae3d67e06b1ba308038a4db4ee44b'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-planar-stages-v1', 'hard-native-planar-stages-v2', 'hard-native-planar-stages-v3',
         'hard-native-cooperative-roots-control-v1', 'hard-native-cooperative-roots-geometry-v2',
         'hard-native-cooperative-roots-timing-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        failed = name in names[:2] or (name == names[3] and group == 0)
        expected = ['FAILED', '1:0'] if failed else ['COMPLETED', '0:0']
        assert observed[f'{job}_{group}'] == expected, (name, group, observed[f'{job}_{group}'])
        code = (root / 'returncode.txt').read_text().strip()
        assert code == ('1' if failed else '0')
        if not failed:
            for task in tasks:
                assert read(root / task['name'] / 'result.json')['passed']
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1],
                            'wrapper_return_code': code})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(N / 'meta/box-edge-wrench428-terminal-v1.json')['frozen_heads']
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
heads[C3.name] = ALIAS
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
profiles = {name: read(N / name / path / 'result.json') for name, path in [
    (names[2], 'group0/profile'), (names[3], 'group1/profile')]}
for result in profiles.values():
    assert result['passed'] and not result['primary_timing']
    probe = result['planar_stage_probes']
    assert probe['passed'] and probe['planar_queries'] == 301 and probe['box_queries'] == 181
    assert max(probe['maximum_center_normal_curvature_residual_differences']) <= 1e-12
geometry = [read(N / names[4] / f'group{i}/geometry/result.json') for i in range(2)]
for result in geometry:
    assert result['passed'] and result['original_source_conformance'] and not result['shared_feature_face_aliases']
    assert sum(row['queries'] for row in result['rows']) == 1152
assert geometry[1]['cooperative_roots_exact_face_choices']
timing, observations = [], []
for group, batch_size in enumerate([4, 64]):
    root = N / names[5] / f'group{group}'
    before, after = [read(root / kind / 'result.json') for kind in ['baseline', 'candidate']]
    assert before['passed'] and after['passed'] and len(before['rows']) == len(after['rows']) == 3
    for left, right in zip(before['rows'], after['rows']):
        assert left['mode'] == right['mode'] and left['batch'] == right['batch'] == batch_size
        if right['mode'] != 'rigid':
            assert left['admitted_timed_environment_steps'] == right['admitted_timed_environment_steps'] == 90 * batch_size
            assert left['loaded_timed_environment_steps'] == right['loaded_timed_environment_steps']
            assert left['sharp_timed_environment_steps'] == right['sharp_timed_environment_steps']
        timing.append({'baseline': left, 'candidate': right,
                      'scene_cost_reduction_fraction': 1 - right['median_seconds'] / left['median_seconds']})
    result = read(root / 'observations/result.json')
    assert result['passed'] and len(result['rows']) == 3
    observations.append(result)
scene = read(N / names[5] / 'group2/full/result.json')
assert scene['passed'] and len(scene['rows']) == 130 and all(all(row['accepted']) for row in scene['rows'])
differences = read(N / names[5] / 'group2/differences/result.json')
assert differences['passed'] and differences['diagnostic_only'] and len(differences['rows']) == 130
bindings = []
for module, campaign in [('planar_stage_probes', names[3]), ('profile_hard_franka', names[3]), ('check_geometry_search', names[4])]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved.resolve()), 'current': identity(WORK / relative), 'exact_match': True})
scope = (META / 'cooperative-roots-scope-v1.txt').read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'profiles': profiles,
    'geometry_controls': geometry, 'paired_timings': timing, 'public_observations': observations,
    'full_field_diagnostic': differences, 'full_frames': 130, 'scope': scope, 'whole_goal_complete': False}
p = META / 'cooperative-roots-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect(names)
files += [p, HELPER_SOURCE, REPLAY_HELPER, META / 'cooperative-roots-scope-v1.txt',
          META / 'storage-probe-planar-stages-v1.txt', META / 'hard-native-planar-stages-start-v1.json',
          META / 'cooperative-roots-validation-start-v1.json']
files += list(META.glob('hard-native-cooperative-roots-*-tasks.json'))
files += [META / 'hard-native-planar-stages-v1-tasks.json']
for name in names:
    files += list((N / name).glob('extra-source*.json')) + list((N / name).glob('extra-source*.patch'))
    files += list((N / name).glob('group*/*/*.pstats'))
    for start in (N / name).glob('group*/*/start.json'):
        result = read(start)
        input_rows = list(result.get('inputs', []))
        for key in ['baseline', 'candidate']:
            if isinstance(result.get(key), dict) and 'sha256' in result[key]:
                input_rows.append(result[key])
        for row in input_rows:
            original = Path(row['path'])
            actual = identity(original)
            assert all(actual[key] == row[key] for key in ['bytes', 'sha256'])
            external.add(original.resolve())
for row in differences['rows']:
    external.add(Path(row['baseline']['path']).resolve())
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('hard-native-cooperative-roots-v1', files, sorted(external), scope)
(DEST / 'hard-native-cooperative-roots-v1/README.md').write_text('# Cooperative planar normal-root searches\n\n' + scope +
    '\n\nSee [the measured planar search](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#cooperative-planar-normal-root-search). '
    'All six campaigns are terminal and their original failures are retained. Engine03ee8b02a is the isolated measured variant, '
    'with CPU serial behavior and exact GPU face-choice controls. New arrays, profiles, logs and metadata return to the restored data mount. '
    'Scientific execution uses Slurm rtx-mid, image1_26, the existing physical environment and private node caches. '
    'The short admitted closing window supplies implementation timing and conformance evidence. Full-cycle admission and physical contact accuracy remain open.\n')
with (META / 'cooperative-roots-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
