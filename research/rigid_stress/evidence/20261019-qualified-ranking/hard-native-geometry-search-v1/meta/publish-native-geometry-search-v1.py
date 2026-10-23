"""Retain conservative geometry searches and admitted sharp-edge scene measurements."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_geometry_hierarchy'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = sorted(path.name for path in N.glob('hard-native-geometry-search-*') if (path / 'submission.json').is_file())
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        code = (root / 'returncode.txt').read_text().strip()
        expected = ['COMPLETED', '0:0'] if code == '0' else ['FAILED', '1:0']
        assert code in ['0', '1'] and observed[f'{job}_{group}'] == expected, (name, group, code, observed[f'{job}_{group}'])
        task_states = []
        for task in tasks:
            path = root / task['name'] / 'result.json'
            result = read(path) if path.is_file() else None
            passed = bool(result and result.get('passed', False))
            if code == '0': assert passed, (name, group, task)
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1], 'tasks': task_states})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(N / 'meta/native-friction-newton-terminal-v1.json')['frozen_heads']
for name in ['hard_contact_geometry_search', 'hard_contact_search_roundoff', 'hard_contact_geometry_hierarchy']:
    heads[name] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip()
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
controls = {}
for version in [2, 3]:
    controls[str(version)] = {}
    for group, backend in [(0, 'cpu'), (1, 'gpu')]:
        result = read(N / f'hard-native-geometry-search-controls-v{version}/group{group}/candidate/result.json')
        assert result['passed'] and result['source_binding']['passed']
        assert not result['source_binding']['mismatches']
        controls[str(version)][backend] = result
scene_timing = {}
for version in [1, 2]:
    scene_timing[str(version)] = []
    for group, batch_size in [(0, 4), (1, 64)]:
        result = read(N / f'hard-native-geometry-search-timing-v{version}/group{group}/paired-b{batch_size}/result.json')
        assert result['passed'] and len(result['rows']) == 3
        scene_timing[str(version)].extend(result['rows'])
profiles = {}
for version in [1, 2]:
    result = read(N / f'hard-native-geometry-search-profile-v{version}/group0/profile/result.json')
    assert result['passed'] and not result['primary_timing']
    profiles[str(version)] = result
scenes = {}
for version in [1, 2, 3]:
    scenes[str(version)] = {}
    for group, mode in [(0, 'full'), (1, 'max')]:
        result = read(N / f'hard-native-geometry-search-scene-v{version}/group{group}/{mode}/result.json')
        assert result['passed'] and len(result['rows']) == 130
        assert all(all(row['accepted']) for row in result['rows'])
        scenes[str(version)][mode] = {'frames': 130, 'admitted_environment_steps': 520}
observations = []
for group in [0, 1]:
    result = read(N / f'hard-native-geometry-search-observations-v1/group{group}/observations/result.json')
    assert result['passed']
    observations.append(result)
differences = {}
for campaign in ['hard-native-geometry-search-differences-v1', 'hard-native-geometry-search-differences-v2',
                 'hard-native-geometry-search-baseline-repeat-v1']:
    for path in sorted((N / campaign).glob('group*/*/result.json')):
        result = read(path)
        assert result['passed'] and result['diagnostic_only']
        fields = {}
        for row in result['rows']:
            for value in row['fields']:
                field = fields.setdefault(value['field'], {'maximum_absolute_difference': 0.,
                    'maximum_relative_l2_difference': 0., 'nonconformant_frames': 0, 'nonexact_frames': 0})
                field['maximum_absolute_difference'] = max(field['maximum_absolute_difference'], value['maximum_absolute_difference'])
                field['maximum_relative_l2_difference'] = max(field['maximum_relative_l2_difference'], value['relative_l2_difference'])
                field['nonconformant_frames'] += not value['conformant']
                field['nonexact_frames'] += not value['exact']
        differences[str(path.relative_to(N))] = {'frames': len(result['rows']),
            'all_fields_conformant': result['all_fields_conformant'], 'fields': fields}
bindings = []
for module, campaign in [('check_geometry_search', 'hard-native-geometry-search-controls-v3'),
                         ('compare_hard_frames', 'hard-native-geometry-search-differences-v2'),
                         ('compare_hard_observations', 'hard-native-geometry-search-observations-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope_path = N / 'meta/native-geometry-search-scope-v1.txt'
scope = scope_path.read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'geometry_controls': controls,
    'paired_scene_timing': scene_timing, 'separate_scene_profiles': profiles,
    'actual_scene_outcomes': scenes, 'public_observation_conformance': observations,
    'raw_field_differences': differences,
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/native-geometry-search-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files = [SCRIPT, HELPER_SOURCE, HELPER, REPLAY_HELPER, receipt_path, scope_path]
external = set()
def add_identities(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys() and Path(value['path']).is_absolute():
            path = Path(value['path'])
            assert identity(path) == {key: value[key] for key in ['path', 'bytes', 'sha256']}
            external.add(path)
        for item in value.values(): add_identities(item)
    elif isinstance(value, list):
        for item in value: add_identities(item)
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    for pattern in ['*.npz', '*.obj', '*.urdf', '*.pstats']:
        external.update(root.glob('group*/**/' + pattern))
    for path in root.glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts: add_identities(read(path))
large = {path for path in files if path.stat().st_size > 2*1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-geometry-search-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-geometry-search-v1/README.md').write_text('# Ordered contact geometry search acceleration\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is an experimental ancestor. '
    'See [measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#ordered-geometry-search-acceleration). '
    'Replay captured tasks with their pinned engine and analysis sources, Slurm rtx-mid, image1_26, '
    'the data-side Python environment and private node caches. Full arrays remain at hashed data paths. '
    'Failed boundary pruning and strict raw-array comparisons remain recorded. Physical accuracy and full-cycle qualification remain open.\n')
with (N / 'meta/native-geometry-search-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
