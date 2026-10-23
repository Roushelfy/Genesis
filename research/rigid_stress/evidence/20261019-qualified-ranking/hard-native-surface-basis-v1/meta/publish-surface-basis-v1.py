"""Retain surface-basis specialization trials, unchanged gates and matched scene measurements."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_surface_basis'
ALIAS = '38583bb2646db36de2e9853f634a9df12a52dcfb'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-surface-geometry-controls-v1', 'hard-native-surface-geometry-profile-v1',
         'hard-native-surface-unroll-controls-v1', 'hard-native-surface-unroll-profile-v1',
         'hard-native-surface-basis-tests-v1', 'hard-native-surface-unroll-timing-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        failed = name in names[:3]
        expected = ['FAILED', '1:0'] if failed else ['COMPLETED', '0:0']
        assert observed[f'{job}_{group}'] == expected, (name, group, observed.get(f'{job}_{group}'))
        assert (root / 'returncode.txt').read_text().strip() == ('1' if failed else '0')
        if not failed:
            for task in tasks:
                assert read(root / task['name'] / 'result.json')['passed']
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1]})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(N / 'meta/corner-sliding428-terminal-v1.json')['frozen_heads']
heads.update({'hard_contact_surface_geometry': '8b2996d211b5de80bbbd5c445c7ae0c3fb757ae0',
              'hard_contact_surface_unroll': '4fe6c6af1a1a33211d17b60fd322701df980da16',
              C3.name: ALIAS})
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
assert not subprocess.check_output(['git', 'diff', heads['hard_contact_surface_unroll'], ALIAS,
                                   '--', 'genesis', 'examples', 'pyproject.toml'], cwd=C3)
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
controls, pytest_reports = [], []
for group in range(2):
    control = read(N / names[2] / f'group{group}/curved-source/result.json')
    assert control['passed']
    controls.append(control)
    failed_log = (N / names[2] / f'group{group}/bounds/pytest.log').read_text()
    assert "missing 4 required positional arguments" in failed_log
    root = N / names[4] / f'group{group}/bounds'
    tests = ET.parse(root / 'tests.xml').getroot().findall('.//testcase')
    assert len(tests) == 7
    assert all(case.find('failure') is None and case.find('error') is None and case.find('skipped') is None for case in tests)
    pytest_reports.append({'backend': ['cpu', 'cuda'][group], 'passed_tests': len(tests),
                           'cases': [case.attrib for case in tests], 'xml': identity(root / 'tests.xml')})
profiles, kernel_means = {}, {}
for kind in ['baseline', 'candidate']:
    root = N / names[3] / 'group0' / kind
    profiles[kind] = read(root / 'result.json')
    assert profiles[kind]['passed'] and not profiles[kind]['primary_timing']
    kernel_means[kind] = {}
    for line in (root / 'kernels.txt').read_text().splitlines():
        match = re.search(r'\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+ms\]\s+(\S+)', line)
        if match:
            for key in ['kernel_hertz_residual_traction', 'kernel_resolve_sources', 'kernel_hertz_combined_field', 'kernel_hertz_planar']:
                if match[4].startswith(key + '_') and match[4].endswith('_range_for'):
                    kernel_means[kind].setdefault(key, []).append(float(match[2]))
timing, observations, configured_caches = [], [], []
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
    script = (N / names[5] / f'workload-{group}.sh').read_text()
    source_keys = [hashlib.sha256(str(WORK.parent / name).encode()).hexdigest()[:16]
                   for name in ['hard_contact_cooperative_roots', 'hard_contact_surface_basis']]
    assert source_keys[0] != source_keys[1]
    for key in source_keys:
        assert f'GS_CACHE_FILE_PATH="$task_cache/{key}/genesis"' in script
        assert f'QD_OFFLINE_CACHE_FILE_PATH="$task_cache/{key}/quadrants"' in script
    configured_caches.append({'group': group, 'source_keys': source_keys, 'script': identity(N / names[5] / f'workload-{group}.sh')})
scene = read(N / names[5] / 'group2/full/result.json')
assert scene['passed'] and len(scene['rows']) == 130 and all(all(row['accepted']) for row in scene['rows'])
differences = read(N / names[5] / 'group2/differences/result.json')
assert differences['passed'] and differences['diagnostic_only'] and len(differences['rows']) == 130
motion = read(N / names[5] / 'group3/motion/result.json')
assert motion['pre_failure_frames_accepted'] == 428 and motion['original_frame428_refusal_reproduced']
relative = Path('research/rigid_stress/shell_comparison/submit.py')
saved = N / names[5] / 'analysis-source' / relative
assert saved.read_bytes() == (WORK / relative).read_bytes()
scope = (META / 'surface-basis-scope-v1.txt').read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'controls': controls, 'pytest': pytest_reports,
    'profiles': profiles, 'kernel_means_ms': kernel_means, 'paired_timings': timing,
    'public_observations': observations, 'full_field_diagnostic': differences,
    'original_refusal_capture': motion, 'configured_source_private_caches': configured_caches,
    'source_bindings': [{'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True}],
    'profile_and_final_engine_code_exact': True, 'scope': scope, 'whole_goal_complete': False}
p = META / 'surface-basis-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect(names)
files += [p, HELPER_SOURCE, REPLAY_HELPER, META / 'surface-basis-scope-v1.txt',
          META / 'surface-geometry-storage-probe-v1.txt', META / 'surface-geometry-start-v1.json', saved]
files += [META / f'{name}-tasks.json' for name in names]
fixture = META / 'surface-basis-fixture-v1.patch'
with fixture.open('xb') as stream:
    stream.write(subprocess.check_output(['git', 'diff', heads['hard_contact_surface_unroll'], ALIAS, '--', 'tests'], cwd=C3))
files.append(fixture)
def input_receipts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            original = Path(value['path'])
            actual = identity(original)
            assert all(actual[k] == value[k] for k in ['bytes', 'sha256']), str(original)
            external.add(original.resolve())
        else:
            for item in value.values(): input_receipts(item)
    elif isinstance(value, list):
        for item in value: input_receipts(item)
for name in names:
    files += list((N / name).glob('extra-source*.json')) + list((N / name).glob('extra-source*.patch'))
    files += list((N / name).glob('group*/*/*.pstats')) + list((N / name).glob('group*/*/*.xml'))
    for start in (N / name).glob('group*/**/start.json'): input_receipts(read(start))
for row in differences['rows']: external.add(Path(row['baseline']['path']).resolve())
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('hard-native-surface-basis-v1', files, sorted(external), scope)
(DEST / 'hard-native-surface-basis-v1/README.md').write_text('# Fixed quadratic surface-basis specialization\n\n' + scope +
    '\n\nSee [the matched surface-basis measurements](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#fixed-quadratic-surface-basis-specialization). '
    'All six campaigns are terminal. Initial compile failures and stale test-fixture failures are preserved. '
    'The final fixture correction changes no engine implementation or assertion. All52 source heads and12 original unfinished files remain preserved. '
    'Scientific work used Slurm rtx-mid/image1_26, existing data-side Python and private node caches; primary source-paired runs additionally isolate Genesis/Quadrants caches by source path. '
    'No independent physical stress accuracy or full-cycle admission follows from these implementation controls.\n')
with (META / 'surface-basis-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
