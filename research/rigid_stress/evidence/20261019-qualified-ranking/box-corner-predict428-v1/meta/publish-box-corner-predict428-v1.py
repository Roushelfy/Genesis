"""Publish independent two-face scale estimates and the surface-orientation audit."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

ORIGINAL = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
FALLBACK = Path('/mnt/home/zhaofeng/project-data-fallback/workspace/Genesis/rigid-stress-recovery/runs/20261023-box-corner')
HELPER_SOURCE = FALLBACK / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
N = RUN = FALLBACK
META = N / 'meta'
HELPER = META / 'publish-canonical125-resilient-recovery-v1.py'
C3 = WORK.parent / 'hard_contact_boundary_lanes'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS == '06cf62f1c44a13465ac3373fb52eee013d8db44e'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['box-corner-predict428-v1', 'box-corner-predict428-v2', 'box-corner-surface428-v1', 'box-corner-surface428-v2']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        failed = name == 'box-corner-elastic428-v5'
        expected = ['FAILED', '1:0'] if failed else ['COMPLETED', '0:0']
        assert observed[f'{job}_{group}'] == expected
        if failed:
            assert not (root / 'elastic/start.json').exists()
            code = None
        else:
            code = (root / 'returncode.txt').read_text().strip()
            assert code == '0'
            for task in tasks:
                assert read(root / task['name'] / 'result.json')['passed']
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1],
                            'wrapper_return_code': code, 'output_initialization_failed': failed})
REPLAY_HELPER = META / 'publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(ORIGINAL / 'meta/box-edge-wrench428-terminal-v1.json')['frozen_heads']
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(ORIGINAL / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
predictions = [read(N / name / 'group0/prediction/result.json') for name in names[:2]]
surface = read(N / names[2] / 'group0/surface/result.json')
refined_surface = read(N / names[3] / 'group0/surface/result.json')
assert all(r['passed'] and len(r['rows']) == 2 for r in predictions)
assert predictions[1]['geometry_correction']['corrected_factor'] is None
assert len(predictions[1]['geometry_correction']['history']) == 8
for r in predictions:
    assert [row['model'] for row in r['rows']] == ['independent_strips', 'finite_face_cap']
    assert all(row['compatibility']['invalid_other_face_force_fraction'] > 0 for row in r['rows'])
    assert r['rows'][1]['depths'][-1]['tensor_5pct_peaks_10pct_met']
    assert not r['rows'][1]['depths'][0]['tensor_5pct_peaks_10pct_met']
assert surface['passed'] and len(surface['rows']) == 4
assert refined_surface['passed'] and len(refined_surface['rows']) == 3
bindings = []
for module, campaign in [('predict_box_corner_profiles', 'box-corner-predict428-v2'), ('inspect_box_corner_surface', 'box-corner-surface428-v2')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved.resolve()), 'current': identity(WORK / relative), 'exact_match': True})
size = int(subprocess.check_output(['du', '-sb', str(FALLBACK.parent.parent)], text=True).split()[0])
assert size < 4_000_000_000
scope = (META / 'box-corner-predict428-scope-v1.txt').read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'independent_predictions': predictions, 'surface_orientation_audit': surface, 'fixed_force_reintegration': refined_surface,
    'fallback_bytes': size, 'fallback_bound_bytes': 4_000_000_000,
    'scope': scope, 'whole_goal_complete': False}
p = META / 'box-corner-predict428-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect([name for name in names if name != 'box-corner-elastic428-v5'])
files += [p, HELPER_SOURCE, REPLAY_HELPER, META / 'box-corner-predict428-scope-v1.txt',
          META / 'storage-incident-v1.json', META / 'readonly-assembly-links-v1.json',
          META / 'box-corner-predict428-start-v1.json', META / 'box-corner-predict428-v1-tasks.json', META / 'box-corner-predict428-v2-tasks.json', META / 'box-corner-surface428-v1-tasks.json', META / 'box-corner-surface428-v2-tasks.json']
for name in names:
    for start in (N / name).glob('group*/*/start.json'):
        result = read(start)
        if 'inputs' in result:
            for row in result['inputs']:
                original = Path(row['path'])
                actual = identity(original)
                assert all(actual[key] == row[key] for key in ['bytes', 'sha256'])
                external.add(original.resolve())
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('box-corner-predict428-v1', files, sorted(external), scope)
(DEST / 'box-corner-predict428-v1/README.md').write_text('# Independent corner profiles and discrete surface geometry\n\n' + scope +
    '\n\nSee [independent scales and surface ordering](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#independent-face-scales-and-surface-ordering). '
    'The frozen engine remains06cf62f1c. Four terminal CPU campaigns use Slurm rtx-mid, image1_26, the existing data environment and private node caches. '
    'The existing user-authorized4GB home fallback holds only these small new runs and metadata; original fields remain on data. '
    'Pressure parameters use geometry/material/resultants only, and the optional bound search excludes reference stress errors. '
    'The independent candidates remain unadmitted and their original failed geometry/accuracy checks are preserved.\n')

with (META / 'box-corner-predict428-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
