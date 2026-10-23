"""Publish bounded two-face elasticity controls and explicitly scoped field diagnostics."""
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
names = [f'box-corner-elastic428-v{v}' for v in range(1, 6)] + ['vector-halfspace-hertz-v1']
names += [f'box-corner-fields428-v{v}' for v in range(1, 4)] + ['box-corner-outage428-v1']
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
elastic = [read(N / f'box-corner-elastic428-v{v}/group0/elastic/result.json') for v in range(1, 5)]
assert not elastic[0]['finite_face_membership_passed'] and elastic[0]['invalid_face_force_fraction'] > 0.18
for result in elastic[1:] + [read(N / 'box-corner-outage428-v1/group0/elastic/result.json')]:
    assert result['finite_face_membership_passed'] and result['remote_boundary_force_fraction'] == 0
    assert result['kernel_relative_error'] < 2e-12
hertz = [read(N / f'vector-halfspace-hertz-v1/group{i}/{task}/result.json')
         for i, task in enumerate(['grid12q4', 'grid20q4', 'grid20q8'])]
assert all(r['pressure_10pct_depth_2pct_targets_met'] for r in hertz)
field = read(N / 'box-corner-outage428-v1/group1/fine/result.json')
quadrature = read(N / 'box-corner-outage428-v1/group0/target8/result.json')
assert all(row['reference_refinement']['tensor_relative_l2'] == 0 for row in field['rows'])
assert not any(row['tensor_5pct_peaks_10pct_met'] for row in field['rows'][:3])
assert field['rows'][-1]['tensor_5pct_peaks_10pct_met']
bindings = []
for module, campaign in [('study_box_corner_elasticity', 'box-corner-outage428-v1'),
                         ('compare_box_corner_fields', 'box-corner-outage428-v1'),
                         ('check_vector_halfspace_reference', 'vector-halfspace-hertz-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved.resolve()), 'current': identity(WORK / relative), 'exact_match': True})
size = int(subprocess.check_output(['du', '-sb', str(FALLBACK.parent.parent)], text=True).split()[0])
assert size < 4_000_000_000
scope = (META / 'box-corner-elastic428-scope-v1.txt').read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'elastic_refinements': elastic,
    'hertz_controls': hertz, 'fine_field': field, 'target_quadrature_control': quadrature,
    'fallback_bytes': size, 'fallback_bound_bytes': 4_000_000_000,
    'scope': scope, 'whole_goal_complete': False}
p = META / 'box-corner-elastic428-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect([name for name in names if name != 'box-corner-elastic428-v5'])
files += sources('box-corner-elastic428-v5') + terminal('box-corner-elastic428-v5')
files += [p, HELPER_SOURCE, REPLAY_HELPER, META / 'box-corner-elastic428-scope-v1.txt',
          META / 'storage-incident-v1.json', META / 'readonly-assembly-links-v1.json',
          META / 'box-corner-outage428-v1-tasks.json']
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
packet = freeze('box-corner-elastic428-v1', files, sorted(external), scope)
(DEST / 'box-corner-elastic428-v1/README.md').write_text('# Frictionless two-face elasticity and finite-depth stress diagnostics\n\n' + scope +
    '\n\nSee [reference and field sensitivity](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#two-face-elastic-reference-and-field-sensitivity). '
    'The captured engine remains06cf62f1c. Replays use Slurm rtx-mid, image1_26, the existing data-side Python environment and private node caches. '
    'The original infinite-plane branch and the data-quota initialization failure are preserved. '
    'Read-only campaign aliases are listed in the storage receipt; large original arrays remain on data. '
    'One small replacement campaign uses the existing user-authorized4GB home exception. '
    'The two-face moment-fit diagnostic uses reference-derived inputs and supplies no runtime physical qualification.\n')
with (META / 'box-corner-elastic428-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
