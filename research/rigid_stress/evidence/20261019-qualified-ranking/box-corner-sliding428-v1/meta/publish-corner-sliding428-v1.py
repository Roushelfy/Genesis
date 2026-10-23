"""Freeze actual contact motion, force-cone bounds and limited sliding corner controls."""
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
ALIAS = '03ee8b02aeeae3d67e06b1ba308038a4db4ee44b'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['box-corner-motion428-v1'] + [f'box-corner-sliding428-v{i}' for i in range(1, 5)]
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
heads = read(N / 'meta/cooperative-roots-terminal-v1.json')['frozen_heads']
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
motion = read(N / names[0] / 'group0/motion/result.json')
assert motion['pre_failure_frames_accepted'] == 428 and motion['original_frame428_refusal_reproduced']
assert [row['step'] for row in motion['rows']] == [419, 420, 427, 428]
corner = motion['rows'][-1]
assert corner['accepted'] == [False, True, True, True] and corner['input_status'] == [32, 0, 0, 0]
assert len(corner['motion']) == 1
bound = corner['motion'][0]['minimum_side_force']
assert abs(bound['side_normal_lower_bound_n'] - 0.002118589938368334) < 1e-14
assert bound['primal_dual_gap_n'] < 1e-14
baseline = read(N / names[3] / 'group1/surface/result.json')
assert baseline['rows'][0]['normal_baseline_arrays_exact']
surface = read(N / names[4] / 'group0/surface/result.json')
assert len(surface['rows']) == 2
assert surface['rows'][0]['surface_ordering']['reversed_adjacent_loaded_pairs'] == 4
assert surface['rows'][0]['surface_metric']['active_principal_green_strain_range'][1] > 4
for row in surface['rows']:
    assert [r['fixed_force_reintegration']['target_order'] for r in row['fixed_force_reintegrations']] == [8, 16]
controls = {name: read(N / relative / 'result.json') for name, relative in [
    ('normal600', 'box-corner-sliding428-v3/group1/normal'),
    ('sliding600', 'box-corner-sliding428-v1/group1/sliding'),
    ('sliding1600_quadratic', 'box-corner-sliding428-v2/group1/sliding'),
    ('sliding1600_cubic', 'box-corner-sliding428-v3/group0/sliding')]}
for result in controls.values():
    assert result['finite_face_membership_passed'] and result['remote_boundary_force_fraction'] == 0
fields = {name: read(N / relative / 'result.json') for name, relative in [
    ('sliding600', 'box-corner-sliding428-v2/group0/fields'),
    ('sliding1600_quadratic', 'box-corner-sliding428-v2/group1/fields'),
    ('sliding1600_cubic', 'box-corner-sliding428-v3/group0/fields')]}
assert fields['sliding1600_cubic']['rows'][0]['reference_refinement']['sample_tensile_peak_relative_change'] > .2
bindings = []
for module, campaign in [('capture_corner_motion', names[0]), ('inspect_corner_sliding', names[4]),
                         ('study_box_corner_elasticity', names[3]), ('compare_box_corner_fields', names[3])]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope = (META / 'corner-sliding428-scope-v1.txt').read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'motion': motion,
    'controls': controls, 'surface_audit': surface, 'normal_baseline_arrays_exact': True,
    'conditional_fields': fields, 'scope': scope, 'physical_reference_qualified': False,
    'runtime_admission_changed': False, 'whole_goal_complete': False}
p = META / 'corner-sliding428-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect(names)
files += [p, HELPER_SOURCE, REPLAY_HELPER, META / 'corner-sliding428-scope-v1.txt',
          META / 'corner-motion-storage-probe-v1.txt']
files += [META / f'{name}-tasks.json' for name in names]
def input_receipts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            original = Path(value['path'])
            actual = identity(original)
            assert all(actual[k] == value[k] for k in ['bytes', 'sha256']), str(original)
            external.add(original.resolve())
        else:
            for item in value.values():
                input_receipts(item)
    elif isinstance(value, list):
        for item in value:
            input_receipts(item)
for name in names:
    for start in (N / name).glob('group*/**/start.json'):
        input_receipts(read(start))
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('box-corner-sliding428-v1', files, sorted(external), scope)
(DEST / 'box-corner-sliding428-v1/README.md').write_text('# Sharp-corner rigid motion and sliding controls\n\n' + scope +
    '\n\nSee [the contact-motion and sliding study](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#contact-motion-and-prescribed-sliding-control). '
    'All five campaigns are terminal. The original frame428 refusal, mesh sensitivity, ordering failures and DLPack fallback logs remain preserved. '
    'Current workers match their captured executed sources. Scientific work used Slurm rtx-mid, image1_26, the existing data-side physical environment and private node caches. '
    'Runtime arrays and logs stay under the restored data root. The49 engine sources and12 original unfinished files are unchanged. '
    'This is an optional local linear-elastic control with prescribed sliding; no production acceptance or physical-reference qualification follows.\n')
with (META / 'corner-sliding428-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
