"""Retain continuous sharp-edge fields, native evaluation and curved-shell source response."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_edge_farfield'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('4a8125c4c')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = [p.name for p in sorted(N.glob('hard-contact-application*')) if (p / 'submission.json').is_file()]
assert len(names) == 12
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
        assert code in ['0', '1'] and observed[f'{job}_{group}'] == expected, (name, group, code)
        task_states = []
        for task in tasks:
            path = root / task['name'] / 'result.json'
            result = read(path) if path.is_file() else None
            passed = bool(result and result.get('passed', False))
            if code == '0':
                assert passed, (name, group, task)
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1], 'tasks': task_states})
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
allowed = [ALIAS]
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('inspect_contact_application', 'hard-contact-application-response-v1'),
                         ('prepare_curved_collision', 'hard-contact-application-assets-v1'),
                         ('check_live_curved_hertz_franka', 'hard-contact-application-substeps-v1'),
                         ('study_far_edge_contact', 'hard-contact-application-geometry-v2'),
                         ('study_edge_wrench_allocation', 'hard-contact-application-law-update-v1'),
                         ('study_edge_source_response', 'hard-contact-application-law-update-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/edge-farfield-terminal-v1.json')['frozen_heads']
heads['hard_contact_edge_farfield'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
assets = read(N / 'hard-contact-application-assets-v1/group0/assets/result.json')
application = read(N / 'hard-contact-application-v2/group0/application/result.json')
assert application['passed'] and application['raw_contact_association_checked']
assert len(application['baseline_frame_exact_keys']) == 8
fixed = []
for group in range(3):
    root = N / f'hard-contact-application-response-v1/group{group}'
    allocation, response = read(root / 'allocation/result.json'), read(root / 'response/result.json')
    assert allocation['constrained']['qualified_constrained_minimum'] and response['passed']
    fixed.append({'allocation': allocation, 'response': response})
assert fixed[0]['response']['rows'][-1]['all_contact_allocation_response']['common_probes']['tensor_relative'] > .05
for row in fixed[1:]:
    assert row['response']['rows'][-1]['all_contact_allocation_response']['common_probes']['tensor_relative'] < .05
allocation = read(N / 'hard-contact-application-law-update-v1/group0/allocation2/result.json')
updated = read(N / 'hard-contact-application-law-update-v1/group0/update2/result.json')
comparison = read(N / 'hard-contact-application-law-update-v1/group1/comparison/result.json')
assert allocation['constrained']['qualified_constrained_minimum']
assert max(allocation['constrained']['relative_changes']) < .05
assert updated['wrench']['moment_residual_over_sum_Fa'] < 1e-8
assert updated['wrench']['force_residual_relative'] < 1e-8
assert updated['wrench']['minimum_friction_cone_margin_relative'] >= -1e-12
change = updated['rows'][-1]['all_contact_allocation_response']
assert change['volume_tensor_relative'] < .05 and change['common_probes']['tensor_relative'] < .05
assert change['common_probes']['vm_peak_relative'] < .1 and change['common_probes']['tensile_peak_relative'] < .1
assert comparison['passed'] and comparison['regular_displacement_identical']
assert all(row['maximum_point_tensor_relative'] == 0 for row in comparison['cases'])
scope = ('Actual raw collision association is exact: the ground application point lies half a penetration '
    '(183.591 micrometres) above its exact material witness, also used by the constraint Jacobian. '
    'Saved post-integration diagnostic queries are not same-contact geometry errors. A div32 collision '
    'mesh preserves the actual quadratic material and has1.960 micrometre sampled flat/P2 discrepancy. '
    'Its two samples per gripper region are grouped before pressure prediction, preserving source free moments. '
    'The all-region wrench/cone allocation needs at most2.965% force change. After support/exponent/center '
    'updates and one further force allocation, complete coarse-volume/near-probe fields change2.676%/1.794%, '
    'sample VM/tensile maxima change1.166%/1.996%, and scaled moment residual is4.97e-11. '
    'This passes the declared5% nearby-load sensitivity budget for this captured frame, not an independent '
    'physical stress accuracy claim. Original fixed-support case fails5%; div32 passes; four substeps add '
    'no benefit on that metric. All three actual native Franka variants still refuse at106 before edge '
    'binding, and their failures are retained. Geometry-v1 refined contacts1/2 are the same region and '
    'are geometry-only source diagnostics; geometry-v2 uses full grouped loads. Native binding, complete '
    'physical accuracy and accepted paired max/full throughput remain open. Sharp-edge point peaks are '
    'unbounded; reported peaks are declared positive-depth samples. Goal active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'application_provenance': application, 'refined_assets': assets, 'fixed_support_variants': fixed,
    'updated_law_allocation': allocation, 'updated_law_response': updated, 'baseline_helper_regression': comparison,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}

receipt_path = N / 'meta/contact-application-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path,
    N / 'meta/contact-application-refined-physical-v1.json',
    N / 'meta/contact-application-substeps-physical-v1.json']
external = set()
def add_identities(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys() and Path(value['path']).is_absolute():
            path = Path(value['path'])
            assert identity(path) == {key: value[key] for key in ['path', 'bytes', 'sha256']}
            external.add(path)
        for item in value.values():
            add_identities(item)
    elif isinstance(value, list):
        for item in value:
            add_identities(item)
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    for pattern in ['*.npz', '*.obj', '*.urdf']:
        external.update(root.glob('group*/**/' + pattern))
    for path in root.glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts:
            add_identities(read(path))

external -= set(files)
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-contact-application-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-contact-application-v1/README.md').write_text('# Collision application provenance and updated sharp-edge loads\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [derivation, measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#collision-application-points-and-updated-support). '
    'Replay captured tasks with pinned sources, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh data outputs and private node caches. Full arrays remain at hashed data paths. '
    'Original failures remain retained; source acceptance targets remain unchanged.\n')
with (N / 'meta/contact-application-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
