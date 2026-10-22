"""Retain geometry sensitivity, parallel boundary fields and matched native costs."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_parallel_boundary'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
formatting = read(N / 'meta/parallel-boundary-format-v1.json')
MEASURED = formatting['measured_commit']
assert ALIAS == formatting['formatted_commit']
BASELINE = '2e39f5f752136de5d7616b3436063f1bf480814b'
assert ALIAS != BASELINE
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-complete-shell-timing-v1', 'hard-franka-contact-gap-v1',
         'hard-native-parallel-boundary-check-v1', 'hard-native-parallel-boundary-paired-v1',
         'hard-native-parallel-boundary-inspection-v1', 'hard-native-parallel-boundary-scene-v1',
         'hard-native-parallel-boundary-cost-check-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        assert (root / 'returncode.txt').read_text().strip() == '0', root
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0'], (name, group, observed)
        for task in tasks:
            assert read(root / task['name'] / 'result.json')['passed'], (name, group, task)
        states[name].append({'job': f'{job}_{group}', 'state': 'COMPLETED', 'exit_code': '0:0'})
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
old = "[root / 'source.json', *sorted(root.glob('extra-source*.json'))]"
assert old in replay_code
replay_code = replay_code.replace(old, "[root / 'source.json', *sorted(root.glob('extra-source*.json')), *sorted(root.glob('group*/**/*source.json'))]")
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
trees = {label: subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip()
         for label, sha in [('parallel', MEASURED), ('formatted', ALIAS), ('baseline', BASELINE)]}
changed = subprocess.check_output(['git', 'diff', '--name-only', MEASURED, ALIAS], cwd=C3, text=True).splitlines()
assert changed == ['genesis/engine/solvers/rigid/stress/hertz_recovery.py']
codes = [subprocess.check_output(['git', 'show', sha + ':' + changed[0]], cwd=C3).decode() for sha in [MEASURED, ALIAS]]
assert ast.dump(ast.parse(codes[0]), include_attributes=False) == ast.dump(ast.parse(codes[1]), include_attributes=False)
for row in replayed:
    label = ('baseline' if row['campaign'] in names[:2] or row['inventory'] in ['extra-source0.json', 'reference-source.json']
             else 'parallel')
    assert row['reconstructed_tree'] == trees[label], row
gap = read(N / names[1] / 'group0/gap/result.json')
assert gap['passed'] and gap['outside_faceted_material_probes'] == 64
inspection = read(N / names[4] / 'group0/inspection/result.json')
assert inspection['passed'] and len(inspection['rows']) == 8
costs = {kind: read(N / names[-1] / f'group0/{kind}/result.json') for kind in ['module', 'scene']}
assert all(result['passed'] for result in costs.values())
assert len(costs['module']['rows']) == 8 and len(costs['scene']['rows']) == 12
bindings = []
for module, campaign in [('study_franka_contact_gap', names[1]), ('time_native_hertz_recovery', names[3]),
                         ('compare_native_hertz_timing', names[-1])]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured_worker = N / campaign / 'analysis-source' / relative
    assert captured_worker.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(captured_worker), 'current': identity(WORK / relative), 'exact_match': True})
preservation = read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved']
preserved = 0
for name, data in preservation.items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/native-hertz-complete-boundary-terminal-v1.json')['frozen_heads']
heads['hard_contact_parallel_boundary'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
scope = ('Parallel angular quadrature preserves complete boundary coverage, pressure and acceptance gates. '
    'CPU/CUDA shell/normal controls, independent higher-order full fields, actual planar friction history and '
    'matched complete-field controls pass. Same-allocation moving facet-centered shell modules and actual '
    'sphere/flat friction scenes retain rigid/max/full costs and every-frame admission/numerical comparisons. '
    'The separate captured Franka gap control quantifies64/89 probes outside the faceted material and a '
    'large discrepancy between actual facet and smoothed quadratic gaps under fixed half-space compliance. '
    'That geometry sensitivity is not a physical hollow-shell pressure/stress qualification. Curved/inner '
    'sources, moving shell friction and accepted Franka throughput remain unresolved; the whole goal is active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states, 'captured_source_replay': replayed,
    'public_engine_alias': ALIAS, 'measured_engine_alias': MEASURED, 'formatting': formatting,
    'baseline': BASELINE, 'source_trees': trees, 'current_worker_bindings': bindings,
    'geometry_gap_control': gap, 'independent_fields': inspection, 'cost_comparisons': costs,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/parallel-boundary-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path,
         N / 'meta/parallel-boundary-format-v1.json', N / 'meta/parallel-boundary-format-v1.patch']
external = set()
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    files.extend(root.glob('group*/**/*source.patch'))
    external.update(root.glob('group*/**/*.npz'))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-parallel-boundary-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-parallel-boundary-v1/README.md').write_text('# Parallel boundary recovery and contact geometry\n\n' + scope +
    '\n\nSource [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') and baseline '
    '2e39f5f75 are public ancestors. See [results and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md). '
    'Use captured tasks with Slurm rtx-mid, image1_26, the existing data-side physical environment and fresh explicit data outputs. '
    'Caches use private node scratch. Full fields remain at hashed data paths; decode gzip separately. The goal remains active.\n')
with (N / 'meta/parallel-boundary-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
