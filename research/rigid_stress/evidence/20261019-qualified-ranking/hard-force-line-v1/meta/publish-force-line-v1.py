"""Retain curved force-line repair, sharp-edge diagnostics and actual geometry direction."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_force_line'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('601cde58f')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-far-edge-contact-v1', 'hard-far-edge-contact-v2', 'hard-force-line-controls-v1', 'hard-force-line-baseline-v1', 'hard-force-line-franka-v1', 'hard-force-line-scene-comparison-v1']
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
allowed = [ALIAS, '3025c965b']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('check_curved_force_line', 'hard-force-line-controls-v1'), ('study_far_edge_contact', 'hard-far-edge-contact-v2'), ('inspect_curved_walk_scene', 'hard-force-line-scene-comparison-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/curved-walk-terminal-v1.json')['frozen_heads']
heads['hard_contact_force_line'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha

controls = [read(N / f'hard-force-line-controls-v1/group{i}/force-line/result.json') for i in [0, 1]]
assert all(result['passed'] and result['free_couple_isolated_refusal_checked'] for result in controls)
assert all(row['input_status'] == [0, 0, 0, 0] for result in controls for row in result['rows'])
negative = read(N / 'hard-force-line-baseline-v1/group0/force-line/result.json')
assert negative['passed'] and negative['baseline_failure_expected']
assert all(row['input_status'] == [0, 4, 4, 4] for row in negative['rows'])
scenes = []
for group in [0, 1]:
    result = read(N / f'hard-force-line-franka-v1/group{group}/franka/result.json')
    rows = result['rows']
    assert not result['passed'] and all(all(row['accepted']) for row in rows[:-1])
    assert rows[-1]['step'] == (108 if group == 0 else 106)
    row = rows[-1]
    scenes.append({'group': group, 'collision_divisions': 0 if group == 0 else 8,
        'frames': len(rows), 'fully_accepted_prefix_steps': len(rows)-1,
        'first_invalid_step': row['step'], 'accepted': row['accepted'],
        'input_status': row['input_status'], 'recovery_status': row['recovery_status']})
comparisons = [read(N / f'hard-force-line-scene-comparison-v1/group0/comparison-{i}/result.json') for i in [0, 1]]
assert all(row['passed'] and row['rigid_inputs_and_acceptance_exact'] for row in comparisons)
edge = [read(N / f'hard-far-edge-contact-v{v}/group{g}/edge/result.json') for v in [1, 2] for g in [0, 1]]
assert all(row['passed'] and not row['global_surface_peak_qualified'] and not row['native_scene_admitted'] for row in edge)
scope = ('Curved force-line projection preserves the resultant moment with known consistent root frames. '
    'CPU/CUDA independent P2 controls and isolated free-couple refusal pass; the frozen C17 tangential cases fail. '
    'Actual original/refined Franka still rejects at108/106; full saved-frame inputs/acceptance match and field '
    'differences remain below the original budget. Actual-P2 sharp-edge normal-only reference and frozen trial '
    'refinement are retained as diagnostics, with unresolved surface peaks and full-force moment mismatch. '
    'The user selected actual fingertip fillet/chamfer dimensions; numeric dimensions remain pending. '
    'Final collision/pressure geometry must share documented dimensions. No accepted complete-grasp throughput, '
    'coupled-friction pressure, complete hollow-shell accuracy or whole-goal completion is claimed.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'force_line_controls': controls, 'negative_control': negative, 'actual_scene_admission': scenes,
    'scene_comparisons': comparisons, 'sharp_edge_diagnostics': edge,
    'user_geometry_direction': 'Use actual fillet/chamfer dimensions; numeric radius/dimensions still pending.',
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/force-line-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = set()
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    external.update(root.glob('group*/**/*.npz'))
    for path in root.glob('group*/**/start.json'):
        start = read(path)
        for item in start.get('inputs', []) + start.get('collision_assets', []):
            external.add(Path(item['path']))
        for key in ['mesh', 'collision_urdf', 'conditions', 'input']:
            if start.get(key) is not None:
                external.add(Path(start[key]['path']))
for comparison in comparisons:
    external.update(Path(item['path']) for row in comparison['frames'] for item in [row['before'], row['after']])
geometry = N / 'assets/franka-fingertip-spec-v1'
assert read(geometry / 'inspection.json')['passed']
files.extend(path for path in geometry.iterdir() if path.suffix in ['.py', '.sh', '.json', '.log'])
external.update(geometry.glob('*.png'))
external.update(geometry.glob('*.pdf'))
external.update(Path(row['path']) for row in read(geometry / 'inspection.json')['meshes'])
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-force-line-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-force-line-v1/README.md').write_text('# Curved force lines and actual edge geometry\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor; '
    'C17 remains the frozen negative control. See '
    '[measurements and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#curved-force-line-projection-and-edge-geometry). '
    'Replay captured tasks with pinned engine source, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh data outputs and private node caches. Original FAILED states and captures remain retained. '
    'Full arrays, visual-mesh sections and the manufacturer document remain at hashed data paths.\n')
with (N / 'meta/force-line-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
