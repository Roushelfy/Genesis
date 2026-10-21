"""Freeze executed moving-scene transfer, coarse stress and independent bridge controls."""
from pathlib import Path
import ast
import gzip
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
campaigns = ['hard-scene-pilot-v1', 'hard-scene-check-v1', 'hard-scene-scaling-v1',
             'hard-scene-replay-capture-v1', 'hard-scene-replay-v1',
             'hard-scene-stable-v3', 'hard-scene-preparation-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
results = []
for campaign, job in zip(campaigns, jobs):
    root = N / campaign
    for group, tasks in enumerate(read(root / 'tasks.json')):
        failed = (campaign == 'hard-scene-scaling-v1' or campaign == 'hard-scene-replay-v1' and group == 0
                  or campaign == 'hard-scene-stable-v3' and group == 1)
        assert observed[f'{job}_{group}'] == (['FAILED', '1:0'] if failed else ['COMPLETED', '0:0']), (job, group, observed)
        assert (root / f'group{group}/returncode.txt').read_text().strip() == ('1' if failed else '0')
        for task in tasks:
            path = root / f'group{group}' / task['name'] / 'result.json'
            if campaign == 'hard-scene-scaling-v1' and task['name'] == 'comparison':
                assert not path.exists()
                continue
            result = read(path)
            if campaign == 'hard-scene-replay-v1' and group == 0:
                assert not result['passed'] and len(result['peak_failures']) == 1
                assert (result['peak_failures'][0]['step'], result['peak_failures'][0]['env']) == (554, 19)
            elif campaign == 'hard-scene-stable-v3' and group == 1:
                assert not result['passed'] and result['exit_code'] == 1
            else:
                assert result.get('passed', result.get('execution_completed', False)), path
            results.append(identity(path))
bindings = []
for module in ['live_hertz_inputs', 'live_hertz_scene', 'check_hertz_scene', 'compare_hertz_scene',
               'study_hertz_scene_replay', 'measure_stress_preparation']:
    rel = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured_path = N / 'hard-scene-preparation-v1/analysis-source' / rel
    assert captured_path.read_bytes() == (WORK / rel).read_bytes()
    bindings.append(identity(captured_path))
comparisons = [read(N / f'hard-scene-stable-v3/group{i}/comparison/result.json') for i in [4, 5]]
assert all(result['passed'] and not result['peak_failures'] for result in comparisons)
controls = [read(N / f'hard-scene-stable-v3/group3/{backend}/result.json') for backend in ['cpu', 'gpu']]
replay = read(N / 'hard-scene-stable-v3/group3/replay/result.json')
assert all(row['maximum_absolute_difference'] == 0 for row in replay['independent_preparation_differences'])
C = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_reproducible')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C, text=True).strip()
for row in read(N / 'hard-scene-stable-v3/source.json')['files']:
    current = identity(C / row['path'])
    assert all(current[key] == row[key] for key in ['bytes', 'sha256'])
assert (C / 'tests/rigid/test_stress.py').read_bytes() == (N / 'meta/hard-scene-stable-test_stress-v1.py').read_bytes()
test_dependencies = []
research_imports = []
for relative in ['tests/rigid/test_stress.py', 'tests/rigid/test_stress_independent_fem.py',
                 'tests/rigid/test_stress_nonconvex.py', 'tests/conftest.py', 'tests/utils/assertions.py']:
    test_dependencies.append(identity(C / relative))
    for node in ast.walk(ast.parse((C / relative).read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('research.rigid_stress'):
            research_imports.append(node.module)
        elif isinstance(node, ast.Import):
            research_imports.extend(item.name for item in node.names if item.name.startswith('research.rigid_stress'))
research_test_sources = []
research_seen = set()
analysis_root = N / 'hard-scene-stable-v3/analysis-source'
analysis_inventory = {row['path']: row for row in read(N / 'hard-scene-stable-v3/analysis-files.json')}
while research_imports:
    module = research_imports.pop()
    relative = Path(*module.split('.')).with_suffix('.py')
    path = analysis_root / relative
    if module in research_seen or not path.is_file():
        continue
    research_seen.add(module)
    current = identity(path)
    assert all(current[key] == analysis_inventory[str(relative)][key] for key in ['bytes', 'sha256'])
    research_test_sources.append(path)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            base = '.'.join(module.split('.')[:-node.level]) if node.level else ''
            target = '.'.join(part for part in [base, node.module] if part)
            if target.startswith('research.rigid_stress'):
                research_imports.append(target)
                research_imports.extend(target + '.' + item.name for item in node.names)
        elif isinstance(node, ast.Import):
            research_imports.extend(item.name for item in node.names if item.name.startswith('research.rigid_stress'))
windows = []
for batch, group in [(4, 4), (64, 5)]:
    with gzip.open(N / f'hard-scene-stable-v3/group{group}/max/history.jsonl.gz', 'rt') as stream:
        history = [json.loads(line) for line in stream]
    for start in range(0, 720, 120):
        loaded, admitted = 0, 0
        for row in history[start:start+120]:
            envs = {env for env, _ in row['source_indices']}
            loaded += len(envs)
            admitted += sum(row['scene_status'][env] == 0 for env in envs)
        windows.append({'batch': batch, 'first_step': start, 'last_step': start+119,
                        'loaded': loaded, 'admitted_loaded': admitted})
receipt = {
    'passed': True, 'remote_goal_commit': 'c1c82eea667f986b7cf494eda5230904a1a752a7',
    'scheduler_accounting': accounting, 'terminal_results': results, 'current_sources_match_executed': bindings,
    'comparisons': comparisons, 'controls': controls, 'fixed_step_windows': windows,
    'fixed_input_replay_before': read(N / 'hard-scene-replay-v1/group1/replay/result.json'),
    'fixed_input_replay_after': replay,
    'original_comparisons': [read(N / 'hard-scene-check-v1/group2/comparison/result.json'), read(N / 'hard-scene-replay-v1/group0/comparison/result.json')],
    'preparation': {name: read(N / f'hard-scene-preparation-v1/group0/{name}/result.json') for name in ['baseline', 'stable']},
    'B64_original_cross_output_peak_gate_passed': False,
    'B64_fixed_cross_output_peak_gate_passed': True,
    'engine_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C, text=True).strip(),
    'engine_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD:genesis'], cwd=C, text=True).strip(),
    'test_dependencies_in_pinned_engine_commit': test_dependencies,
    'captured_research_test_dependencies': [identity(path) for path in research_test_sources],
    'scope': 'Actual B4/B64 moving Franka native geometry, witnessed Hertz transfer, original-wrench/facet-cone acceptance, coarse elastic recovery and max/full output, paired rigid costs, every-step comparisons and three-environment CPU/GPU independent P2 controls. The original B64 peak failure and old-engine regression failure are preserved; fixed-order shared-operator preparation passes the unchanged complete scene comparison and exact independent-build operator checks. Rejected observations remain NaN while diagnostic rigid dynamics continues. Proportional shear and unresolved near-field/shell accuracy prevent public estimator promotion; strict discrete admission is not physical stress qualification.',
    'whole_goal_complete': False,
}
terminal_path = N / 'meta/hard-scene-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend(research_test_sources)
files.extend([HELPER_SOURCE, terminal_path, N / 'meta/hard-scene-pilot-start-v1.json',
              N / 'meta/document-hard-scene-v1.py', N / 'hard-scene-scaling-v1/peak-conformance-diagnostic.json',
              N / 'meta/hard-scene-stable-test_stress-v1.py'])
for version in [1, 2]:
    files.extend(N / f'hard-scene-stable-v{version}' / name for name in ['tasks.json', 'analysis-files.json', 'submission-failure.json'])
for campaign in campaigns:
    files.extend((N / campaign).glob('group*/**/*.jsonl.gz'))
    files.extend((N / campaign).glob('extra-source*.*'))
    files.extend((N / campaign).glob('group*/**/*.xml'))
    external.update((N / campaign).glob('group*/**/*.obj'))
large = {path for path in files if path.stat().st_size > 2 * 1024 * 1024}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-scene-transfer-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text(
    '# Moving-scene Hertz transfer and coarse max/full output\n\n' + receipt['scope'] +
    '\n\nReplay the captured tasks with the existing Slurm submitter in `rtx-mid`, container `1_26`, the pinned scientific environment, fresh explicit data-side output paths and private node-local caches. Baseline `260efa7da` equals the engine tree of public task commit `aa803cdb9`; the fixed variant is its captured source patch and the selected engine commit in the terminal receipt. Captured workers and the copied regression test are authoritative. Decode gzip payloads into a fresh data directory; this packet is immutable. Complete fields, trajectories and generated meshes remain hashed on data. Fixed step windows describe rollout time, not identified finger-contact phases. Original B64 and old-engine regression failures remain failed; the fixed variant passes their original acceptance budgets. No physical-model or comparison tolerance was relaxed.\n')
with (N / 'meta/hard-scene-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
