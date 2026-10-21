"""Freeze cache ablations, device traces and matched large-batch input controls."""

from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
A = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_bounds')
O = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_optimized')
campaigns = ['hard-input-cache-v1', 'hard-input-trace-v1', 'hard-input-scaling-v1',
             'hard-input-scaling-baseline-v1', 'hard-input-trace-v2', 'hard-input-cache-v2',
             'hard-input-scaling-check-v1', 'hard-input-bounds-v1',
             'hard-input-bounds-trace-v1', 'hard-input-bounds-paired-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
results = []
for campaign, job in zip(campaigns, jobs):
    root = N / campaign
    failed = campaign == 'hard-input-trace-v1'
    for group, tasks in enumerate(read(root / 'tasks.json')):
        assert observed[f'{job}_{group}'] == (['FAILED', '1:0'] if failed else ['COMPLETED', '0:0'])
        assert (root / f'group{group}/returncode.txt').read_text().strip() == ('1' if failed else '0')
        if not failed:
            for task in tasks:
                path = root / f'group{group}' / task['name'] / 'result.json'
                result = read(path)
                assert result.get('passed', result.get('execution_completed', False)), path
                if 'motion_matches' in result:
                    assert result['motion_matches'], path
                results.append(identity(path))
final_bindings = []
for module in ['live_hertz_inputs', 'check_hertz_regions', 'check_precomputed_surface']:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured_path = N / 'hard-input-bounds-v1/analysis-source' / relative
    assert captured_path.read_bytes() == (WORK / relative).read_bytes()
    final_bindings.append(identity(captured_path))
for module in ['trace_hertz_inputs', 'compare_hertz_batch_prefix', 'compare_hertz_implementations']:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured_path = N / 'hard-input-bounds-trace-v1/analysis-source' / relative
    assert captured_path.read_bytes() == (WORK / relative).read_bytes()
    final_bindings.append(identity(captured_path))
relative = 'genesis/engine/solvers/rigid/stress/curvature.py'
assert (A / relative).read_bytes() == (WORK / relative).read_bytes()
current = identity(A / relative)
expected = next(row for row in read(N / 'hard-input-bounds-v1/source.json')['files'] if row['path'] == relative)
assert all(current[key] == expected[key] for key in ['bytes', 'sha256'])
final_bindings.append(current)
summary = read(N / 'meta/hard-input-scaling-summary-v1.json')
assert summary['passed']
paired = [read(N / f'hard-input-bounds-paired-v1/group0/{name}_comparison/result.json')
          for name in ['baseline', 'bounds']]
receipt = {
    'passed': True, 'remote_goal_commit': 'c1c82eea667f986b7cf494eda5230904a1a752a7',
    'scheduler_accounting': accounting, 'terminal_results': results,
    'current_sources_match_executed': final_bindings,
    'cache_candidate_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=O, text=True).strip(),
    'selected_engine_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=A, text=True).strip(),
    'selected_change': 'Shared squared face bounds for the ordered nearest-surface query only. Original exhaustive normal-root search, active adapter API, grouping and numerical acceptance remain in use. Temporal hint/cached-root variants are preserved as experiments, not promoted.',
    'same_allocation_B256': [{'variant': name, 'timing': result['timing'],
                            'added_seconds_per_step': result['incremental_scene_step_seconds']}
                           for name, result in zip(['published', 'closest_bounds'], paired)],
    'failed_attempt': 'Both initial trace runs export CUDA traces, then fail in the memory-summary writer because vars(cache) includes a Quadrants internal dictionary. The next writer enumerates declared tensor fields; the final baseline/bounds writer uses the existing iter_data utility. Original failed logs/sources/traces remain retained.',
    'scope': 'Native contact-input cache and conservative-search ablations, CPU/GPU geometry and lifecycle controls, exact every-step grouping and matched source/trajectory comparisons, B4/B64/B256/B1024 independent environments, cross-batch prefix isolation, inclusive warmed timing, device launch traces and logical field memory. Only the small nearest-surface bound change is selected. Input-only: no traction transfer, complete/max stress, physical-pressure qualification or friction-history admission.',
    'whole_goal_complete': False,
}
terminal_path = N / 'meta/hard-input-cache-and-scaling-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path,
    N / 'meta/document-hard-input-cache-and-scaling-v1.py',
    N / 'meta/summarize-hard-input-traces-v1.py', N / 'meta/summarize-hard-input-scaling-v1.py',
    N / 'meta/hard-input-scaling-summary-v1.json', N / 'meta/hard-input-cache-counts-v1.json',
    N / 'meta/hard-input-trace-summary-v1.json', N / 'meta/hard-input-trace-summary-v2.json',
    N / 'meta/hard-input-bounds-trace-summary-v1.json',
    N / 'meta/hard-input-cache-v1-final-source.json', N / 'meta/hard-input-cache-v1-final-source.patch',
    N / 'meta/hard-input-cache-v2-final-source.json', N / 'meta/hard-input-cache-v2-final-source.patch'])
for campaign in campaigns:
    root = N / campaign
    files.extend(root.glob('extra-source*.*'))
    # The task named cache is a numerical control, not a compiler cache directory.
    files.extend(root.glob('group*/cache/*.json'))
    files.extend(root.glob('group*/**/*.jsonl.gz'))
    external.update(root.glob('group*/**/*.obj'))
# Complete large traces, histories and expanded snapshot summaries stay on data.
large = {path for path in files if path.name == 'trace.json' or path.stat().st_size > 2 * 1024 * 1024}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-input-cache-and-scaling-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text(
    '# Contact-input cache ablations, device traces and batch scaling\n\n' + receipt['scope'] +
    '\n\nThe manifest retains original trace-writer failures, exact task/import/source inventories, compact terminal results and all external data hashes. Large histories, raw Chrome traces, expanded snapshot summaries, NPZ arrays and generated meshes stay on data. The cache variants are retained through their captured source patches and worker modules. The selected nearest-surface-only variant uses the existing live adapter. Decode gzip payloads into a fresh data directory for replay; do not edit this immutable packet.\n\n' +
    'Use the recorded `tasks.json`, `workload-*.sh` and source inventory/patch for each variant. Source patches apply to the engine at public task commit `ccb703cb30334602f2ce8ed6f9b1bfe5326380bf`; the experiment worktrees started from an equivalent engine tree. Worker snapshots are authoritative where active research adapters differ. All numerical execution uses the existing Slurm helper, `rtx-mid`, container `1_26`, explicit data outputs and private node-local caches.\n')
with (N / 'meta/hard-input-cache-and-scaling-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
