"""Freeze complete B256/B1024 scene comparisons after both groups terminate."""

from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
campaign = 'hard-scene-large-batch-v1'
root = N / campaign
job = str(read(root / 'submission.json')['job']).split(';')[0]
accounting = subprocess.check_output(['sacct', '--array', '-j', job, '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
for group in range(2):
    assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0']
    assert (root / f'group{group}/returncode.txt').read_text().strip() == '0'
comparisons = [read(root / f'group{group}/comparison/result.json') for group in range(2)]
assert all(result['passed'] and not result['peak_failures'] for result in comparisons)
for module in ['live_hertz_inputs', 'live_hertz_scene', 'compare_hertz_scene']:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    assert (WORK / relative).read_bytes() == (root / 'analysis-source' / relative).read_bytes()
C = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_reproducible')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C, text=True).strip()
assert read(root / 'source.json')['revision'] == '900110f112c9d1313ebac9b3d75ef076ad7e197d'
for row in read(root / 'source.json')['files']:
    current = identity(C / row['path'])
    assert all(current[key] == row[key] for key in ['bytes', 'sha256'])
receipt = {'passed': True, 'remote_goal_commit': 'c1c82eea667f986b7cf494eda5230904a1a752a7',
           'scheduler_accounting': accounting, 'comparisons': comparisons,
           'engine_commit': '900110f112c9d1313ebac9b3d75ef076ad7e197d',
           'previous_scene_packet': 'hard-scene-transfer-v1',
           'scope': 'Complete 720-step paired pure-rigid/max/full moving Franka scenes at B256/B1024, three warmed repeats, same independently evolving conditions and unchanged complete-history input/status/source/peak/trajectory gates. Research-only readback and synchronized phase profiles occur in separate audited rollouts. Shared-operator preparation uses the already-tested reproducible variant. Raw aggregate rates include rejected loads and do not qualify useful physical shell stress. Large full fields and trajectories remain hashed on data; no active hollow-shell study output is included.',
           'whole_goal_complete': False}
terminal_path = N / 'meta/hard-scene-large-batch-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect([campaign])
files.extend([HELPER_SOURCE, terminal_path, N / 'meta/hard-scene-next-phase-launch-v1.json'])
files.extend(root.glob('group*/**/*.jsonl.gz'))
external.update(root.glob('group*/**/*.obj'))
large = {path for path in files if path.stat().st_size > 2 * 1024 * 1024}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-scene-large-batch-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text(
    '# Larger complete Hertz stress scenes\n\n' + receipt['scope'] +
    '\n\nReplay the captured task groups with the existing Slurm submitter in `rtx-mid`, container `1_26`, the pinned scientific environment and explicit fresh data-side output/cache paths. Use the engine commit in the terminal receipt and captured workers. Original condition banks and input-only receipts remain hashed and separately published. Decode gzip payloads into a fresh data directory; this packet is immutable. Exact acceptance is a numerical/discrete gate; proportional shear, finite-face/source-moment rejections and unresolved near-field/shell accuracy remain visible. Host peak RSS is scheduler accounting, rather than GPU or declared logical tensor storage.\n')
with (N / 'meta/hard-scene-large-batch-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
