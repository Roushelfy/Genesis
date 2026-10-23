"""Check the delivered research change against preserved source and packet identities."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import re
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
W = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
T = W.parent / 'rigid-stress-recovery'

def git(*args, cwd=W):
    return subprocess.check_output(['git', *args], cwd=cwd)

work, task = git('rev-parse', 'HEAD').decode().strip(), git('rev-parse', 'HEAD', cwd=T).decode().strip()
assert not git('diff', task, work, '--', 'research/rigid_stress', 'DATA_LOCATION.md')
baseline = git('diff', '88f388c35dd21416428c4b1f035bb815fc1bf836', '6374b95c9cabf21a458eda8aea5d204aecf7566a', '--', 'genesis', 'tests', 'examples')
assert baseline == git('diff', task, work, '--', 'genesis', 'tests', 'examples')
start = json.loads((N / 'meta/edge-shell-exact-columns-start-v1.json').read_text())
for name, head in start['frozen_heads'].items():
    assert git('rev-parse', 'HEAD', cwd=W.parent / name).decode().strip() == head
preserved = 0
for name, data in start['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((W.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
packets = []
for name in ['sharp-edge-shell-integrals-v1', 'hard-batch2048-v1']:
    packet = Path('research/rigid_stress/evidence/20261019-qualified-ranking') / name
    manifest = json.loads(git('show', f'{work}:{packet}/manifest.json'))
    for row in manifest['files']:
        payload = git('show', f'{work}:{packet / row["copy"]}')
        assert len(payload) == row['bytes'] and hashlib.sha256(payload).hexdigest() == row['sha256']
        original = gzip.decompress(payload) if row['encoding'] == 'gzip' else payload
        assert len(original) == row['original_bytes'] and hashlib.sha256(original).hexdigest() == row['original_sha256']
    packets.append({'name': name, 'payloads': manifest['payloads'], 'bytes': manifest['payload_bytes']})
for name in ['SHARP_EDGE_SHELL_CONTROL_20261023.md', 'SHARP_EDGE_PRESSURE_20261022.md']:
    report = W / 'research/rigid_stress/docs' / name
    for target in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (report.parent / target.split('#')[0]).resolve().exists(), target
receipt = {
    'passed': True,
    'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'work_head': work,
    'task_head': task,
    'same_research_docs': True,
    'original_engine_baseline_difference_unchanged': True,
    'frozen_heads_preserved': len(start['frozen_heads']),
    'original_unfinished_files_preserved': preserved,
    'new_packets_verified_from_git': packets,
    'whole_goal_complete': False,
}
with (N / 'meta/edge-shell-integrals-delivery-v1.json').open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
print(json.dumps(receipt))
