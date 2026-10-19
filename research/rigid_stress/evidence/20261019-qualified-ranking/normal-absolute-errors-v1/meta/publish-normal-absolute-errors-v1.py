"""Copy terminal absolute-error evidence; leave all full arrays on data."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DEST = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/normal-absolute-errors-v1')
campaign = ROOT / 'pad-absolute-errors-cpu-v2'
result = json.loads((campaign / 'group0/audit/result.json').read_text())
assert result['passed'] and len(result['rows']) == 81 and len(result['details']) == 9
assert (campaign / 'group0/returncode.txt').read_text().strip() == '0'
state = subprocess.check_output(['sacct', '-j', '728856_0', '-X', '-n', '-P', '--format=JobID,State,ExitCode'], text=True)
assert state.strip() == '728856_0|COMPLETED|0:0'
(ROOT / 'meta/pad-absolute-errors-cpu-v2-terminal.json').write_text(json.dumps({'scheduler': state, 'passed': True}, indent=2)+'\n')
paths = {Path(__file__)}
for receipt in result['receipts']:
    source = Path(receipt['path'])
    h = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    assert h.hexdigest() == receipt['sha256'] and source.stat().st_size == receipt['bytes']
    if source.suffix == '.json':
        paths.add(source)
for pattern in ('source*.json', 'source*.patch', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh', 'group*/returncode.txt', 'group*/*.log', 'group*/**/result.json', 'group*/**/*.csv', 'logs/*.log'):
    paths.update(p for p in campaign.glob(pattern) if p.is_file())
for relative in ('qipc_comparison/pad_absolute_errors.py', 'qipc_comparison/artifacts.py', 'shell_comparison/submit.py'):
    paths.add(campaign / 'analysis-source/research/rigid_stress' / relative)
for name in ('pad-absolute-errors-v1-tasks.json', 'pad-absolute-errors-v1-cancel.json', 'finite-area-prescribed-final-fields-v1-cancel.json', 'check-cpu-analysis-metadata-v1.py', 'check-cpu-analysis-metadata-v1.json', 'pad-absolute-errors-cpu-v2-terminal.json'):
    paths.add(ROOT / 'meta' / name)
# Original unstarted GPU request and the active prescribed-field replacement:
# immutable submissions only; active results and logs are deliberately excluded.
for name in ('pad-absolute-errors-v1', 'finite-area-prescribed-final-fields-v1', 'finite-area-prescribed-final-fields-cpu-v2'):
    for filename in ('submission.json', 'tasks.json', 'source.json', 'source.patch', 'analysis-files.json', 'job.sh'):
        paths.add(ROOT / name / filename)
DEST.mkdir(exist_ok=False)
rows = []
for source in sorted(paths):
    assert source.is_file() and source.stat().st_size < 1_000_000, str(source)
    content = source.read_bytes()
    relative = source.relative_to(ROOT)
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert target.read_bytes() == content
    rows.append({'source': str(source), 'copy': str(relative), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
manifest = {'files': rows, 'verified_file_count': len(rows), 'verified_bytes': sum(row['bytes'] for row in rows), 'whole_goal_complete': False, 'scope': 'Terminal9-case/81-row dimensional normal errors,69 rehashed input identities and4 metadata controls. All column/full-field arrays stay on data. Unstarted GPU requests and active CPU prescribed-field comparison contribute immutable submission/cancellation records only.'}
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({key: manifest[key] for key in ('verified_file_count', 'verified_bytes')}))
