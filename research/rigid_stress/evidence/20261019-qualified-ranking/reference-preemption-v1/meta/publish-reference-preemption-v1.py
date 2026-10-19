"""Preserve interrupted reference receipts and verified submission signal controls."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DESTINATION = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/reference-preemption-v1')
old = ROOT/'solid-pad-stvk-final125-reference-v1'
new = ROOT/'solid-pad-stvk-final125-reference-v2'
state = [subprocess.check_output(['scontrol', 'show', 'job', f'728589_{i}'], text=True) for i in range(2)]
assert all('Requeue=0' in s for s in state)
assert json.loads((ROOT/'submission-signal-control-v2/result.json').read_text())['passed']
retry = ROOT/'meta/reference125-retry-v2.json'
retry.write_text(json.dumps({'scheduler_state_at_capture': state, 'original_job': 728537, 'retry_job': 728589, 'completed': False, 'scope': 'Submission configuration only; active physical results excluded.'}, indent=2)+'\n')
race = ROOT/'meta/submission-signal-control-v1-harness-race.json'
race.write_text(json.dumps({'initial_test_harness': str(ROOT/'submission-signal-control-v1'), 'five_self_exit_cases_passed': True, 'external_term_test': 'The initial test echoed readiness in the parent before forking the foreground sleep. Early group TERM could precede that fork, so communicate timed out at5s; the recorded shell eventually returned143. This was a test synchronization race.', 'correction': 'The v2 test waits for readiness emitted by the foreground Python child before sending group TERM. All seven v2 cases pass.'}, indent=2)+'\n')
paths = {ROOT/'meta/reference125-preemption-v1.json', retry, race, Path(__file__)}
for folder in (old, new):
    for pattern in ('submission.json', 'tasks.json', 'source.json', 'source.patch', 'analysis-files.json', 'job.sh', 'workload-*.sh'):
        paths.update(folder.glob(pattern))
for pattern in ('group*/returncode.txt', 'group*/reference.log', 'group*/analytic.log', 'group*/analytic/result.json', 'group*/reference/start.json', 'group*/reference/steps.jsonl', 'logs/*.log'):
    paths.update(old.glob(pattern))
for version in (1, 2):
    paths.update(p for p in (ROOT/f'submission-signal-control-v{version}').rglob('*') if p.is_file())
for name in ('submit', 'check_submission_signals'):
    paths.add(new/f'analysis-source/research/rigid_stress/shell_comparison/{name}.py')
DESTINATION.mkdir(exist_ok=False)
rows = []
for source in sorted(paths):
    relative = source.relative_to(ROOT)
    data = source.read_bytes()
    assert len(data) < 1_000_000
    target = DESTINATION/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert target.read_bytes() == data
    rows.append({'source': str(source), 'copy': str(relative), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
manifest = {'files': rows, 'verified_file_count': len(rows), 'verified_bytes': sum(r['bytes'] for r in rows), 'whole_goal_complete': False, 'scope': 'Original preempted/requeued incomplete reference histories, signal-control successes and initial harness race, and immutable retry submission/source receipts. Active retry results excluded. Zero old shell returncodes are not physics admission.'}
(DESTINATION/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({k: manifest[k] for k in ('verified_file_count', 'verified_bytes')}))
