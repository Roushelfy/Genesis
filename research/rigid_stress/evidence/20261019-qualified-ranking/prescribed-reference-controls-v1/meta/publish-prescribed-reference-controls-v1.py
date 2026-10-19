"""Freeze terminal prescribed-load qualification and numerical/allocation controls."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DEST = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/prescribed-reference-controls-v1')
CAMPAIGNS = ('finite-area-prescribed-final125-v1', 'finite-area-prescribed-final-fields-cpu-v2', 'finite-area-prescribed-reference-qualification-v1', 'column-decomposition-controls-v1', 'shell-transfer-memory-control-v1', 'finite-area-prescribed-shell250-v1')
SUPPORT = ('artifacts', 'finite_area_reference', 'p2_elastic', 'cvxopt_factor', 'contact_fem_exact_columns', 'common_column_peaks', 'qualify_prescribed_load', 'pad_normal_qualification', 'column_decomposition', 'check_column_decomposition', 'finite_area_shell', 'shell_load_replay', 'check_shell_transfer_memory')
qualification = json.loads((ROOT/'finite-area-prescribed-reference-qualification-v1/group0/qualification/result.json').read_text())
assert qualification['prescribed_reference_change_targets_passed'] and all(qualification['checks'].values())
analytic = json.loads((ROOT/'column-decomposition-controls-v1/group0/analytic/result.json').read_text())
memory = json.loads((ROOT/'shell-transfer-memory-control-v1/group0/control/result.json').read_text())
assert analytic['passed'] and len(analytic['cases']) == 7 and memory['passed']
scheduler = subprocess.check_output(['sacct', '-j', '728727,728860,728896,728884,728885,728882', '-X', '-n', '-P', '--format=JobID,State,Elapsed,ExitCode'], text=True)
for line in scheduler.splitlines():
    assert ('728882' in line and '|FAILED|' in line) or '|COMPLETED|' in line, line
assert len(scheduler.splitlines()) == 9, scheduler
paths = {Path(__file__)}
for receipt in qualification['receipts']:
    source = Path(receipt['path'])
    h = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    assert h.hexdigest() == receipt['sha256'] and source.stat().st_size == receipt['bytes']
    if source.suffix == '.json': paths.add(source)
audit = {'scheduler': scheduler, 'prescribed_reference_checks_passed': len(qualification['checks']), 'input_receipts_rehashed': len(qualification['receipts']), 'tensor_controls_passed': 7, 'memory_control_passed': True, 'whole_goal_complete': False}
(ROOT/'meta/prescribed-reference-controls-v1-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
for name in CAMPAIGNS:
    folder = ROOT/name
    for pattern in ('source*.json', 'source*.patch', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh', 'group*/returncode.txt', 'group*/*.log', 'group*/**/result.json', 'group*/**/start.json', 'logs/*.log'):
        paths.update(p for p in folder.glob(pattern) if p.is_file())
    for module in SUPPORT:
        p = folder/f'analysis-source/research/rigid_stress/qipc_comparison/{module}.py'
        if p.exists(): paths.add(p)
for name in ('finite-area-prescribed-final-targets-v1.json', 'finite-area-prescribed-final-fields-v1-preflight.json', 'finite-area-prescribed-shell250-v1-preflight.json', 'finite-area-prescribed-shell250-v1-oom.json', 'finite-area-prescribed-reference-qualification-v1-preflight.json', 'finite-area-prescribed-native-decomposition-v1-preflight.json', 'finite-area-prescribed-decomposition-targets-v1.json', 'prescribed-reference-controls-v1-audit.json'):
    paths.add(ROOT/'meta'/name)
# Only immutable submitted inputs from live physics/field workers.
for name in ('finite-area-prescribed-shell250-v2', 'finite-area-prescribed-native-decomposition-v1'):
    for filename in ('submission.json', 'tasks.json', 'source.json', 'source.patch', 'analysis-files.json', 'job.sh'):
        paths.add(ROOT/name/filename)
    for module in SUPPORT:
        p = ROOT/name/f'analysis-source/research/rigid_stress/qipc_comparison/{module}.py'
        if p.exists(): paths.add(p)
DEST.mkdir(exist_ok=False)
rows = []
for source in sorted(paths):
    assert source.is_file() and source.stat().st_size < 1_000_000, str(source)
    content = source.read_bytes(); relative = source.relative_to(ROOT); target = DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    assert target.read_bytes() == content
    rows.append({'source':str(source),'copy':str(relative),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
manifest = {'files':rows,'verified_file_count':len(rows),'verified_bytes':sum(r['bytes'] for r in rows),'whole_goal_complete':False,'scope':'Terminal independent prescribed-C2 reference qualification, signed tensor manufactured controls, exact before/after load-map allocation control and original250um replay OOM. Live replay/decomposition contribute immutable submissions/source only. All columns/full fields and active friction output stay on data.'}
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({key:manifest[key] for key in ('verified_file_count','verified_bytes')}))
