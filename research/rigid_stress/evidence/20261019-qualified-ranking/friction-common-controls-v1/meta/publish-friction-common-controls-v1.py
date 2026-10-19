"""Publish terminal common-sampling controls and explicitly unexecuted field plans."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DEST = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/friction-common-controls-v1')
CAMPAIGNS = {'stvk-time-contact-common-fields-v1': 2, 'stvk-time-contact-common-audit-v1': 1}
SUPPORT = ('artifacts', 'contact_fem_exact_columns', 'contact_fem_convergence', 'common_metric', 'common_column_peaks', 'p2_stvk', 'p2_elastic', 'pad_normal_qualification', 'audit_stvk_time_contact')
paths = set()
for name, groups in CAMPAIGNS.items():
    folder = ROOT/name
    for group in range(groups):
        assert (folder/f'group{group}/returncode.txt').read_text().strip() == '0'
    for pattern in ('source*.json', 'source*.patch', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh', 'group*/returncode.txt', 'group*/*.log', 'group*/**/result.json', 'logs/*.log'):
        paths.update(p for p in folder.glob(pattern) if p.is_file())
    for name in SUPPORT:
        path = folder/f'analysis-source/research/rigid_stress/qipc_comparison/{name}.py'
        if path.exists(): paths.add(path)
audit = json.loads((ROOT/'stvk-time-contact-common-audit-v1/group0/audit/result.json').read_text())
assert audit['passed'] and len(audit['checks']) == 24 and len(audit['verified_reference_fields']) == 8
paths.update(Path(r['path']) for r in audit['receipts'])
for name in ('stvk-time-contact-common-fields-v1-preflight.json', 'solid-pad-stvk-normal-load-controls-v1-targets.json', 'solid-pad-stvk-normal-load-controls-v1-tasks.json', 'friction-final-field-plan-v1.json', 'stvk-spatial250-fields-v1-tasks.json', 'stvk-wall250-fields-v1-tasks.json', 'stvk-final125-fields-v1-tasks.json', 'stvk-friction-final-method-fields-v1-tasks.json'):
    paths.add(ROOT/'meta'/name)
# Submission only: the normal material-model control has an active writer.
for name in ('submission.json', 'tasks.json', 'source.json', 'source.patch', 'analysis-files.json', 'job.sh', 'workload-0.sh'):
    paths.add(ROOT/'solid-pad-stvk-normal-load-controls-v1'/name)
paths.add(Path(__file__))
DEST.mkdir(exist_ok=False)
rows = []
for source in sorted(paths):
    data = source.read_bytes()
    assert len(data) < 1_000_000
    relative = source.relative_to(ROOT)
    target = DEST/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert target.read_bytes() == data
    rows.append({'source': str(source), 'copy': str(relative), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
manifest = {'files': rows, 'verified_file_count': len(rows), 'verified_bytes': sum(r['bytes'] for r in rows), 'whole_goal_complete': False, 'scope': 'Completed coarse StVK time/contact and common q9/q25 sampling controls, own-history/current-momentum receipts and eight independently rehashed fields. Large fields stay on data. Separate normal-load material targets/submission and future field task plans are explicitly pending; no active physical results are copied or admitted.'}
(DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({k: manifest[k] for k in ('verified_file_count', 'verified_bytes')}))
