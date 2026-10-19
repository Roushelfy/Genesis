"""Publish terminal250um refinement and plot-data control metadata only."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DEST = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/friction-spatial250-v1')
CAMPAIGNS = {'solid-pad-stvk-spatial250-cache-v1': 1, 'stvk-spatial250-fields-v1': 2, 'friction-plot-data500-control-v1': 1}
SUPPORT = ('artifacts', 'plate_solid_pad', 'check_p2_stvk', 'audit_solid_pad', 'contact_fem_exact_columns', 'contact_fem_convergence', 'common_metric', 'common_column_peaks', 'p2_stvk', 'p2_elastic', 'compliant_plate_reference', 'friction_plot_data')
paths = set()
for name, groups in CAMPAIGNS.items():
    folder = ROOT/name
    for group in range(groups):
        assert (folder/f'group{group}/returncode.txt').read_text().strip() == '0'
    for pattern in ('source*.json', 'source*.patch', 'submission.json', 'tasks.json', 'analysis-files.json', 'job.sh', 'workload-*.sh', 'group*/returncode.txt', 'group*/*.log', 'group*/**/result.json', 'group*/**/start.json', 'logs/*.log'):
        paths.update(p for p in folder.glob(pattern) if p.is_file())
    for module in SUPPORT:
        path = folder/f'analysis-source/research/rigid_stress/qipc_comparison/{module}.py'
        if path.exists(): paths.add(path)
audit = json.loads((ROOT/'meta/friction-spatial250-v1-audit.json').read_text())
assert audit['metadata_and_hash_audit_passed'] and audit['plot_data_checks_passed']
assert len(audit['verified_reference_fields']) == 4
assert all(not row['field_target_passed'] for row in audit['refinement'])
paths.update(Path(r['path']) for r in audit['receipts'] if Path(r['path']).suffix == '.json')
for name in ('audit-friction-spatial250-v1.py', 'friction-spatial250-v1-audit.json', 'stvk-spatial250-fields-v1-preflight.json'):
    paths.add(ROOT/'meta'/name)
paths.add(Path(__file__))
# No active writers, NPZ fields, prepared arrays or history CSV/JSONL are copied.
for source in paths:
    assert source.is_file() and source.stat().st_size < 1_000_000, str(source)
DEST.mkdir(exist_ok=False)
rows = []
for source in sorted(paths):
    data = source.read_bytes()
    relative = source.relative_to(ROOT)
    target = DEST/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert target.read_bytes() == data
    rows.append({'source': str(source), 'copy': str(relative), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
manifest = {'files': rows, 'verified_file_count': len(rows), 'verified_bytes': sum(r['bytes'] for r in rows), 'whole_goal_complete': False, 'scope': 'Terminal250um nonlinear mechanics/current-momentum controls,500-to250um q9/q25 field and peak changes, and500um plot-data preparation. All45 input/output metadata receipts independently rehashed. Spatial field gates fail; no final method ranks or rendered friction figures. Complete fields, histories and prepared plotting arrays stay on data. Active fine-reference/material writers are excluded.'}
(DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({k: manifest[k] for k in ('verified_file_count', 'verified_bytes')}))
