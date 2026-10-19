"""Publish terminal reconstruction/material controls, including diagnostic failures."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DEST=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/native-reconstruction-material-v1')
CAMPAIGNS=('finite-area-prescribed-cvxopt-control-v1','finite-area-prescribed-common-controls-v1','native-pad-traction-capture-control-v1','native-pad-traction-capture-control-v2','native-pad-traction-capture-control-v3','native-pad-traction-capture-scr-control-v4','native-pad-reconstruction-dpr-control-v1','native-pad-reconstruction-scr-control-v1','native-pad-traction-capture-full-v1','native-pad-reconstruction-full-audit-v1','solid-pad-stvk-normal-load-controls-v1')
SUPPORT=('artifacts','finite_area_reference','p2_elastic','cvxopt_factor','p2_stvk','contact_fem_exact_columns','common_column_peaks','native_traction_snapshot','replay_native_pad_tractions','audit_native_reconstruction','native_pad_recovery','plate_solid_pad','audit_solid_pad')
paths=set()
for name in CAMPAIGNS:
    folder=ROOT/name
    for pattern in ('source*.json','source*.patch','submission.json','tasks.json','analysis-files.json','job.sh','workload-*.sh','group*/returncode.txt','group*/*.log','group*/**/result.json','group*/**/start.json','logs/*.log'):
        paths.update(p for p in folder.glob(pattern) if p.is_file())
    for module in SUPPORT:
        p=folder/f'analysis-source/research/rigid_stress/qipc_comparison/{module}.py'
        if p.exists():paths.add(p)
audit=json.loads((ROOT/'meta/native-reconstruction-material-v1-audit.json').read_text())
assert audit['metadata_and_hash_audit_passed'] and audit['core_cases']==10
paths.update(Path(r['path']) for r in audit['receipts'] if Path(r['path']).suffix=='.json')
for name in ('audit-native-reconstruction-material-v1.py','native-reconstruction-material-v1-audit.json','native-pad-reconstruction-targets-v1.json','native-pad-traction-capture-v3-scr-cancel.json','finite-area-prescribed-final-targets-v1.json','finite-area-prescribed-cvxopt-admission-v1.json','finite-area-prescribed-common-controls-v1-preflight.json','finite-area-prescribed-final-fields-v1-tasks.json'):
    paths.add(ROOT/'meta'/name)
# Immutable submission only; the finer prescribed reference is still writing.
for name in ('submission.json','tasks.json','source.json','source.patch','analysis-files.json','job.sh','workload-0.sh'):
    paths.add(ROOT/'finite-area-prescribed-final125-v1'/name)
paths.add(Path(__file__))
for p in paths:assert p.is_file() and p.stat().st_size<1_000_000,str(p)
DEST.mkdir(exist_ok=False)
rows=[]
for source in sorted(paths):
    content=source.read_bytes();relative=source.relative_to(ROOT);target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert target.read_bytes()==content
    rows.append({'source':str(source),'copy':str(relative),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
manifest={'files':rows,'verified_file_count':len(rows),'verified_bytes':sum(r['bytes'] for r in rows),'whole_goal_complete':False,'scope':'Terminal actual native quadrature and10-state independent assembly/core checks, normal500um0.3/1N material contrasts, prescribed-C2 q9/q25 controls and verified factor adapter. Original schema/type failures and a known-invalid diagnostic cancellation are retained. All large arrays stay on data. The active125um/three-wall prescribed run contributes only immutable submission receipts; its results and all active friction writers are excluded.'}
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('verified_file_count','verified_bytes')}))
