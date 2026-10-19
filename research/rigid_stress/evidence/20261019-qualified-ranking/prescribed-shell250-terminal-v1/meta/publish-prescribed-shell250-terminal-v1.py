"""Freeze admitted static replay after the earlier reference/control cutoff."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
DEST=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/prescribed-shell250-terminal-v1')
run=ROOT/'finite-area-prescribed-shell250-v2/group0/shell'
d=json.loads((run/'result.json').read_text());assert d['static_equations_admitted']
terminal=ROOT/'meta/finite-area-prescribed-shell250-v2-terminal.json';audit=json.loads(terminal.read_text())
rest=run/'rest.npz';content=rest.read_bytes();audit['rest']={'path':str(rest),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
terminal.write_text(json.dumps(audit,indent=2)+'\n')
paths={Path(__file__),terminal,ROOT/'meta/finite-area-prescribed-shell-decomposition-v1-tasks.json'}
folder=ROOT/'finite-area-prescribed-shell250-v2'
for pattern in ('source*.json','source*.patch','submission.json','tasks.json','analysis-files.json','job.sh','workload-*.sh','group*/returncode.txt','group*/*.log','group*/**/result.json','group*/**/start.json','group*/**/steps.jsonl','logs/*.log'):
    paths.update(p for p in folder.glob(pattern) if p.is_file())
for module in ('finite_area_shell','shell_load_replay','artifacts'):
    paths.add(folder/f'analysis-source/research/rigid_stress/qipc_comparison/{module}.py')
for filename in ('submission.json','tasks.json','source.json','source.patch','analysis-files.json','job.sh'):
    paths.add(ROOT/'finite-area-prescribed-shell-decomposition-v1'/filename)
DEST.mkdir(exist_ok=False);rows=[]
for source in sorted(paths):
    assert source.is_file() and source.stat().st_size<1_000_000,str(source)
    content=source.read_bytes();relative=source.relative_to(ROOT);target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert target.read_bytes()==content
    rows.append({'source':str(source),'copy':str(relative),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
manifest={'files':rows,'verified_file_count':len(rows),'verified_bytes':sum(r['bytes'] for r in rows),'whole_goal_complete':False,'scope':'Terminal250um Shell static prescribed-C2 replay after the allocation fix. Full frame/rest arrays stay on data with hashes. The new three-field comparison contributes immutable submissions only; no end-to-end physical-contact or final-method rank claimed.'}
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('verified_file_count','verified_bytes')}))
