"""Export compact publication evidence; full runtime artifacts stay on data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

cli=argparse.ArgumentParser()
cli.add_argument('--data',type=Path,required=True)
cli.add_argument('--output',type=Path,required=True)
a=cli.parse_args()
required=['throughput-v1','regression-v5','scene-v7','scene-audit-v3','contacts-v8','two-sides-v1','large-solve-v1','manual-v1','trace-v1','trace-v2','profile-device-v2','physics-v2','dependencies-v1']
for c in required:
 for i,g in enumerate(json.loads((a.data/c/'tasks.json').read_text())):
  for t in g:
   p=a.data/c/f'group{i}'/(t['name']+'.returncode.txt')
   assert p.exists() and p.read_text().strip()=='0',p

a.output.mkdir(parents=True,exist_ok=False)
external=[]
records=[]
def identify(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def copy(p,rel):
 dest=a.output/rel
 dest.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(p,dest)
 records.append({'publication':str(rel),'source':identify(p)})

controls=['operators-v5','operators-spinning-v1','contacts-v8','two-sides-v1','precision-v1','precision-v2','factor-controls-v1','scene-v7','scene-audit-v3','regression-v5','physics-v2']
for c in controls:
 for p in sorted((a.data/c).glob('group*/*/*.json')):
  if p.name=='progress.json':continue
  assert p.stat().st_size<500000,p
  copy(p,Path('controls')/p.relative_to(a.data))
 for p in sorted((a.data/c).glob('group*/*/junit.xml')):
  copy(p,Path('controls')/p.relative_to(a.data))
for p in sorted(a.data.iterdir()):
 if not (p/'tasks.json').exists():continue
 for name in ['tasks.json','submission.json']:
  if (p/name).exists():copy(p/name,Path('receipts')/p.name/name)
 for source in sorted(p.glob('*source*.json')) + [p/'analysis-files.json']:
  if source.exists():external.append(identify(source))
 for receipt in sorted(p.glob('group*/*.returncode.txt')):
  copy(receipt,Path('receipts')/receipt.relative_to(a.data))
for c in ['throughput-v1','large-solve-v1','manual-v1','trace-v1','trace-v2','profile-device-v2']:
 for i,g in enumerate(json.loads((a.data/c/'tasks.json').read_text())):
  for task in g:external.append(identify(a.data/c/f'group{i}'/task['name']))
for pattern in ['scene-v7/group0/cycle/field-*.npz','physics-v2/group*/*/*-fields.npz','trace-v*/group*/*.trace.json']:
 for p in sorted(a.data.glob(pattern)):external.append(identify(p))
for name in ['measurements-final.json','requirements-final.json','campaign-status-final.json','delivery-validation.json']:
 copy(a.data/'meta'/name,Path('metadata')/name)
copy(a.data/'meta/build_publication.py',Path('receipts/build_publication.py'))
copy(a.data/'dependencies-v1/group0/pyfe3d/install.json',Path('metadata/reference-install.json'))
conditions=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-finite-edge-franka-boundary-v2/group0/franka/conditions.npz')
copy(conditions,Path('conditions.npz'))
(a.output/'manifest.json').write_text(json.dumps({'copied':records,'external':external},indent=2)+'\n')
(a.output/'README.md').write_text('''# Native shell DPR publication evidence\n\nCompact records for the 2026-10-24 native thin-shell delivery. See the report\n`../../docs/SHELL_DPR_NATIVE_20261024.md` and running guide. The primary timing\nsource is 50a8e3bf3. `metadata/measurements-final.json` preserves repeat timings,\ninvalid counts, memory, configuration identities and intrusive trace summaries.\nThe trace step denominator counts CPU annotation scopes only; GPU annotation\nspans are not additional steps.\n\nControls include numerical and independent physical comparisons, complete\nFranka history, full-field audits, regression receipts and retained FP32\nrejection. Earlier failed/unsubmitted attempts remain in the campaign ledger.\nThe two missing-pyfe3d attempts were replayed with a pinned isolated data\noverlay. Raw physical availability records inherit an old OpenSees note from\nthe CPU study: this is historical, not a current installation probe or an\nexecuted OpenSees reference. Current pyfe3d/Code_Aster probes are distinct.\n\n`manifest.json` hashes every copied source and references full source inventories,\nlarge benchmark JSON, field arrays and traces on data. Those external files\nare retained in place. Small condition-bank and result metadata are publication\ndeliverables; runtime arrays, dependencies and logs remain under data.\n''')
print('files',len(records),'external receipts',len(external),'bytes',sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file()))
