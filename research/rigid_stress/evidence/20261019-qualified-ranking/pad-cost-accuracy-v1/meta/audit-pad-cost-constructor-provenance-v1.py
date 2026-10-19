"""Read-only provenance audit of captured material/recovery constructor syntax."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
receipts={}
def identity(path):
 data=path.read_bytes()
 result={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 receipts[str(path)]=result
 return result

def signature(campaign,method,batch=False):
 filename='native_pad_recovery.py' if method!='shell' else ('plate_shell_pad_batch.py' if batch else 'plate_shell_coupled.py')
 attribute='RigidStressOptions' if method!='shell' else 'Shell'
 relative='research/rigid_stress/qipc_comparison/'+filename
 manifest=ROOT/campaign/'analysis-files.json'
 identity(manifest)
 expected=next(row for row in json.loads(manifest.read_text()) if row['path']==relative)
 source=ROOT/campaign/'analysis-source'/relative
 actual=identity(source)
 assert all(actual[key]==expected[key] for key in ('bytes','sha256'))
 calls=[node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr==attribute]
 assert len(calls)==1
 # Output changes are separately named in the timing/error association. All
 # constitutive, solver and reconstruction choices must match exactly.
 return {key.arg:ast.dump(key.value,include_attributes=False) for key in calls[0].keywords if key.arg!='output_mode'}

normal=json.loads((ROOT/'pad-cost-accuracy-controls-v1/group0/normal/result.json').read_text())
assert normal['passed'] and len(normal['matched_configurations'])==24
pairs=[]
for row in normal['matched_configurations']:
 method=row['method']
 baseline='native-pad-full-source-stability-v1' if method!='shell' else 'compliant-pad-fine-coarse-history-v1'
 campaign=Path(row['run']).relative_to(ROOT).parts[0]
 a=signature(baseline,method)
 b=signature(campaign,method,row['envs']==4)
 assert a==b
 pairs.append({'case':'normal','method':method,'timed_run':row['run'],'constructor_signature':a,'passed':True})
for method in ('dpr','scr','shell'):
 baseline='native-pad-full-source-stability-v1' if method!='shell' else 'compliant-pad-friction250-history-v1'
 for campaign in (['pad-friction-keyframe-cost-v1','pad-primary-friction-cost-v1'] if method=='shell' else ['pad-friction-keyframe-cost-v1']):
  a=signature(baseline,method)
  assert a==signature(campaign,method)
  pairs.append({'case':'friction-static-provenance-only','method':method,'cost_campaign':campaign,'constructor_signature':a,'passed':True})
assert len(pairs)==28
output=ROOT/'meta/pad-cost-constructor-provenance-v1.json'
assert not output.exists()
output.write_text(json.dumps({'passed':True,'comparisons':pairs,'receipts':list(receipts.values()),'scope':'Source/configuration provenance only. All captured material/recovery constructor arguments except separately declared output mode match the corresponding accuracy worker. The script performs no simulation or scientific numerical test and grants no friction reference/error admission.'},indent=2)+'\n')
print(json.dumps({'passed':True,'constructor_comparisons':28,'source_receipts':len(receipts)}))
