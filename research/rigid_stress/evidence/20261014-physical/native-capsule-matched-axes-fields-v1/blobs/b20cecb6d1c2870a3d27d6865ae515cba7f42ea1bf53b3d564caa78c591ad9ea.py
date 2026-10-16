from pathlib import Path
import hashlib,json,subprocess,sys
wt=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery')
p=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation')
root=wt/'research/rigid_stress/evidence/20261014-physical'
def parent(label):
 f=root/label/'manifest.json';return {'label':label,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
def publish(plan):
 f=p/'meta'/f'{plan["label"]}-publication-plan-v1.json'
 if f.exists():assert json.loads(f.read_text())==plan
 else:f.write_text(json.dumps(plan,indent=2)+'\n')
 log=p/'meta'/f'{plan["label"]}-publication-v1.log';assert not log.exists()
 print('Publishing',plan['label'],flush=True)
 with log.open('w') as stream:
  rc=subprocess.run([sys.executable,'-m','research.rigid_stress.scripts.publish_covered_cad_checkpoint','--runtime-root',str(p.parent),'--destination-root','research/rigid_stress/evidence/20261014-physical','--plan',str(f)],cwd=wt,stdout=stream,stderr=subprocess.STDOUT).returncode
 print(plan['label'],'exit',rc,flush=True);assert rc==0
 proof=p/f'{plan["label"]}-publication/runtime-verification.json';assert json.loads(proof.read_text())['passed']
def base(kind,label,jobs,parents,extras,limitations,shape=None):
 r={'kind':kind,'label':label,'closed_jobs':{name:{'job':str(job)+'_0','returncode':0} for name,job in jobs},'parent_manifests':[parent(x) for x in parents],'extra_files':extras+['meta/publish-matched-fixed-followup-v1.py','meta/fixed-native-followup-publication-orchestration-setup-failure-v1.json'],'limitations':limitations,'pending_work':['Every original physical convergence/model/island/contact/egg-Franka/interior-hotspot/history/feedback/verified-warmed-cost requirement remains.'],'goal_completion':False}
 if shape:r['geometry']=shape
 return r
for shape in ['ellipsoid','capsule']:
 f=p/'meta'/f'fixed-polyhedron-{shape}-fields-v1-publication-plan-v1.json'
 publish(json.loads(f.read_text()))
publish(base('fixed_polyhedron_peak_cells','fixed-polyhedron-peak-cells-v1',[('fixed-polyhedron-raw-peak-cell-diagnosis-v1',720202)],['fixed-polyhedron-ellipsoid-fields-v1','fixed-polyhedron-capsule-fields-v1','fixed-polyhedron-convex-setup-v1'],['meta/fixed-polyhedron-raw-peak-cell-diagnosis-command-v1.json','meta/fixed-polyhedron-local-hotspot-meshes-commands-v1.json'],'All30 original raw maxima/positions and all original full fields remain. Exact auditor/field peak checks precede boundary-depth, incident-traction compatibility, every-cell mean-ratio quality and unsmoothed same-vertex tensor diagnostics. Tensor jumps and local traction residuals are diagnostics, not error budgets, singularity proofs or causal explanations. Pending local mesh generation solves zero new equilibria and is excluded from accepted physical results.'))
for shape,start in [('ellipsoid',720132),('capsule',720140)]:
 closed=[]
 for offset,(prefix,version) in enumerate([(f'extended-native-{shape}-level0','v2'),(f'native-{shape}-fixed-original-p4-q10','v1'),(f'native-{shape}-fixed-original-p16-q10','v1'),(f'native-{shape}-fixed-original-p8-q14','v1')]):
  closed.extend([(f'{prefix}-models-{version}',start+2*offset),(f'{prefix}-independent-audit-{version}',start+2*offset+1)])
 publish(base('native_matched_axis_fields',f'native-{shape}-matched-axes-fields-v1',closed,['native-intrinsic-capacity-v1','native-intrinsic-large-cuda-v1','frozen-traction-inputs-v1','extended-six-shape-qualification-v1',f'extended-finite-{shape}-fields-v1',f'fine-reference-{shape}-v2'],['meta/native-convex-matched-distance-quadrature-commands-v1.json'],'Complete four runs retain all five loads/four models at identical physical inputs. Ports4/8/16 with quadrature10 and ports8 with quadrature14 each have a separate fresh auditor;76 native fields and four unsupported pressure free-moment rows remain. Original force/physical-scatter/corner/quadrature/energy/source/friction gates remain unchanged. These data do not establish mesh/geometry/contact/model continuum error, feedback, application accuracy or warmed cost.',shape))
parents=[f'native-{s}-matched-axes-fields-v1' for s in ['ellipsoid','capsule']]+[f'extended-native-{s}-level{level}-fields-v1' for s in ['ellipsoid','capsule'] for level in [1,2]]+[f'extended-finite-{s}-fields-v1' for s in ['ellipsoid','capsule']]+['extended-six-shape-qualification-v1']
publish(base('native_matched_axes_summary','native-convex-matched-axes-summary-v1',[('native-convex-matched-numerical-axes-comparison-v1',720212)],parents,['meta/native-convex-matched-numerical-axes-comparison-command-v1.json','meta/native-convex-matched-distance-quadrature-commands-v1.json'],'Complete120 same-mesh numerical-control rows and120 model-bias rows retain all original unsupported inputs. Exact volume-integrated raw symmetric-tensor L2 norms pass independent degree-two four-point and constant-tensor controls. All228 accepted native fields have separate original audits. Every numerical control has byte-identical frozen inputs; every prescribed-traction reference is separately full-field audited. Model bias across three spatial levels is conditional against each mesh reference, not cross-mesh field convergence or resolved contact truth. Original unweighted statistics/raw peaks remain. No continuum error bound, causal error subtraction, physical operating range or warmed cost is accepted.'))
print('All six publications independently verified against actual runtime originals.',flush=True)
