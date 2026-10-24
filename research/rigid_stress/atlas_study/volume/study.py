"""3D feature-mesh bank validation; common physical tensor probes.

Reference changes are retained. No coarse finite field is continuum contact GT.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import gc,hashlib,json,logging,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .model import *
logging.getLogger('skfem').setLevel(logging.ERROR)
ROOT=Path(__file__).resolve().parent
ANCHORS=[(-.003,0.),(0.,0.),(.003,0.)]

def cases():
    rows=[]
    for i,x in enumerate([-.0038,-.0018,-.00151,-.00149,-.0004,.0004,.00149,.00151,.0018,.0038]):
        rows.append((f'moving_{i:02d}',[Contact((x,.00025*np.sin(i)),.001,(.015*np.cos(i),.015*np.sin(i),-.05))]))
    for a in (.00055,.0007,.0013):
        rows.append((f'radius_{a:g}',[Contact((.0003,-.0004),a,(.012,-.008,-.05))]))
    rows += [('two_far',[Contact((-.0032,.0003),.001,(.015,.005,-.05)),Contact((.0032,-.0004),.001,(-.008,.005,-.035))]),
             ('two_near',[Contact((-.0003,-.00045),.001,(.01,.004,-.05)),Contact((.0004,.0005),.0007,(-.009,-.003,-.035))]),
             ('three_asymmetric',[Contact((-.003,.0003),.001,(-.008,.003,-.04)),Contact((.0005,-.0004),.0007,(.006,-.006,-.03)),Contact((.0035,.0005),.0013,(.015,.002,-.05))])]
    return rows

def common_probes():
    x,y,z=np.meshgrid(np.linspace(-.009,.009,11),np.linspace(-.009,.009,11),np.linspace(.001,.0099,7),indexing='ij')
    globalp=np.array([x.ravel(),y.ravel(),z.ravel()])
    x,y,z=np.meshgrid(np.linspace(-.005,.005,35),np.linspace(-.003,.003,17),HEIGHT-np.array([.00005,.00025,.0006,.0012,.0024]),indexing='ij')
    near=np.array([x.ravel(),y.ravel(),z.ravel()])
    return np.column_stack((globalp,near)),globalp.shape[1]

def weighted_error(a,b):
    aa=np.asarray(a).reshape(-1,6);bb=np.asarray(b).reshape(-1,6)
    w=np.array([1,1,1,2,2,2])
    return float(np.sqrt(np.sum((aa-bb)**2*w)/max(np.sum(bb**2*w),1e-300)))

def compare(a,b,nglobal):
    av,bv=vm(a),vm(b)
    at,bt=np.linalg.eigvalsh(tensor(a))[:,-1],np.linalg.eigvalsh(tensor(b))[:,-1]
    return dict(full_sampled_tensor_relative=weighted_error(a,b),global_sampled_tensor_relative=weighted_error(a[:6*nglobal],b[:6*nglobal]),
                near_sampled_tensor_relative=weighted_error(a[6*nglobal:],b[6*nglobal:]),
                sampled_vm_peak_Pa=float(av.max()),reference_sampled_vm_peak_Pa=float(bv.max()),sampled_vm_peak_relative=float(abs(av.max()-bv.max())/bv.max()),
                sampled_tensile_peak_Pa=float(at.max()),reference_sampled_tensile_peak_Pa=float(bt.max()),sampled_tensile_peak_relative=float(abs(at.max()-bt.max())/max(abs(bt.max()),1e-30)))

def blended_weights(contact,width=.00045):
    x=contact.center[0]
    for left in range(2):
        boundary=.5*(ANCHORS[left][0]+ANCHORS[left+1][0])
        if abs(x-boundary)<width:
            t=(x-boundary+width)/(2*width);t=t*t*(3-2*t)
            return [(left,1-t),(left+1,t)]
    return [(int(np.argmin([abs(x-a[0]) for a in ANCHORS])),1.)]

def run(out=ROOT/'results',fine_h=.00075):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    queries=cases();points,nglobal=common_probes()
    np.savez_compressed(out/'probe_definition.npz',points=points,nglobal=nglobal)
    report=dict(scope='Independent CPU 3D P2 tetrahedral fixed-box linear elasticity; feature-mesh combination under prescribed finite tractions. NOT hollow-egg/Franka or continuum-contact qualification.',
                geometry=dict(box_m=[L,L,HEIGHT],bottom='all displacement components clamped on z=0',E_Pa=E,nu=NU),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'model.py',Path(__file__))},
                probe_metric='Unweighted common physical tensor-point Frobenius norm, not integrated volume norm. Global regular-grid and dense near-field norms separate. Peaks sample minimum depth50um, not continuum/global boundary maxima.',
                anchors_m=ANCHORS,atlas_hmin_m=.0011,core_radius_m=.0022,reference_hmin_m=fine_h,global_probe_count=nglobal,near_probe_count=points.shape[1]-nglobal,models={},rows=[])
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2,default=lambda a:int(a) if isinstance(a,np.integer) else a.tolist()))
    fields={};infos={}
    definitions=[('reference',ANCHORS,fine_h),('union',ANCHORS,.0011),('coarse',[],.0011)]+[(f'atlas_{i}',[a],.0011) for i,a in enumerate(ANCHORS)]
    for name,anchors,h in definitions:
        print('building',name,flush=True)
        start=time.perf_counter();mesh,history=make_mesh(anchors,hmin=h,core=.0022);model=BoxModel(mesh,name)
        S=model.stress_map(points)
        stat=model.storage();stat.update(mesh_seconds=time.perf_counter()-start-model.assembly_seconds-model.factor_seconds,refinement_history=history,stress_map_bytes=S.data.nbytes+S.indices.nbytes+S.indptr.nbytes)
        report['models'][name]=stat;fields[name]={};infos[name]={};save()
        for label,contacts in queries:
            fs=[];cis=[];ss=[]
            for c in contacts:
                f,ci=model.load(c,order=24);q=model.solve(f);s=np.asarray(S@q).ravel()
                fs.append(f);ss.append(s);cis.append(ci)
            f=sum(fs);q=model.solve(f);combined=np.asarray(S@q).ravel()
            fields[name][label]=dict(individual=ss,total=combined)
            infos[name][label]=dict(contact_checks=cis,residual=model.residual(q,f),superposition_relative=float(np.linalg.norm(combined-sum(ss))/max(np.linalg.norm(combined),1e-300)))
            if name=='reference' and label in ['radius_0.00055','two_near','three_asymmetric','moving_03']:
                f48=sum(model.load(c,order=48)[0] for c in contacts);s48=np.asarray(S@model.solve(f48)).ravel()
                infos[name][label]['quadrature24_vs48']=dict(load_relative=float(np.linalg.norm(f-f48)/np.linalg.norm(f48)),stress_relative=weighted_error(combined,s48))
            print(name,label,'done',flush=True)
        del model,S,mesh;gc.collect()
    for label,contacts in queries:
        hard=np.zeros_like(fields['reference'][label]['total']);blend=hard.copy()
        for i,c in enumerate(contacts):
            nearest=int(np.argmin([abs(c.center[0]-a[0]) for a in ANCHORS]))
            hard+=fields[f'atlas_{nearest}'][label]['individual'][i]
            for j,w in blended_weights(c):blend+=w*fields[f'atlas_{j}'][label]['individual'][i]
        reference=fields['reference'][label]['total'];union=fields['union'][label]['total'];coarse=fields['coarse'][label]['total']
        row=dict(case=label,contacts=[vars(c) for c in contacts],hard_atlas=compare(hard,reference,nglobal),blended_atlas=compare(blend,reference,nglobal),coarse=compare(coarse,reference,nglobal),union=compare(union,reference,nglobal),atlas_vs_joint_refinement=compare(hard,union,nglobal),checks={name:infos[name][label] for name in fields})
        corrected=coarse.copy()
        for i,c in enumerate(contacts):
            nearest=int(np.argmin([abs(c.center[0]-a[0]) for a in ANCHORS]))
            corrected+=fields[f'atlas_{nearest}'][label]['individual'][i]-fields['coarse'][label]['individual'][i]
        row['coarse_plus_full_delta_identity_relative']=float(np.linalg.norm(corrected-hard)/max(np.linalg.norm(hard),1e-300))
        if len(contacts)>1:
            selected=sorted(set(int(np.argmin([abs(c.center[0]-a[0]) for a in ANCHORS])) for c in contacts))
            wrong=sum(fields[f'atlas_{j}'][label]['total'] for j in selected)
            row['wrong_full_load_sum_negative_control']=compare(wrong,reference,nglobal)
        report['rows'].append(row)
        np.savez_compressed(out/(label+'.npz'),points=points,reference=reference,union=union,coarse=coarse,hard_atlas=hard,blended_atlas=blend)
    report['selected_checks']=dict(all_linear_solves=all(info['residual']<1e-9 for bymodel in infos.values() for info in bymodel.values()),per_mesh_superposition=all(info['superposition_relative']<1e-10 for bymodel in infos.values() for info in bymodel.values()),force_moment_preserved=all(ci['force_relative']<1e-12 and ci['moment_normalized']<1e-10 for bymodel in infos.values() for info in bymodel.values() for ci in info['contact_checks']),coarse_delta_identity=all(r['coarse_plus_full_delta_identity_relative']<1e-12 for r in report['rows']))
    report['all_selected_checks_passed']=all(report['selected_checks'].values())
    save();return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results');p.add_argument('--fine-h',type=float,default=.00075)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.out,a.fine_h)
