"""CPU feature-mesh atlas for a clamped flat MITC4 shell.

Exact common plate geometry, finite Hertz pressure integrated independently
with analytical transverse moments. No physical snapping of contact centers.
All output comparisons evaluate the same material coordinates.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,hashlib,platform,argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]/'genesis-atlas'
if not (REPO/'research/rigid_stress/shell_cpu').exists():
    REPO=next(p for p in HERE.parents if (p/'research/rigid_stress/shell_cpu').exists())
sys.path.insert(0,str(REPO/'research/rigid_stress'))
from shell_cpu.core import ShellModel
from shell_cpu.review_checks import independent_rectangle_weights

L=.04;H=.0005;E=1e10;NU=.3
ANCHORS=np.array([-.0065,0.,.0065])

def knots(lo,hi,step):
    return np.linspace(lo,hi,int(round((hi-lo)/step))+1)

def grid(axis_x,axis_y):
    x=np.asarray(axis_x);y=np.asarray(axis_y)
    v=np.array([[a,b,0.] for b in y for a in x]);nx=len(x)
    q=np.array([[j*nx+i,j*nx+i+1,(j+1)*nx+i+1,(j+1)*nx+i]
                for j in range(len(y)-1) for i in range(nx-1)])
    return v,q

def refined_knots(center=0.,halfwidth=.008,step=.001,coarse=.005):
    vals=np.r_[knots(-L/2,L/2,coarse),knots(center-halfwidth,center+halfwidth,step)]
    return np.unique(np.round(vals,12))

def probe_operator(model,x,y,points):
    """Exact Q4 in-plane stress at common physical points on an aligned plate."""
    p=np.asarray(points);ix=np.clip(np.searchsorted(x,p[:,0],side='right')-1,0,len(x)-2)
    iy=np.clip(np.searchsorted(y,p[:,1],side='right')-1,0,len(y)-2)
    dx=x[ix+1]-x[ix];dy=y[iy+1]-y[iy]
    xi=2*(p[:,0]-x[ix])/dx-1;eta=2*(p[:,1]-y[iy])/dy-1
    signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    dn=np.stack((signs[None,:,0]*(1+eta[:,None]*signs[None,:,1])/(2*dx[:,None]),
                 signs[None,:,1]*(1+xi[:,None]*signs[None,:,0])/(2*dy[:,None])),axis=-1)
    B=np.zeros((len(p),3,24));K=np.zeros_like(B)
    for i in range(4):
        B[:,0,6*i]=dn[:,i,0];B[:,1,6*i+1]=dn[:,i,1]
        B[:,2,6*i]=dn[:,i,1];B[:,2,6*i+1]=dn[:,i,0]
        K[:,0,6*i+4]=dn[:,i,0];K[:,1,6*i+3]=-dn[:,i,1]
        K[:,2,6*i+4]=dn[:,i,1];K[:,2,6*i+3]=-dn[:,i,0]
    maps=np.stack([np.einsum('ij,pjk->pik',model.C,B+side*H/2*K) for side in (1,-1)])
    elem=iy*(len(x)-1)+ix;dofs=model.element_dofs[elem]
    rr=np.repeat(np.arange(2*len(p)*3),24)
    cc=np.broadcast_to(dofs[None,:,None,:],maps.shape).ravel()
    return sparse.coo_matrix((maps.ravel(),(rr,cc)),shape=(2*len(p)*3,model.ndof)).tocsr()

class Plate:
    def __init__(self,name,x,y,probes):
        self.name=name;self.x=np.asarray(x);self.y=np.asarray(y)
        v,q=grid(x,y);t=time.perf_counter()
        self.m=ShellModel(v,q,E=E,nu=NU,h=H,gauge='translation')
        self.assembly_s=time.perf_counter()-t
        edge=np.any(np.isclose(np.abs(v[:,:2]),L/2,atol=1e-12),axis=1)
        self.fixed=(6*np.flatnonzero(edge)[:,None]+np.arange(6)).ravel()
        self.free=np.setdiff1d(np.arange(self.m.ndof),self.fixed)
        K=self.m.K[self.free][:,self.free].tocsc();self.scale=1/np.sqrt(K.diagonal())
        t=time.perf_counter();self.lu=splu(sparse.diags(self.scale)@K@sparse.diags(self.scale),permc_spec='MMD_AT_PLUS_A')
        self.factor_s=time.perf_counter()-t
        self.S=probe_operator(self.m,x,y,probes)
        self.memory_bytes=int(self.lu.L.data.nbytes+self.lu.U.data.nbytes+self.lu.L.indices.nbytes+self.lu.U.indices.nbytes+self.lu.L.indptr.nbytes+self.lu.U.indptr.nbytes)
    def load(self,c):
        x,y,fn,sx,sy,a=c
        w=independent_rectangle_weights(self.m.vertices,self.m.quads,np.array([x,y,H/2]),a)
        F=fn*np.array([sx,sy,-1.]);f=np.zeros((len(w),6))
        f[:,:3]=w[:,None]*F;f[:,3:]=w[:,None]*np.cross(np.array([0,0,H/2]),F)
        force=f[:,:3].sum(axis=0)
        moment=(np.cross(self.m.vertices,f[:,:3])+f[:,3:]).sum(axis=0)
        source=np.cross(np.array([x,y,H/2]),F)
        info=dict(mass_error=float(abs(w.sum()-1)),force_relative=float(np.linalg.norm(force-F)/np.linalg.norm(F)),
                  moment_normalized=float(np.linalg.norm(moment-source)/(a*np.linalg.norm(F))),min_weight=float(w.min()))
        return f.ravel(),info
    def solve(self,f):
        q=np.zeros_like(f);s=self.scale if f.ndim==1 else self.scale[:,None]
        q[self.free]=s*self.lu.solve(np.asfortranarray(s*f[self.free]))
        return q
    def field(self,f):
        q=self.solve(f);return np.asarray(self.S@q).reshape(2,-1,3),q
    def metadata(self):
        return dict(name=self.name,quads=len(self.m.quads),dofs=self.m.ndof,assembly_s=self.assembly_s,factor_s=self.factor_s,
                    lu_bytes=self.memory_bytes,lu_nnz=int(self.lu.L.nnz+self.lu.U.nnz),nx=len(self.x)-1,ny=len(self.y)-1)

def vm(s):
    a,b,t=np.moveaxis(s,-1,0);return np.sqrt(np.maximum(a*a-a*b+b*b+3*t*t,0))

def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def measure(a,b,mask=None):
    if mask is not None:a=a[:,mask];b=b[:,mask]
    vma,vmb=vm(a),vm(b)
    weights=np.array([1.,1.,2.])
    return dict(tensor_Frobenius_relative=float(np.sqrt(np.sum((a-b)**2*weights)/np.sum(b*b*weights))),
                peak_vm_Pa=float(vma.max()),reference_peak_vm_Pa=float(vmb.max()),
                peak_vm_relative=float(abs(vma.max()-vmb.max())/max(vmb.max(),1e-30)))

def cases():
    out=[]
    for x in np.linspace(-.009,.009,13):
        out.append((f'move_{x*1000:.2f}',[(float(x),.00037,.05,.25,-.1,.0024)]))
    for a in [.0012,.0018,.0036]:
        out.append((f'radius_{a*1000:.1f}',[(.0023,-.00041,.05,-.3,.2,a)]))
    out.extend([
      ('two_far',[(-.008,.0003,.04,.2,.1,.0024),(.008,-.0007,.07,-.3,.1,.0018)]),
      ('two_close',[(.001,.0004,.05,.2,-.1,.0024),(.0023,-.0006,.03,-.25,.15,.0018)]),
      ('three_asym',[(-.009,.0002,.02,.3,.1,.0012),(.001,-.0007,.07,-.2,.2,.0036),(.008,.0006,.04,-.2,-.1,.0024)]),
      ('release',[]),
    ])
    return out

def select(c):return int(np.argmin(abs(ANCHORS-c[0])))

def blend_weights(c):
    x=c[0]
    if x<=ANCHORS[0]:return [(0,1.)]
    if x>=ANCHORS[-1]:return [(2,1.)]
    i=np.searchsorted(ANCHORS,x)-1;t=(x-ANCHORS[i])/(ANCHORS[i+1]-ANCHORS[i]);t=t*t*(3-2*t)
    return [(int(i),float(1-t)),(int(i+1),float(t))]

def all_fields(models,cs):
    result={};loads={};gate={};time_ms={};solutions={}
    for m in models:
        start=time.perf_counter();pairs=[m.load(c) for c in cs];transfer_ms=(time.perf_counter()-start)*1000
        F=np.column_stack([p[0] for p in pairs]) if pairs else np.zeros((m.m.ndof,1))
        start=time.perf_counter();q=m.solve(F);fields=np.asarray(m.S@q).reshape(2,-1,3,F.shape[1]);solve_ms=(time.perf_counter()-start)*1000
        result[m.name]=fields if cs else fields[...,:0]
        solutions[m.name]=q;loads[m.name]=F
        gate[m.name]=[p[1] for p in pairs]
        r=m.m.K@q-F;r[m.fixed]=0
        gate[m.name].append(dict(linear_residual=float(np.linalg.norm(r)/max(np.linalg.norm(F),1e-30))))
        time_ms[m.name]=dict(transfer_ms=transfer_ms,solve_stress_ms=solve_ms)
    return result,loads,gate,time_ms,solutions

def main(out=HERE/'results',quick=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # A common material grid, including the far field and separate near-contact density.
    p=np.linspace(-.0193,.0193,59);probes=np.array([[x,y] for y in p for x in p])
    models=[]
    base=knots(-L/2,L/2,.005)
    models.append(Plate('coarse',base,base,probes))
    for i,a in enumerate(ANCHORS):
        models.append(Plate(f'atlas{i}',refined_knots(a),refined_knots(0,.006),probes))
    refinements=[('joint',.00075)]
    if not quick:refinements.append(('reference_check',.0005))
    refinements.append(('reference',.0005 if quick else .000375))
    for name,h in refinements:
        x=np.unique(np.round(np.r_[base,knots(-.017,.017,h)],12))
        y=np.unique(np.round(np.r_[base,knots(-.007,.007,h)],12))
        models.append(Plate(name,x,y,probes));print('built',models[-1].metadata(),flush=True)
    rows=[];fields_saved={};timings=[];gates=[]
    selected=cases() if not quick else cases()[::4]+cases()[-4:]
    for name,cs in selected:
        f,loads,gate,times,qs=all_fields(models,cs)
        total={k:v.sum(axis=-1) for k,v in f.items()}
        atlas=np.zeros_like(total['coarse']);blend=atlas.copy()
        for j,c in enumerate(cs):
            atlas+=f[f'atlas{select(c)}'][...,j]
            for i,w in blend_weights(c):blend+=w*f[f'atlas{i}'][...,j]
        # Algebraic coarse+delta equivalence, exact on the common probes.
        delta=total['coarse'].copy()
        for j,c in enumerate(cs):delta+=f[f'atlas{select(c)}'][...,j]-f['coarse'][...,j]
        variants=[('coarse',total['coarse']),('atlas',atlas),('blend',blend),('joint',total['joint'])]
        if 'reference_check' in total:variants.append(('reference_check',total['reference_check']))
        metrics={k:measure(v,total['reference']) for k,v in variants} if cs else {}
        near=np.zeros(len(probes),bool)
        for c in cs:near|=np.linalg.norm(probes-np.array(c[:2]),axis=1)<2*c[-1]
        if cs and (~near).any():
            metrics['far_atlas']=measure(atlas,total['reference'],~near)
            metrics['near_atlas']=measure(atlas,total['reference'],near)
        row=dict(case=name,contacts=cs,metrics=metrics,coarse_delta_identity=rel(delta,atlas),
                 selected_atlases=[select(c) for c in cs],
                 full_load_wrong_sum_error=rel(sum(total[f'atlas{i}'] for i in range(3)),total['reference']) if cs else 0.)
        rows.append(row);gates.append(dict(case=name,models=gate));timings.append(dict(case=name,models=times))
        if name in ['move_0.00','two_far','three_asym']:
            fields_saved[name]=dict(coarse=total['coarse'],atlas=atlas,blend=blend,reference=total['reference'])
        print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(dict(scope='Exact common clamped plate geometry; shell in-plane FEM only, not 3D contact reference. Tensor-grid local refinement includes strips. No online mesh/factor changes.',models=[m.metadata() for m in models],rows=rows),indent=2))
    # Hard switching versus smooth operator blend at identical load coordinates.
    switches=[]
    eps=.00000001
    for edge in (ANCHORS[:-1]+ANCHORS[1:])/2:
        pair=[];center_fields=None
        for x in [edge-eps,edge,edge+eps]:
            c=(float(x),.00037,.05,.25,-.1,.0024)
            fa=[]
            for i in range(3):
                load,_=models[i+1].load(c);s,_=models[i+1].field(load);fa.append(s)
            hard=fa[select(c)];smooth=sum(w*fa[i] for i,w in blend_weights(c))
            load,_=models[-1].load(c);reference,_=models[-1].field(load)
            pair.append((hard,smooth,reference))
            if x==edge:center_fields=fa
        left=int(np.searchsorted(ANCHORS,edge)-1)
        switches.append(dict(x_m=float(edge),delta_x_m=2*eps,hard_relative_jump=rel(pair[-1][0],pair[0][0]),smooth_relative_jump=rel(pair[-1][1],pair[0][1]),
                             reference_physical_motion_delta=rel(pair[-1][2],pair[0][2]),
                             same_point_mesh_discrepancy=rel(center_fields[left+1],center_fields[left])))
    # Focused cached online solve/stress benchmark for a two-contact query.
    cs=cases()[-4][1];bank={m.name:np.column_stack([m.load(c)[0] for c in cs]) for m in models}
    cost={}
    for m in models:
        F=bank[m.name];times=[]
        for _ in range(5):
            t=time.perf_counter();s=m.S@m.solve(F);times.append((time.perf_counter()-t)*1000)
        cost[m.name]=dict(median_ms=float(np.median(times)),range_ms=[min(times),max(times)])
    timed_atlas=[]
    for _ in range(5):
        start=time.perf_counter();s=np.zeros((2,len(probes),3))
        for i in sorted(set(select(c) for c in cs)):
            inds=[j for j,c in enumerate(cs) if select(c)==i]
            m=models[i+1];F=bank[m.name][:,inds].sum(axis=1)
            s+=np.asarray(m.S@m.solve(F)).reshape(2,-1,3)
        timed_atlas.append((time.perf_counter()-start)*1000)
    cost['selected_atlas_sum']=dict(median_ms=float(np.median(timed_atlas)),range_ms=[min(timed_atlas),max(timed_atlas)],scope='Actual two separated contacts; one RHS per selected feature mesh plus common-probe tensor sum. Reuses precomputed loads and factors.')
    meta=dict(scope='SHELL in-plane plate atlas; exact common geometry, clamped entire perimeter. All contacts on actual top surface. Separate reference-grid variation is recorded, never called physical ground truth.',
              material=dict(E=E,nu=NU,h=H,L=L),metric='Unweighted common material-probe Frobenius norm; not a continuum area-integrated L2. VM is maximum over these declared probes and two surfaces.',probes=probes.tolist(),models=[m.metadata() for m in models],rows=rows,switches=switches,gates=gates,timings_diagnostic=timings,cached_two_rhs_solve_stress_cost=cost,
              cpu=platform.processor(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(meta,indent=2))
    np.savez_compressed(out/'fields.npz',probes=probes,**{case+'_'+method:v for case,d in fields_saved.items() for method,v in d.items()})
    # Render convergence/accuracy and a common-probe stress map, no smoothing.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    moving=[r for r in rows if r['case'].startswith('move')]
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in ('coarse','atlas','blend','joint'):
        axs[0].plot([r['contacts'][0][0]*1000 for r in moving],[100*r['metrics'][method]['tensor_Frobenius_relative'] for r in moving],'o-',label=method)
        axs[1].plot([r['contacts'][0][0]*1000 for r in moving],[100*r['metrics'][method]['peak_vm_relative'] for r in moving],'o-',label=method)
    axs[0].set(xlabel='contact x (mm)',ylabel='common sampled tensor difference (%)',title='Moving true contact on fixed feature meshes')
    axs[1].set(xlabel='contact x (mm)',ylabel='sampled VM peak difference (%)',title='Reference is a finer shell, not 3D truth')
    axs[0].legend();[a.grid(alpha=.2) for a in axs];fig.savefig(out/'accuracy.png',dpi=160);plt.close(fig)
    if fields_saved:
        case=next(iter(fields_saved));d=fields_saved[case]
        fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained');vmax=max(vm(v).max() for v in d.values())/1000
        for ax,method in zip(axs,['coarse','atlas','reference']):
            im=ax.imshow(vm(d[method]).max(axis=0).reshape(len(p),len(p))/1000,origin='lower',extent=[p[0]*1000,p[-1]*1000,p[0]*1000,p[-1]*1000],vmin=0,vmax=vmax,cmap='inferno');ax.set(title=method,xlabel='x (mm)',ylabel='y (mm)')
        fig.colorbar(im,ax=axs,label='sampled shell VM (kPa)',shrink=.8);fig.suptitle(case);fig.savefig(out/'fields.png',dpi=160);plt.close(fig)
    print('FINISHED',out,flush=True)
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'results');p.add_argument('--quick',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):main(a.output,a.quick)
