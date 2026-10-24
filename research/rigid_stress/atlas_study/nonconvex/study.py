"""2D plane-stress P2 feature-mesh atlas on an exact sharp dogbone domain.

Prescribed finite edge tractions, not a contact solve. Full sampled tensor fields
are combined at common physical probes. Corner exclusions are explicit because
sharp reentrant corners have no finite mesh-independent global stress maximum.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import hashlib,json,time,platform
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from skfem import MeshTri,Basis,ElementTriP2,ElementVector,asm,BilinearForm
from skfem.helpers import sym_grad,ddot,trace
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
E=1.;NU=.3;MU=E/(2*(1+NU));LAM=E*NU/(1-NU*NU)
ANCHORS=np.array([-.65,0.,.65])
C=E/(1-NU*NU)*np.array([[1,NU,0],[NU,1,0],[0,0,(1-NU)/2]])

@BilinearForm
def elastic(u,v,w):
    eu,ev=sym_grad(u),sym_grad(v)
    return 2*MU*ddot(eu,ev)+LAM*trace(eu)*trace(ev)

def grid(step,anchors=(),neck=False,level=1):
    x=np.array([0,.5,1,1.5,2.,2.5,3,3.5,4.])
    y=np.array([-1,-.8,-.5,-.2,0,.2,.5,.8,1.])
    if anchors:
        x=np.r_[x,[3.2,3.4,3.6,3.8,3.9]]
        for a in anchors:
            y=np.r_[y,np.arange(max(-1,a-.36),min(1,a+.36)+1e-9,step)]
    if neck:
        x=np.r_[x,np.arange(1.,3.+1e-9,.125/level)]
        y=np.r_[y,np.arange(-.35,.35+1e-9,.05/level),[-.2,.2]]
    # Quantization removes machine-close duplicates without changing geometry.
    x=np.unique(np.round(x,12));y=np.unique(np.round(y,12))
    # Prevent accidental near-coincident knots when combining independent grids.
    # Geometry-defining horizontal boundaries are never moved or removed.
    mandatory=np.array([-1.,-.2,.2,1.]);min_gap=.5*step
    candidates=[v for v in y if np.min(abs(mandatory-v))>=min_gap-1e-12]
    kept=list(mandatory)
    for v in candidates:
        if np.min(abs(np.asarray(kept)-v))>=min_gap-1e-12:kept.append(v)
    y=np.array(sorted(kept))
    m=MeshTri.init_tensor(x,y)
    center=m.p[:,m.t].mean(axis=1)
    remove=(center[0]>1.5)&(center[0]<2.5)&(np.abs(center[1])>.2+1e-12)
    return m.remove_elements(np.flatnonzero(remove))


def stress_map(basis,points):
    # Bound the inverse-map search fallback for high-aspect graded triangles.
    # scikit-fem may otherwise form all(point,element) pairs in one allocation.
    if points.shape[1]>128:
        chunks=[stress_map(basis,points[:,i:i+128]) for i in range(0,points.shape[1],128)]
        return sparse.vstack([sparse.vstack([z[k*(z.shape[0]//3):(k+1)*(z.shape[0]//3)] for z in chunks]) for k in range(3)]).tocsr()
    cells=basis.mesh.element_finder(mapping=basis.mapping)(*points)
    X=basis.mapping.invF(points[:,:,None],tind=cells)
    vals=[];rr=[];cc=[];n=points.shape[1]
    for k in range(basis.Nbfun):
        grad=basis.elem.gbasis(basis.mapping,X,k,tind=cells)[0].grad[...,0]
        strain=np.array([grad[0,0],grad[1,1],grad[0,1]+grad[1,0]])
        sig=C@strain
        for c in range(3):
            vals.extend(sig[c]);rr.extend(c*n+np.arange(n));cc.extend(basis.element_dofs[k,cells])
    return sparse.coo_matrix((vals,(rr,cc)),shape=(3*n,basis.N)).tocsr()


def moments(c,a,lo,hi):
    s=np.clip(np.array([(lo-c)/a,(hi-c)/a]),-1.,1.)
    r=np.sqrt(np.maximum(0,1-s*s));ang=np.arcsin(s)
    I0=.5*(s*r+ang);I1=-(1-s*s)**1.5/3;I2=(ang-s*(1-2*s*s)*r)/8
    i0,i1,i2=[np.diff(x)[0]*2/np.pi for x in (I0,I1,I2)]
    return i0,c*i0+a*i1,c*c*i0+2*c*a*i1+a*a*i2


class Model:
    def __init__(self,mesh,points,label):
        start=time.perf_counter();self.mesh=mesh;self.label=label
        self.b=Basis(mesh,ElementVector(ElementTriP2()),intorder=4)
        self.K=asm(elastic,self.b).tocsc()
        fixed=self.b.get_dofs(lambda x:np.isclose(x[0],0)).all()
        self.free=np.setdiff1d(np.arange(self.b.N),fixed)
        tf=time.perf_counter();self.Kfree=self.K[self.free][:,self.free];self.scale=1/np.sqrt(self.Kfree.diagonal());self.lu=splu((sparse.diags(self.scale)@self.Kfree@sparse.diags(self.scale)).tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True});self.factor_s=time.perf_counter()-tf
        self.S=stress_map(self.b,points)
        self.K_frobenius=float(np.linalg.norm(self.Kfree.data))
        self.faces=mesh.facets_satisfying(lambda x:np.isclose(x[0],4.))
        self.setup_s=time.perf_counter()-start
    def load(self,contact):
        c,a,F=contact;F=np.asarray(F,float);f=np.zeros(self.b.N)
        if c-a<-1-1e-12 or c+a>1+1e-12:raise ValueError('Patch outside physical edge')
        for face in self.faces:
            nodes=self.mesh.facets[:,face]
            if self.mesh.p[1,nodes[0]]>self.mesh.p[1,nodes[1]]:nodes=nodes[::-1]
            l,r=self.mesh.p[1,nodes];lo=max(l,c-a);hi=min(r,c+a)
            if hi<=lo:continue
            m0,m1,m2=moments(c,a,lo,hi);L=r-l
            t1=(m1-l*m0)/L;t2=(m2-2*l*m1+l*l*m0)/(L*L)
            weights=np.array([2*t2-3*t1+m0,2*t2-t1,4*t1-4*t2])
            dofs=np.column_stack((self.b.nodal_dofs[:,nodes],self.b.facet_dofs[:,face]))
            for k in range(2):np.add.at(f,dofs[k],F[k]*weights)
        return f
    def solve(self,contacts):
        f=sum((self.load(c) for c in contacts),start=np.zeros(self.b.N))
        u=np.zeros(self.b.N);u[self.free]=self.scale*self.lu.solve(self.scale*f[self.free]);res=f[self.free]-self.Kfree@u[self.free];u[self.free]+=self.scale*self.lu.solve(self.scale*res);return np.asarray(self.S@u),u,f
    def meta(self):
        return dict(label=self.label,nodes=int(self.mesh.nvertices),triangles=int(self.mesh.nelements),dofs=int(self.b.N),setup_s=self.setup_s,factor_s=self.factor_s,lu_nnz=int(self.lu.L.nnz+self.lu.U.nnz),lu_arrays_bytes=int(sum(a.nbytes for z in [self.lu.L,self.lu.U] for a in [z.data,z.indices,z.indptr])))


def frob(x):
    s=np.asarray(x).reshape(3,-1);return np.sqrt(np.sum(s[:2]**2)+2*np.sum(s[2]**2))

def relative(x,y):return float(frob(x-y)/max(frob(y),1e-30))

def vm(x):
    s=np.asarray(x).reshape(3,-1);return np.sqrt(np.maximum(s[0]**2-s[0]*s[1]+s[1]**2+3*s[2]**2,0))

def atlas(models,contacts,blend=False):
    groups={}
    for c in contacts:
        if blend:
            y=c[0];k=int(np.searchsorted(ANCHORS,y))
            if k==0:choices=[(0,1.)]
            elif k==len(ANCHORS):choices=[(len(ANCHORS)-1,1.)]
            else:
                w=(y-ANCHORS[k-1])/(ANCHORS[k]-ANCHORS[k-1]);choices=[(k-1,1-w),(k,w)]
        else:choices=[(int(np.argmin(np.abs(ANCHORS-c[0]))),1.)]
        for j,w in choices:
            if w>0:groups.setdefault(j,[]).append((c[0],c[1],np.asarray(c[2])*w))
    return sum((models[j].solve(cc)[0] for j,cc in groups.items()),start=np.zeros(models[0].S.shape[0]))


def timing(fn,reps=5):
    fn();t=[]
    for _ in range(reps):
        start=time.perf_counter();fn();t.append((time.perf_counter()-start)*1000)
    return dict(median_ms=float(np.median(t)),range_ms=[float(min(t)),float(max(t))])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Irrational-ish offsets avoid systematically probing the same mesh edges.
    xx=np.linspace(.113,3.947,107);yy=np.linspace(-.973,.973,83)
    X,Y=np.meshgrid(xx,yy);p=np.array([X.ravel(),Y.ravel()])
    valid=((p[0]<1.5)|(p[0]>2.5)|(np.abs(p[1])<.2))
    corners=np.array([[1.5,-.2],[1.5,.2],[2.5,-.2],[2.5,.2]])
    distance=np.linalg.norm(p[:,:,None]-corners.T[:,None,:],axis=0).min(axis=1)
    valid&=distance>.105
    p=p[:,valid];neck=(p[0]>1.7)&(p[0]<2.3)&(np.abs(p[1])<.16)
    contacts=lambda y,a=.12,t=.2:(float(y),float(a),[-1.,float(t)])
    cases=[]
    for y in [-.76,-.65,-.48,-.325,-.16,0,.16,.325,.48,.65,.76]:cases.append((f'move_{y:g}',[contacts(y)]))
    for a in [.06,.12,.20]:cases.append((f'radius_{a:g}',[contacts(.42,a)]))
    cases.extend([('two_far',[contacts(-.66,.12,.2),contacts(.61,.14,-.15)]),('two_near',[contacts(.22,.10,.25),contacts(.39,.13,-.1)]),('asymmetric',[contacts(-.72,.09,.1),contacts(.11,.16,-.2),contacts(.61,.12,.3)])])
    output=dict(scope='2D plane-stress P2 exact nonconvex dogbone, prescribed finite edge Hertz-line traction; no 3D or contact-solve claim. Common physical stress probes exclude clamp x<0.113 and radius0.105 around four sharp reentrant corners; full mathematical corner maximum is singular.',E=E,nu=NU,geometry=dict(length=4.,end_halfwidth=1.,neck_x=[1.5,2.5],neck_halfwidth=.2),anchors=ANCHORS.tolist(),probes=p.shape[1],models=[],cases=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),factorization='Equilibrated FP64 SuperLU MMD_AT_PLUS_A, SymmetricMode, no row threshold pivot for known SPD clamped elasticity; one FP64 residual correction',stress_metric='Unweighted Frobenius norm over identical physical sampled points, not a continuum-domain integral',fine_region_halfwidth=.36,knot_quality='minimum merged y-knot gap >=half requested contact step; exact domain boundaries retained',coverage_note='Selected fine windows need not contain every tested pressure interval; coverage failures are recorded and all physical loads still integrated, no support truncation')
    coarse=Model(grid(.04),p,'coarse');output['models'].append(coarse.meta())
    banks={}
    for neckref in [False,True]:
        name='contact_and_neck' if neckref else 'contact_only'
        models=[Model(grid(.04,[a],neckref),p,f'{name}_{a:g}') for a in ANCHORS]
        banks[name]=models;output['models'] += [m.meta() for m in models]
    union=Model(grid(.04,list(ANCHORS),True),p,'union_refined');fine2=Model(grid(.02,list(ANCHORS),True,2),p,'intermediate_reference');fine=Model(grid(.01,list(ANCHORS),True,4),p,'finer_reference')
    output['models'] += [union.meta(),fine2.meta(),fine.meta()]
    tensors={}
    def maskfield(s,mask):return s.reshape(3,-1)[:,mask].ravel()
    for name,cc in cases:
        ref,ur,fr=fine.solve(cc);ref2,_,_=fine2.solve(cc);u,su,fu=union.solve(cc);c,_,_=coarse.solve(cc)
        row=dict(name=name,contacts=cc,reference_change=relative(u,ref),reference_last_change=relative(ref2,ref),coarse_error=relative(c,ref),coarse_neck_error=relative(maskfield(c,neck),maskfield(ref,neck)),reference_sample_peak_vm=float(vm(ref).max()),reference_peak_position=p[:,np.argmax(vm(ref))].tolist())
        fields={'reference':ref,'coarse':c}
        for bn,models in banks.items():
            for blended in [False,True]:
                key=bn+('_blend' if blended else '_hard');s=atlas(models,cc,blended);fields[key]=s
                row[key]=dict(full_error=relative(s,ref),neck_error=relative(maskfield(s,neck),maskfield(ref,neck)),peak_vm=float(vm(s).max()),peak_relative_error=float(abs(vm(s).max()-vm(ref).max())/vm(ref).max()),error_to_union=relative(s,u))
        row['hard_fine_coverage']=[dict(anchor=float(ANCHORS[int(np.argmin(abs(ANCHORS-c[0])))]),margin=float(.36-abs(ANCHORS[int(np.argmin(abs(ANCHORS-c[0])))]-c[0])-c[1]),covered=bool(abs(ANCHORS[int(np.argmin(abs(ANCHORS-c[0])))]-c[0])+c[1]<=.36)) for c in cc]
        raw_residual=(fine.K@ur-fr)[fine.free]
        row['reference_residual']=float(np.linalg.norm(raw_residual)/np.linalg.norm(fr[fine.free]))
        row['reference_normwise_backward_error']=float(np.linalg.norm(raw_residual)/(fine.K_frobenius*np.linalg.norm(ur[fine.free])+np.linalg.norm(fr[fine.free])))
        total_force=np.array([fr[0::2].sum(),fr[1::2].sum()]);exact_force=np.sum([c[2] for c in cc],axis=0)
        moment=float(fine.b.doflocs[0,1::2]@fr[1::2]-fine.b.doflocs[1,0::2]@fr[0::2]);exact_moment=sum(4*c[2][1]-c[0]*c[2][0] for c in cc)
        row['load_force_relative']=float(np.linalg.norm(total_force-exact_force)/np.linalg.norm(exact_force));row['load_moment_relative']=float(abs(moment-exact_moment)/max(abs(exact_moment),1.))
        tensors[name]=fields;output['cases'].append(row);print(name,'coarse',row['coarse_error'],'atlas',row['contact_and_neck_hard']['full_error'],'refchange',row['reference_change'],flush=True)
    # Exact same-model superposition and equivalent coarse+delta algebra.
    cc=cases[-3][1];m=banks['contact_and_neck'][0]
    s_all=m.solve(cc)[0];s_sep=sum((m.solve([c])[0] for c in cc),start=np.zeros_like(s_all))
    co_all=coarse.solve(cc)[0]
    delta=co_all.copy()
    for c in cc:
        j=int(np.argmin(np.abs(ANCHORS-c[0])));delta+=banks['contact_and_neck'][j].solve([c])[0]-coarse.solve([c])[0]
    correct=atlas(banks['contact_and_neck'],cc)
    wrong=sum((banks['contact_and_neck'][j].solve(cc)[0] for j in [0,2]),start=np.zeros_like(correct))
    output['algebra']=dict(same_mesh_superposition=relative(s_sep,s_all),coarse_plus_delta=relative(delta,correct),wrong_full_load_sum_vs_correct=relative(wrong,correct))
    output['switches']=[]
    for center in [-.325,.325]:
        cc0=[contacts(center)];left=[contacts(center-1e-8)];right=[contacts(center+1e-8)]
        h0=atlas(banks['contact_and_neck'],cc0);b0=atlas(banks['contact_and_neck'],cc0,True)
        reference_motion=float(frob(fine.solve(right)[0]-fine.solve(left)[0])/frob(fine.solve(cc0)[0]))
        output['switches'].append(dict(center=center,reference_relative_motion=reference_motion,hard_relative_jump=float(frob(atlas(banks['contact_and_neck'],right)-atlas(banks['contact_and_neck'],left))/frob(h0)),blend_relative_jump=float(frob(atlas(banks['contact_and_neck'],right,True)-atlas(banks['contact_and_neck'],left,True))/frob(b0))))
    cc=cases[-3][1]
    output['timing_scope']='Sequential single-thread after setup, force integration+cached solve+all common stress probes; few contacts, no rigid robot/GPU. Same authored exact geometry, same load law; reference uncertainty/error not equal across methods.'
    output['timings']={name:timing(fn) for name,fn in [('coarse',lambda:coarse.solve(cc)),('atlas_hard',lambda:atlas(banks['contact_and_neck'],cc)),('atlas_blend',lambda:atlas(banks['contact_and_neck'],cc,True)),('union',lambda:union.solve(cc)),('fine',lambda:fine.solve(cc))]}
    output['selected_checks']=dict(superposition=output['algebra']['same_mesh_superposition']<1e-10,coarse_delta=output['algebra']['coarse_plus_delta']<1e-10,all_loads_conservative=max(r['load_force_relative'] for r in output['cases'])<1e-9 and max(r['load_moment_relative'] for r in output['cases'])<1e-9,all_linear_residuals=max(r['reference_residual'] for r in output['cases'])<1e-9)
    output['selected_checks_pass']=all(output['selected_checks'].values())
    output['strict_residual_gate']=dict(tolerance=1e-9,passed=output['selected_checks']['all_linear_residuals'],observed_max=max(r['reference_residual'] for r in output['cases']),note='Preserved strict diagnostic: FP64 raw residual can reach its cancellation floor. This failure is not silently reclassified; see separately reported normwise backward error and precision_diagnostic.json.')
    output['algebra_and_backward_checks_pass']=all(v for k,v in output['selected_checks'].items() if k!='all_linear_residuals') and max(r['reference_normwise_backward_error'] for r in output['cases'])<1e-12
    output['source_unchanged_during_run']=output['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'results.json').write_text(json.dumps(output,indent=2))
    np.savez_compressed(OUT/'fields.npz',points=p,**{key:val for key,val in tensors['two_far'].items()})
    plot(output,p,tensors['two_far'],coarse.mesh,banks['contact_and_neck'][0].mesh)
    print(json.dumps(dict(timings=output['timings'],algebra=output['algebra'],switches=output['switches'],checks=output['selected_checks'])),flush=True)


def plot(results,p,fields,coarse,feature):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    vmax=float(vm(fields['reference']).max())
    for ax,key,title in [(axes[0,0],'reference','Finer reference: sampled VM'),(axes[0,1],'coarse','Coarse: sampled VM'),(axes[1,0],'contact_only_hard','Contact-only atlas'),(axes[1,1],'contact_and_neck_hard','Contact + fixed-neck atlas')]:
        im=ax.scatter(p[0],p[1],c=vm(fields[key]),s=5,cmap='inferno',vmin=0,vmax=vmax,marker='s',linewidths=0)
        ax.set_aspect('equal');ax.set_title(title);ax.set(xlim=(-.05,4.05),ylim=(-1.05,1.05));ax.set_xlabel('x');ax.set_ylabel('y')
        fig.colorbar(im,ax=ax,shrink=.72,label='2D VM / unit thickness')
    fig.suptitle('Nonconvex dogbone: full sampled fields, finite edge tractions\nSharp reentrant corner exclusion is explicit; no continuum maximum claimed')
    fig.savefig(OUT/'stress_fields.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,m,title in [(axes[0],coarse,'Coarse mesh'),(axes[1],feature,'Feature mesh + fixed neck refinement')]:
        ax.triplot(m.p[0],m.p[1],m.t.T,linewidth=.25,color='#2f5d78');ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x');ax.set_ylabel('y')
    fig.savefig(OUT/'meshes.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
