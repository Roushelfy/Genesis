"""Closed-egg balanced-load feature-mesh supplement.

All per-query source groups are complete balanced normal pairs, so no
mesh-dependent inertia relief is used. Geometry is sampled from the same
smooth authored egg, but Q4 approximations still differ across meshes.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]/'genesis-atlas'
if not (REPO/'research/rigid_stress/shell_cpu').exists():
    REPO=next(p for p in HERE.parents if (p/'research/rigid_stress/shell_cpu').exists())
sys.path.insert(0,str(REPO/'research/rigid_stress'))
from shell_cpu.core import ShellModel,element_operators
from shell_cpu.geometry import egg_geometry
from shell_cpu.contacts import Contact,ShellDPRTransfer

H=.0005;AXES=np.array([.022,.022,.030]);BETA=-.18
FRAMES=np.array([([1,0,0],[0,1,0],[0,0,1]),([-1,0,0],[0,-1,0],[0,0,1]),
                ([0,1,0],[-1,0,0],[0,0,1]),([0,-1,0],[1,0,0],[0,0,1]),
                ([0,0,1],[1,0,0],[0,1,0]),([0,0,-1],[-1,0,0],[0,1,0])])
ANCHORS=np.array([-.25,0,.25])

def knots(lo,hi,h):return np.linspace(lo,hi,int(round((hi-lo)/h))+1)
def unique(a):return np.unique(np.round(a,12))
def egg(d):
    d=np.asarray(d);d=d/np.linalg.norm(d,axis=-1,keepdims=True)
    scale=1+BETA*d[...,2]
    return d*AXES*np.stack([scale,scale,np.ones_like(scale)],axis=-1)

def mesh(axes):
    points=[];quads=[];index={};grids=[]
    for normal,first,second in FRAMES:
        def values(vector):
            j=np.argmax(abs(vector));return axes[j] if vector[j]>0 else -axes[j][::-1]
        U,V=values(first),values(second);table=np.empty((len(V),len(U)),int)
        start=len(quads)
        for j,v in enumerate(V):
            for i,u in enumerate(U):
                p=normal+u*first+v*second;key=tuple(np.round(p,12))
                if key not in index:index[key]=len(points);points.append(p)
                table[j,i]=index[key]
        for j in range(len(V)-1):
            for i in range(len(U)-1):quads.append([table[j,i],table[j,i+1],table[j+1,i+1],table[j+1,i]])
        grids.append((U,V,start))
    d=np.array(points);outer=egg(d);directors=egg_geometry(outer,axes=AXES,asym=BETA)[0]
    return outer-H/2*directors,np.array(quads),directors,grids

def probes():
    d=[]
    for normal,first,second in FRAMES:
        for v in np.linspace(-.87,.87,7):
            for u in np.linspace(-.87,.87,7):d.append(normal+u*first+v*second)
    for normal,first,second in FRAMES[:2]:
        for v in np.linspace(-.4,.0,13):
            for u in np.linspace(-.5,.5,25):d.append(normal+u*first+v*second)
    return np.array(d)

def stress_operator(model,grids,directions):
    maps=[];columns=[]
    for direction in directions:
        j=np.argmax(abs(direction));f=2*j+int(direction[j]<0)
        normal,first,second=FRAMES[f];U,V,start=grids[f]
        ratio=direction@normal;u,v=direction@first/ratio,direction@second/ratio
        ix=np.clip(np.searchsorted(U,u,side='right')-1,0,len(U)-2)
        iy=np.clip(np.searchsorted(V,v,side='right')-1,0,len(V)-2)
        xi=2*(u-U[ix])/(U[ix+1]-U[ix])-1;eta=2*(v-V[iy])/(V[iy+1]-V[iy])-1
        e=start+iy*(len(U)-1)+ix;ids=model.quads[e]
        bm,bb,_s,_d,A,_j=element_operators(model.vertices[ids],model.directors[ids],xi,eta)
        a,b=A[:,0],A[:,1]
        basis=np.stack((np.outer(a,a),np.outer(b,b),np.outer(a,b)+np.outer(b,a)),axis=-1).reshape(9,3)
        maps.append([basis@model.C@(bm+side*H/2*bb) for side in (1,-1)])
        columns.append(model.element_dofs[e])
    M=np.array(maps).transpose(1,0,2,3);cols=np.broadcast_to(np.array(columns)[None,:,None,:],M.shape).ravel()
    rows=np.repeat(np.arange(2*len(directions)*9),24)
    return sparse.coo_matrix((M.ravel(),(rows,cols)),shape=(2*len(directions)*9,model.ndof)).tocsr()

class Egg:
    def __init__(self,name,axes,directions):
        self.name=name;t=time.perf_counter();v,q,d,g=mesh(axes)
        self.model=ShellModel(v,q,E=1e10,nu=.3,h=H,directors=d,gauge='translation')
        self.assembly_s=time.perf_counter()-t;self.transfer=ShellDPRTransfer(self.model)
        self.free=np.setdiff1d(np.arange(self.model.ndof),self.model.gauge_dofs)
        K=self.model.K[self.free][:,self.free].tocsc();self.D=1/np.sqrt(K.diagonal())
        t=time.perf_counter();self.lu=splu((sparse.diags(self.D)@K@sparse.diags(self.D)).tocsc(),permc_spec='MMD_AT_PLUS_A');self.factor_s=time.perf_counter()-t
        self.S=stress_operator(self.model,g,directions)
    def load_pair(self,theta,fn=.05,radius=.002):
        z=(-1+np.sqrt(1+8*BETA*BETA))/(4*BETA);rad=np.sqrt(1-z*z)
        dirs=np.array([[rad*np.cos(theta),rad*np.sin(theta),z],[-rad*np.cos(theta),-rad*np.sin(theta),z]])
        points=egg(dirs);normals,k=egg_geometry(points,axes=AXES,asym=BETA)
        loads=[];infos=[]
        for p,n,c in zip(points,normals,k):
            force=-fn*n
            # The inherited oracle's exactly-zero-mu check has no absolute
            # roundoff allowance and can reject an exactly parallel fixture.
            # No friction law is needed for these declared normal-only pairs.
            # Retain the initial failures; record the actual roundoff shear.
            f,info=self.transfer.transfer(Contact(p,force,n,c,radius=radius,friction=None),order=16)
            unit=n/np.linalg.norm(n);normal_force=-force@unit
            info['fixture_relative_tangential_roundoff']=float(np.linalg.norm(force+normal_force*unit)/normal_force)
            loads.append(f);infos.append(info)
        f=sum(loads)
        wrench=self.model.rigid_modes.T@f
        return f,dict(raw_wrench_norm=float(np.linalg.norm(wrench)),contact_checks=infos)
    def field(self,f):
        q=np.zeros(self.model.ndof);q[self.free]=self.D*self.lu.solve(self.D*f[self.free])
        s=np.asarray(self.S@q).reshape(2,-1,3,3)
        residual=self.model.K@q-f
        return s,float(np.linalg.norm(residual)/max(np.linalg.norm(f),1e-30))
    def metadata(self):return dict(name=self.name,dofs=self.model.ndof,quads=len(self.model.quads),assembly_s=self.assembly_s,factor_s=self.factor_s)

def vm(s):
    dev=s-np.trace(s,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3
    return np.sqrt(1.5*np.sum(dev*dev,axis=(-2,-1)))
def err(a,b):return dict(tensor_relative=float(np.linalg.norm(a-b)/np.linalg.norm(b)),sampled_peak_vm_Pa=float(vm(a).max()),peak_relative=float(abs(vm(a).max()/vm(b).max()-1)))

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);directions=probes();base=np.linspace(-1,1,5);models=[]
    models.append(Egg('coarse',[base,base,base],directions))
    z=unique(np.r_[base,knots(-.5,.2,.1)])
    for i,a in enumerate(ANCHORS):
        y=unique(np.r_[base,knots(a-.35,a+.35,.1)])
        models.append(Egg(f'atlas{i}',[base,y,z],directions));print('built',models[-1].metadata(),flush=True)
    for name,h in [('joint',.06),('reference',.04)]:
        y=unique(np.r_[base,knots(-.6,.6,h)]);z=unique(np.r_[base,knots(-.5,.2,h)])
        models.append(Egg(name,[base,y,z],directions));print('built',models[-1].metadata(),flush=True)
    queries=[('angle_'+str(t),[(t,.05,.002)]) for t in [-.23,-.12,0.,.12,.23]]
    queries+=[('two_balanced_pairs',[(-.17,.035,.002),(.16,.05,.0015)]),('auto_tiny',[(.07,.05,None)])]
    rows=[];failures=[];saved={}
    for name,pairs in queries:
        fields={};checks={};start=time.perf_counter()
        try:
            for m in models:
                ff=[];gg=[]
                for theta,fn,a in pairs:
                    load,info=m.load_pair(theta,fn,a);s,res=m.field(load);info['solve_residual']=res;ff.append(s);gg.append(info)
                fields[m.name]=ff;checks[m.name]=gg
            reference=sum(fields['reference']);atlas=sum(fields[f'atlas{int(np.argmin(abs(ANCHORS-np.tan(t))))}'][j] for j,(t,_,_) in enumerate(pairs))
            row=dict(case=name,pairs=pairs,metrics={key:err(sum(fields[key]),reference) for key in ['coarse','joint']},atlas=err(atlas,reference),checks=checks,seconds=time.perf_counter()-start)
            rows.append(row)
            if name=='angle_0.0':saved=dict(reference=reference,atlas=atlas,coarse=sum(fields['coarse']))
        except Exception as e:
            failures.append(dict(case=name,error=repr(e)))
        print(json.dumps(rows[-1] if rows and rows[-1]['case']==name else failures[-1]),flush=True)
        (out/'results.json').write_text(json.dumps(dict(scope='Closed egg supplemental shell-only common-material-coordinate stress comparison. Balanced source pairs preserve zero resultant and moment before any solve; inertia relief disabled. Same smooth authored geometry but differing Q4 approximations. No 3D peak qualification.',normal_fixture_note='Pure normal prescribed source; no friction-cone law requested. Initial explicit mu=0 oracle rejects roundoff tangential components and is retained separately. Actual tangential roundoff is recorded; forces and integration gates unchanged.',models=[m.metadata() for m in models],rows=rows,failures=failures,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    if saved:np.savez_compressed(out/'fields.npz',directions=directions,**saved)
    print('FINISHED',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'egg_results');a=p.parse_args()
    with threadpool_limits(limits=1):main(a.output)
