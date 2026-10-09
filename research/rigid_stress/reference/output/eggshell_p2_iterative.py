#!/usr/bin/env python3
"""Low-memory exact-equation P2 convergence: geometric two-level PCG.

The coarse prism coordinates define a P2 prolongation. Different faceted
geometry is permitted in this preconditioner: the fine operator/load are
unchanged and true fine residuals and tolerance sensitivity are checked.
"""
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,cg
from threadpoolctl import threadpool_limits
import matplotlib.tri as mtri

import eggshell_p2_converged as nd
import eggshell_convergence_cpu as conv

HERE=Path(__file__).resolve().parent


def unit_and_depth(f):
    ns=f.nbase//(f.wall_layers+1);outer=f.base_xyz[:ns]
    uz=outer[:,2]/.030;scale=1-.18*uz
    unit=np.column_stack([outer[:,0]/(.022*scale),outer[:,1]/(.022*scale),uz])
    base=np.tile(unit,(f.wall_layers+1,1));depth=np.repeat(np.arange(f.wall_layers+1)/f.wall_layers,ns)
    a=f.edge_keys//f.nbase;b=f.edge_keys%f.nbase
    return np.r_[base,(base[a]+base[b])/2],np.r_[depth,(depth[a]+depth[b])/2],unit


def prolongation(coarse,fine):
    fu,depth,_=unit_and_depth(fine);_,_,cu=unit_and_depth(coarse)
    cl1=np.abs(cu).sum(axis=1);fl1=np.abs(fu).sum(axis=1)
    chart=cu[:,:2]/cl1[:,None];points=fu[:,:2]/fl1[:,None]
    face_idx=np.empty(len(fu),dtype=int);faces=coarse.outer_faces
    for sign in [1,-1]:
        ids=np.flatnonzero(sign*cu[faces].mean(axis=1)[:,2]>0)
        mask=fu[:,2]>=0 if sign==1 else fu[:,2]<0
        triangulation=mtri.Triangulation(chart[:,0],chart[:,1],triangles=faces[ids])
        # Move only the finder query inward to resolve floating boundary points;
        # actual interpolation uses the original point and preserves endpoints.
        p=points[mask]*(1-1e-13)+1e-13/3
        found=triangulation.get_trifinder()(p[:,0],p[:,1])
        assert found.min()>=0
        face_idx[mask]=ids[found]
    triangle=np.sort(faces[face_idx],axis=1);xy=chart[triangle]
    matrix=np.stack([xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]],axis=2)
    beta12=np.linalg.solve(matrix,(points-xy[:,0])[:,:,None])[:,:,0]
    beta=np.column_stack([1-beta12.sum(axis=1),beta12])
    angular=beta/cl1[triangle];angular/=angular.sum(axis=1)[:,None]
    assert angular.min()>-1e-9
    layer=np.minimum(np.floor(depth*coarse.wall_layers).astype(int),coarse.wall_layers-1)
    tau=depth*coarse.wall_layers-layer;a,b,c=angular.T
    which=np.where(tau<=c,0,np.where(tau<=b+c,1,2))
    lamb=np.empty((len(fu),4))
    mask=which==0;lamb[mask]=np.column_stack([a,b,c-tau,tau])[mask]
    mask=which==1;lamb[mask]=np.column_stack([a,b+c-tau,tau-c,c])[mask]
    mask=which==2;lamb[mask]=np.column_stack([1-tau,tau-b-c,b,c])[mask]
    assert lamb.min()>-1e-9
    tet=layer*len(faces)*3+face_idx*3+which
    indices=coarse.elements[tet]
    weights=np.column_stack([lamb*(2*lamb-1),4*lamb[:,conv.EDGES[:,0]]*lamb[:,conv.EDGES[:,1]]])
    assert np.max(np.abs(weights.sum(axis=1)-1))<1e-10
    nodes=sparse.coo_matrix((weights.ravel(),(np.repeat(np.arange(len(fu)),10),indices.ravel())),
                           shape=(len(fu),len(coarse.xyz))).tocsr();nodes.eliminate_zeros()
    return sparse.kron(nodes,sparse.eye(3),format='csr')[fine.free][:,coarse.free].tocsr()


def solve_job(coarse,level,layers,adaptive,out,export=False):
    label=f'QP2-L{level}-T{layers}'+(f'-A{adaptive}' if adaptive else '')
    f,geometry=nd.assemble(level,layers,adaptive);t=perf_counter()
    A=f.k[f.free][:,f.free].tocsr();P=prolongation(coarse,f);Pt=P.T.tocsr()
    diag=A.diagonal();invdiag=1/diag
    absA=A.copy();absA.data=np.abs(absA.data)
    bound=float(np.max(np.asarray(absA.sum(axis=1)).ravel()/diag));del absA;gc.collect()
    omega=1/bound
    def precondition(r):
        z=omega*invdiag*r;remaining=r-A@z
        z+=P@coarse.factor.solve(Pt@remaining)
        z+=omega*invdiag*(r-A@z)
        return z
    M=LinearOperator(A.shape,matvec=precondition,dtype=np.float64)
    raw,_=conv.grip_load(f,10);rhs=raw[f.free]
    x=P@coarse.factor.solve(Pt@rhs);setup_s=perf_counter()-t
    passes=[]
    for tolerance in [1e-8,1e-10]:
        niter=[0];t=perf_counter()
        def callback(xk):
            niter[0]+=1
            if niter[0]%100==0:
                print(json.dumps({'model':label,'tol':tolerance,'iteration':niter[0],
                                  'true_relative_residual':float(np.linalg.norm(A@xk-rhs)/np.linalg.norm(rhs)),
                                  'rss_MB':conv.rss_mb()}),flush=True)
        x,info=cg(A,rhs,x0=x,rtol=tolerance,atol=0.,M=M,maxiter=4000,callback=callback)
        residual=float(np.linalg.norm(A@x-rhs)/np.linalg.norm(rhs));seconds=perf_counter()-t
        assert info==0,(label,info,residual)
        u=np.zeros(f.ndof);u[f.free]=x
        metrics=f.metrics(u,raw,raw,float(np.linalg.norm((f.k@u-raw)[f.free])/max(np.linalg.norm(raw),1.)))
        passes.append({'rtol':tolerance,'iterations':niter[0],'solve_s':seconds,'true_relative_residual':residual,'metrics':metrics})
        print(json.dumps({'model':label,'tol':tolerance,'iterations':niter[0],'peak_MPa':metrics['max_von_Mises_Pa']/1e6,
                          'solve_s':seconds,'residual':residual}),flush=True)
    sensitivity=abs(passes[0]['metrics']['max_von_Mises_Pa']/passes[1]['metrics']['max_von_Mises_Pa']-1)
    assert sensitivity<1e-4,(label,sensitivity)
    edges=f.base_xyz[f.outer_faces];h=np.linalg.norm(edges-np.roll(edges,1,axis=1),axis=2)
    result={'label':label,'surface_level':level,'wall_layers':layers,'adaptive_refinements':adaptive,
            'nodes':len(f.xyz),'elements':f.ne,'DOFs':f.ndof,'stress_samples':4*f.ne,
            'mean_surface_edge_mm':float(h.mean()*1000),'minimum_surface_edge_mm':float(h.min()*1000),
            'assembly_s':f.assembly_s,'preconditioner_setup_s':setup_s,'rss_MB':conv.rss_mb(),
            'geometry':geometry,'solver':'two-level PCG, exact fine operator; coarse P2 L5 T2',
            'tolerance_peak_sensitivity':sensitivity,'passes':passes,'metrics':passes[-1]['metrics']}
    if export:
        class Factor:
            def solve(self,b):
                if b.ndim==1:
                    x,info=cg(A,b,rtol=1e-10,atol=0.,M=M,maxiter=4000)
                    assert info==0,('Response solve failed',info)
                    return x
                columns=[]
                for i in range(b.shape[1]):
                    x,info=cg(A,b[:,i],rtol=1e-10,atol=0.,M=M,maxiter=4000);assert info==0
                    columns.append(x)
                return np.column_stack(columns)
        f.factor=Factor();nd.export_response(f,label,out)
    del M,P,Pt,A,f;gc.collect()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='+',default=['6:2:2','6:4:0','7:2:0'])
    p.add_argument('--append',action='store_true');p.add_argument('--export',action='store_true');p.add_argument('--output-dir',type=Path,default=HERE)
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);path=out/'eggshell_p2_iterative_convergence.json'
    data=json.loads(path.read_text()) if args.append and path.exists() else {'jobs':[],'GPU_tested':False}
    with threadpool_limits(limits=1):
        coarse,_=nd.build(5,2,0)
        for i,s in enumerate(args.jobs):
            level,layers,adaptive=map(int,s.split(':'))
            if any(r['surface_level']==level and r['wall_layers']==layers and r['adaptive_refinements']==adaptive for r in data['jobs']):continue
            result=solve_job(coarse,level,layers,adaptive,out,export=args.export and i==len(args.jobs)-1)
            data['jobs'].append(result);path.write_text(json.dumps(data,indent=2))
    print('Iterative cross-refinement complete.',flush=True)


if __name__=='__main__':main()
