#!/usr/bin/env python3
"""Complete crossed P2 convergence using node-block nested dissection.

Uses the existing P2 assembly and symmetric central 4 N/finger load unchanged.
Only the direct-solver ordering changes. Original files are not modified.
Optional response export uses three symmetry-compatible normal-load heights;
it is not the earlier arbitrary 19-dimensional full-body load space.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pymetis
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

import eggshell_convergence_cpu as conv

HERE=Path(__file__).resolve().parent


def column_graph(f):
    """Collapse the shell-normal stack into its quadratic surface column.

    Each volume vertex maps to a surface vertex, each edge midpoint to either
    that vertex or its surface edge. ND on this exact connectivity graph then
    expands each surface column into all its volume displacement unknowns.
    This avoids arbitrary separators splitting the thin wall along its depth.
    """
    ns=int(f.base_xyz.shape[0]/(f.wall_layers+1))
    ids=np.arange(f.nbase)%ns
    a=(f.edge_keys//f.nbase)%ns;b=(f.edge_keys%f.nbase)%ns
    lo=np.minimum(a,b);hi=np.maximum(a,b);distinct=lo!=hi
    keys=lo[distinct]*ns+hi[distinct];unique,inverse=np.unique(keys,return_inverse=True)
    mids=lo.copy();mids[distinct]=ns+inverse
    columns=np.r_[ids,mids];nc=ns+len(unique)
    faces=f.outer_faces;fe=np.sort(faces[:,[[0,1],[0,2],[1,2]]],axis=2)
    key=fe[:,:,0]*ns+fe[:,:,1];edgecols=ns+np.searchsorted(unique,key)
    assert np.all(unique[edgecols-ns]==key)
    fc=np.column_stack([faces,edgecols])
    from scipy import sparse
    row=np.repeat(fc,6,axis=1).ravel();col=np.tile(fc,(1,6)).ravel()
    graph=sparse.coo_matrix((np.ones(len(row)),(row,col)),shape=(nc,nc)).tocsr()
    graph.setdiag(0);graph.eliminate_zeros()
    return graph,columns


class OrderedFactor:
    def __init__(self,f):
        t=perf_counter()
        graph,columns=column_graph(f)
        _,inverse_columns=pymetis.nested_dissection(adjacency=pymetis.CSRAdjacency(graph.indptr,graph.indices),
                                                    options=pymetis.Options(seed=17))
        nodeperm=np.argsort(np.asarray(inverse_columns)[columns],kind='stable')
        globalperm=(3*np.asarray(nodeperm)[:,None]+np.arange(3)).ravel()
        inverse=np.full(f.ndof,-1,dtype=np.int32);inverse[f.free]=np.arange(len(f.free))
        self.perm=inverse[globalperm];self.perm=self.perm[self.perm>=0]
        assert len(self.perm)==len(f.free)
        del graph,columns,inverse_columns,nodeperm,globalperm,inverse;gc.collect()
        self.order_s=perf_counter()-t;t=perf_counter()
        k=f.k[f.free][:,f.free];ordered=k[self.perm][:,self.perm];del k;gc.collect()
        print(json.dumps({'phase':'ND-factor','nodes':len(f.xyz),'rss_MB':conv.rss_mb()}),flush=True)
        self.lu=splu(ordered,permc_spec='NATURAL',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        self.factor_s=perf_counter()-t;self.nnz=self.lu.nnz

    def solve(self,b):
        ordered=self.lu.solve(np.asfortranarray(b[self.perm]))
        result=np.empty_like(ordered);result[self.perm]=ordered
        return result


def assemble(level,layers,adaptive):
    xyz,tet,outer,meta=conv.quarter_mesh(level,layers,adaptive)
    # Reuse exactly the verified assembly, defer only the original LU call.
    with patch.object(conv,'splu',lambda *args,**kwargs:None):
        f=conv.FEM(xyz,tet,outer,2,label=f'ND-QP2-L{level}-T{layers}-A{adaptive}',quarter=True)
    f.wall_layers=layers
    # Canonical vertical translation needs only the consistent mass row sums.
    nodal_mass=np.asarray(f.m.sum(axis=1)).ravel()[::3]
    def canonical(u):
        result=u.copy();result[2::3]-=float(nodal_mass@u[2::3])/f.mass
        return result
    f.canonical=canonical
    del f.m,f.r,f.mr,f.gram,f.gfactor;gc.collect()
    return f,meta


def build(level,layers,adaptive):
    f,meta=assemble(level,layers,adaptive)
    f.factor=OrderedFactor(f);f.factor_s=f.factor.factor_s
    return f,meta


def run_job(level,layers,adaptive,quad=10,export=False,out=HERE):
    label=f'QP2-L{level}-T{layers}'+(f'-A{adaptive}' if adaptive else '')
    f,geometry=build(level,layers,adaptive)
    raw,_=conv.grip_load(f,quad);t=perf_counter();u,rhs,res=f.solve(raw);solve_s=perf_counter()-t
    metrics=f.metrics(u,raw,rhs,res)
    edges=f.base_xyz[f.outer_faces[:,[0,1,2]]]
    lengths=np.linalg.norm(edges-np.roll(edges,1,axis=1),axis=2)
    result={'label':label,'surface_level':level,'wall_layers':layers,'adaptive_refinements':adaptive,
            'nodes':len(f.xyz),'elements':f.ne,'DOFs':f.ndof,'stress_samples':4*f.ne,
            'mean_surface_edge_mm':float(lengths.mean()*1000),'minimum_surface_edge_mm':float(lengths.min()*1000),
            'assembly_s':f.assembly_s,'ordering_s':f.factor.order_s,'factor_s':f.factor_s,
            'LU_nnz':f.factor.nnz,'rhs_solve_s':solve_s,'rss_MB':conv.rss_mb(),
            'ordering':'surface-column METIS nested dissection',
            'geometry':geometry,'metrics':metrics}
    # Verify actual relative equilibrium residual too, beyond the historical 1N scaling.
    result['relative_free_equilibrium_residual']=float(np.linalg.norm((f.k@u-rhs)[f.free])/np.linalg.norm(rhs[f.free]))
    assert result['relative_free_equilibrium_residual']<1e-7
    print(json.dumps({'completed':label,'peak_MPa':metrics['max_von_Mises_Pa']/1e6,
                      'compliance_full_microNm':metrics['compliance_Nm']*4e6,
                      'factor_s':f.factor_s,'rss_MB':conv.rss_mb()}),flush=True)
    if export:
        export_response(f,label,out)
    del f;gc.collect()
    return result


def export_response(f,label,out):
    t=perf_counter()
    h=np.column_stack([conv.grip_load(f,10,force=1.,z=z)[0] for z in [-.010,0.,.012]])
    ubasis=np.zeros_like(h);ubasis[f.free]=f.factor.solve(h[f.free])
    A=np.empty((f.ne*4,6,3))
    for lo in range(0,f.ne,512):
        hi=min(f.ne,lo+512);g=conv.shape_gradients(f.glambda[lo:hi],np.eye(4),2)
        v=ubasis[f.dofs[lo:hi]].reshape(-1,10,3,3)
        du=np.einsum('eiar,eqib->eqabr',v,g,optimize=True)
        eps=np.stack([du[:,:,0,0],du[:,:,1,1],du[:,:,2,2],
                      du[:,:,0,1]+du[:,:,1,0],du[:,:,1,2]+du[:,:,2,1],du[:,:,0,2]+du[:,:,2,0]],axis=2)
        stress=np.einsum('sc,eqcr->eqsr',f.d,eps,optimize=True).reshape(-1,6,3)
        a=A[lo*4:hi*4]
        a[:,0]=(stress[:,0]-stress[:,1])/np.sqrt(2);a[:,1]=(stress[:,1]-stress[:,2])/np.sqrt(2)
        a[:,2]=(stress[:,2]-stress[:,0])/np.sqrt(2);a[:,3:]=np.sqrt(3)*stress[:,3:]
    q=np.array([0.,4.,0.]);peak=np.linalg.norm(A@q,axis=1).max()
    raw=h@q;u,rhs,res=f.solve(raw);direct=f.metrics(u,raw,rhs,res)['max_von_Mises_Pa']
    assert abs(peak-direct)/direct<1e-8
    path=out/'eggshell_converged_response.npz'
    np.savez_compressed(path,A=A,load_heights_m=np.array([-.010,0.,.012]),central_4N_peak_Pa=direct,
                        model_label=label,scope='Quarter symmetry: opposite normal forces at three heights. Central load is convergence tested.')
    print(json.dumps({'response_export':str(path),'bytes':A.nbytes,'seconds':perf_counter()-t,'error':abs(peak-direct)/direct}),flush=True)


def save(data,out):
    (out/'eggshell_p2_complete_convergence.json').write_text(json.dumps(data,indent=2))
    rows=[]
    for r in data['jobs']:
        rows.append({**{k:v for k,v in r.items() if np.isscalar(v)},**r['metrics']})
    if rows:
        names=list(dict.fromkeys(k for r in rows for k in r))
        with (out/'eggshell_p2_complete_convergence.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='+',default=['5:2:0','6:2:0','6:2:1','6:2:2','6:4:0','7:2:0'])
    p.add_argument('--append',action='store_true');p.add_argument('--export',action='store_true')
    p.add_argument('--output-dir',type=Path,default=HERE);args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    path=out/'eggshell_p2_complete_convergence.json'
    data=json.loads(path.read_text()) if args.append and path.exists() else {'jobs':[],
         'force_per_finger_N':4.,'thickness_m':.0005,'ordering':'METIS surface-column nested dissection then natural-order SuperLU',
         'criterion':'Crossed surface/local/thickness changes; aim <1% peak and <0.5% compliance; not a rigorous true-error bound.',
         'GPU_device_available':False}
    with threadpool_limits(limits=1):
        for i,spec in enumerate(args.jobs):
            level,layers,adaptive=map(int,spec.split(':'))
            if any(r['surface_level']==level and r['wall_layers']==layers and r['adaptive_refinements']==adaptive for r in data['jobs']):continue
            data['jobs'].append(run_job(level,layers,adaptive,export=args.export and i==len(args.jobs)-1,out=out));save(data,out)
    print('Completed requested convergence jobs.',flush=True)


if __name__=='__main__':main()
