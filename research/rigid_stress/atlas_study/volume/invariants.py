"""Focused independent P2 operator/traction checks, separate from atlas errors."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi
from threadpoolctl import threadpool_limits
from .model import *
ROOT=Path(__file__).resolve().parent

def run():
    model=BoxModel(make_mesh()[0]);points=np.array([[-.004,.003,.001,-.007],[.002,-.001,.005,.003],[.008,.004,.006,.002]])
    S=model.stress_map(points)
    G=np.array([[.001,.002,.0003],[-.001,.003,.0014],[.0006,-.001,-.002]])
    q=(model.scalar_basis.doflocs.T@G.T).ravel()
    exact=model.lam*np.trace(G)*np.eye(3)+model.mu*(G+G.T)
    affine=float(np.linalg.norm(tensor(S@q)-exact)/np.sqrt(len(points.T))/np.linalg.norm(exact))
    xx,yy,zz=model.scalar_basis.doflocs
    quadratic=np.column_stack((xx*xx+yy*zz,xx*yy+zz*zz,yy*yy+xx*zz)).ravel()
    xx,yy,zz=points
    gradients=np.stack((np.column_stack((2*xx,zz,yy)),np.column_stack((yy,xx,2*zz)),np.column_stack((zz,2*yy,xx))),axis=1)
    analytic=model.lam*np.trace(gradients,axis1=1,axis2=2)[:,None,None]*np.eye(3)+model.mu*(gradients+gradients.transpose(0,2,1))
    quadratic_error=float(np.linalg.norm(tensor(S@quadratic)-analytic)/np.linalg.norm(analytic))
    norms=[]
    x=model.scalar_basis.doflocs.T
    for i in range(3):
        q=np.broadcast_to(np.eye(3)[i],x.shape).ravel();norms.append(float(np.linalg.norm(model.K@q)/(np.linalg.norm(model.K.data)*np.linalg.norm(q))))
        q=np.cross(np.eye(3)[i],x).ravel();norms.append(float(np.linalg.norm(model.K@q)/(np.linalg.norm(model.K.data)*np.linalg.norm(q))))
    c1=Contact((-.002,.001),.001,(.01,.005,-.05));c2=Contact((.003,-.001),.0013,(-.012,.003,-.04))
    f1,i1=model.load(c1,order=12);f2,i2=model.load(c2,order=12);u1,u2=model.solve(f1),model.solve(f2)
    reciprocity=float(abs(f1@u2-f2@u1)/max(abs(f1@u2),abs(f2@u1)))
    qorder=4;v,w=roots_jacobi(qorder,.5,0);r=c1.radius*np.sqrt((v+1)/2);t=2*np.pi*(np.arange(4*qorder)+.5)/(4*qorder)
    p=np.stack((c1.center[0]+r[:,None]*np.cos(t),c1.center[1]+r[:,None]*np.sin(t),np.full((qorder,4*qorder),HEIGHT)),axis=0).reshape(3,-1)
    weights=np.broadcast_to((w/w.sum())[:,None]/(4*qorder),(qorder,4*qorder)).ravel()
    independent=(np.asarray(model.scalar_basis.probes(p).T@weights).ravel()[:,None]*c1.force).ravel()
    f,_=model.load(c1,order=qorder)
    probe_error=float(np.linalg.norm(f-independent)/np.linalg.norm(independent))
    checks=dict(affine_constitutive_map=affine<1e-12,quadratic_constitutive_map=quadratic_error<1e-12,rigid_zero_stiffness=max(norms)<1e-13,reciprocity=reciprocity<1e-10,independent_shape_scatter=probe_error<1e-12)
    out=dict(source_sha256=hashlib.sha256((ROOT/'model.py').read_bytes()).hexdigest(),affine_tensor_relative=affine,quadratic_tensor_relative=quadratic_error,rigid_stiffness_relative=norms,reciprocity_relative=reciprocity,independent_scatter_relative=probe_error,load_checks=[i1,i2],checks=checks,all_passed=all(checks.values()))
    (ROOT/'invariants.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
