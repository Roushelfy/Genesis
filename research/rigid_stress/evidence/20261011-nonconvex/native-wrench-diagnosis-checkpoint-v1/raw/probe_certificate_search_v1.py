"""Offline test of bounded six-dimensional polar search, with uniform constraint penalties."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
from research.rigid_stress.wrench import project_coulomb

root=Path(__file__).parent
output=root/'certificate-search-v1'
output.mkdir(parents=True,exist_ok=False)
with np.load(root/'exact-602-socp-v1/component-1.npz') as saved:
    operator,normals,mu,target=(saved[key] for key in ('operator','normals','friction','target'))
normal=np.mean(normals,axis=0)
normal/=np.linalg.norm(normal)
compression=normals@normal
negative_support=-compression+mu*np.linalg.norm(normal-compression[:,None]*normals,axis=1)
depth=-negative_support.max()
record=[]
ray=target.copy()
for tau in (1e2,1e4,1e6,1e8,1e10):
    def objective(value):
        projected,jacobian=project_coulomb(np.einsum('qij,i->qj',operator,value),normals,mu)
        delta=value-target
        energy=0.5*(delta@delta+tau*np.einsum('qi,qi->',projected,projected)/len(normals))
        gradient=delta+tau*np.einsum('qij,qj->i',operator,projected)/len(normals)
        return energy,gradient
    def hessian(value):
        _,jacobian=project_coulomb(np.einsum('qij,i->qj',operator,value),normals,mu)
        return np.eye(6)+tau*np.einsum('qij,qjk,qlk->il',operator,jacobian,operator)/len(normals)
    result=minimize(objective,ray,jac=True,hess=hessian,method='trust-exact',
                    options={'gtol':1e-11,'maxiter':128,'initial_trust_radius':1.,'max_trust_radius':1e6})
    ray=result.x
    vectors=np.einsum('qij,i->qj',operator,ray)
    compression=np.einsum('qi,qi->q',vectors,normals)
    polar=compression+mu*np.linalg.norm(vectors-compression[:,None]*normals,axis=1)
    adjusted=ray.copy()
    if depth>0:
        adjusted[:3]-=max(polar.max()+1e-12,0)/depth*normal
    adjusted/=np.linalg.norm(adjusted)
    vectors=np.einsum('qij,i->qj',operator,adjusted)
    compression=np.einsum('qi,qi->q',vectors,normals)
    polar_adjusted=compression+mu*np.linalg.norm(vectors-compression[:,None]*normals,axis=1)
    item={'tau':tau,'evaluations':result.nfev,'iterations':result.nit,'gradient_norm':float(np.linalg.norm(result.jac)),
          'force_polar_interior_depth':float(depth),'polar_max':float(polar.max()),
          'adjusted_polar_max':float(polar_adjusted.max()),'adjusted_margin':float(target@adjusted),
          'adjusted_ray':adjusted.tolist()}
    record.append(item)
    print(json.dumps(item),flush=True)
    if polar_adjusted.max()<0 and target@adjusted>1e-8:
        np.savez_compressed(output/'certificate.npz',ray=adjusted)
        break
(output/'summary.json').write_text(json.dumps({'records':record},indent=2)+'\n')
