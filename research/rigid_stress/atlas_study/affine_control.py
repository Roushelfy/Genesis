"""Independent constant-strain traction patch test of the 3D box model."""
from pathlib import Path
import hashlib
import json
import numpy as np
from skfem import FacetBasis, ElementVector, ElementTetP2, LinearForm, asm
from .volume.model import BoxModel, make_mesh, E, NU, L, HEIGHT


def run():
    mesh,_=make_mesh([])
    model=BoxModel(mesh,'affine_traction_control')
    a=np.array([1.e-6,-.4e-6,.3e-6])
    lam=E*NU/((1+NU)*(1-2*NU));mu=E/(2*(1+NU))
    grad=np.outer(a,[0.,0.,1.]);strain=.5*(grad+grad.T)
    sig=2*mu*strain+lam*np.trace(strain)*np.eye(3)
    @LinearForm
    def exact_boundary_traction(v,w):
        t=np.einsum('ij,j...->i...',sig,w.n)
        return np.sum(t*v,axis=0)
    fb=FacetBasis(mesh,ElementVector(ElementTetP2()),facets=mesh.boundary_facets(),intorder=2)
    force=asm(exact_boundary_traction,fb)
    exact=(model.scalar_basis.doflocs[2,:,None]*a).ravel()
    q=model.solve(force)
    rng=np.random.default_rng(38271)
    points=rng.uniform(size=(3,120));points[:2]=(points[:2]-.5)*.98*L;points[2]=(.01+.98*points[2])*HEIGHT
    operator=model.stress_map(points)
    s=(operator@q).reshape(-1,6)
    target=np.array([sig[0,0],sig[1,1],sig[2,2],sig[0,1],sig[0,2],sig[1,2]])
    result=dict(scope='3D P2 box: analytic constant-strain field reproduced from independently integrated exact boundary traction. Not a feature-atlas/contact-accuracy test.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dofs=int(model.N),
        model_sha256=hashlib.sha256((Path(__file__).resolve().parent/'volume/model.py').read_bytes()).hexdigest(),
        displacement_relative=float(np.linalg.norm(q-exact)/np.linalg.norm(exact)),
        stress_relative=float(np.linalg.norm(s-target)/(np.sqrt(len(s))*np.linalg.norm(target))),
        residual=model.residual(q,force),target_stress_voigt_Pa=target.tolist())
    result['passed']=max(result['displacement_relative'],result['stress_relative'],result['residual'])<1e-10
    (Path(__file__).resolve().parent/'affine_control.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Affine traction control failed')
    return result


if __name__=='__main__':run()
