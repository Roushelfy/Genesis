"""Actual free-body P2 FEM control of common consistent-mass inertia relief.

No physical clamp is retained. This is one exact box mesh, not a certification
that distinct feature meshes implement identical mass quadratures.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import qr
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import roots_jacobi
from skfem import Basis, ElementVector, ElementTetP2, BilinearForm, asm
from .volume.model import BoxModel,make_mesh,Contact,L,HEIGHT


def run():
    mesh,_=make_mesh([]);model=BoxModel(mesh,'freebody_control')
    coords=model.scalar_basis.doflocs.T;center=np.array([0.,0.,HEIGHT/2])
    R=np.zeros((model.N,6))
    for j in range(3):
        R[j::3,j]=1.
        R[:,3+j]=np.cross(np.eye(3)[j],coords-center).ravel()
    @BilinearForm
    def mass_form(u,v,w):return np.sum(u*v,axis=0)
    basis=Basis(mesh,ElementVector(ElementTetP2()),intorder=4)
    M=asm(mass_form,basis).tocsr();MR=M@R;G=R.T@MR
    mass=L*L*HEIGHT
    Gexact=np.diag([mass,mass,mass,mass*(L*L+HEIGHT*HEIGHT)/12,
                    mass*(L*L+HEIGHT*HEIGHT)/12,mass*(2*L*L)/12])
    normalized=R.copy();normalized[:,3:]/=L
    _,_,pivot=qr(normalized.T,pivoting=True,mode='economic')
    free=np.setdiff1d(np.arange(model.N),pivot[:6])
    A=model.K[free][:,free].tocsc();D=1/np.sqrt(A.diagonal())
    lu=splu((sparse.diags(D)@A@sparse.diags(D)).tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,options={'SymmetricMode':True})
    def solve(f):
        b=f-MR@np.linalg.solve(G,R.T@f)
        q=np.zeros_like(f);q[free]=D*lu.solve(D*b[free])
        return q,b
    c=Contact((.0007,-.0004),.001,(.01,-.006,-.05))
    ftop,info=model.load(c,order=24)
    # Mirror the identical nodal traction distribution onto the bottom surface.
    # Tensor symmetry of this coarse authored box makes this node map exact.
    from scipy.spatial import cKDTree
    mirror=coords.copy();mirror[:,2]=HEIGHT-mirror[:,2]
    dist,ids=cKDTree(coords).query(mirror)
    assert dist.max()<1e-12
    fbottom=np.zeros_like(ftop)
    fbottom.reshape(-1,3)[ids]=-ftop.reshape(-1,3)
    # Tangential sourceoffset creates a real net torque; relief handles it.
    q1,b1=solve(ftop);q2,b2=solve(fbottom);q,b=solve(ftop+fbottom)
    points=np.array([[.001,-.002,.003],[-.001,.0015,.002],[.009,.005,.001]])
    S=model.stress_map(points)
    null=float(np.linalg.norm(model.K@R)/(np.linalg.norm(model.K.data)*np.linalg.norm(R)))
    out=dict(scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((Path(__file__).resolve().parent/'volume/model.py').read_bytes()).hexdigest(),
        dofs=int(model.N),consistent_mass_gram_relative=float(np.linalg.norm(G-Gexact)/np.linalg.norm(Gexact)),
        rigid_stiffness_relative=null,common_projection_linear_relative=float(np.linalg.norm(b-b1-b2)/np.linalg.norm(b)),
        full_stress_superposition_relative=float(np.linalg.norm(S@(q-q1-q2))/np.linalg.norm(S@q)),
        compatible_resultant_relative=float(np.linalg.norm(R.T@b)/(np.linalg.norm(R)*np.linalg.norm(b))),
        full_FEM_residual_relative=float(np.linalg.norm(model.K@q-b)/np.linalg.norm(b)),
        top_contact_checks=info)
    out['passed']=max(out[k] for k in ['consistent_mass_gram_relative','rigid_stiffness_relative','common_projection_linear_relative','full_stress_superposition_relative','compatible_resultant_relative','full_FEM_residual_relative'])<1e-10
    (Path(__file__).resolve().parent/'freebody_control.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    if not out['passed']:raise RuntimeError('Free-body FEM control failed')
    return out


if __name__=='__main__':run()
