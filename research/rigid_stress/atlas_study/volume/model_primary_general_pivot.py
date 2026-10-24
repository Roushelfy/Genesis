"""Independent CPU P2 tetrahedral linear-elasticity atlas control.

Exact common box geometry, bottom clamp, finite Hertz-disk applied tractions.
All model instances use the same physical material and boundary conditions.
Locally refined conforming meshes use scikit-fem longest-edge bisection.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import roots_jacobi
from skfem import MeshTet, Basis, ElementTetP2, ElementVector, asm
from skfem.models.elasticity import linear_elasticity

L=.02
HEIGHT=.01
E=1.e10
NU=.3

@dataclass
class Contact:
    center: tuple[float,float]
    radius: float
    force: tuple[float,float,float]


def make_mesh(anchors=(), hmin=.0008, core=.0024, max_levels=24):
    """Fixed mesh around OFFLINE anchors, not current queried contacts.

    Additional refinements needed for conformity are retained. hmin is a target
    edge scale rather than a rigorous bound on a continuum stress error.
    """
    mesh=MeshTet.init_tensor(np.linspace(-L/2,L/2,5),np.linspace(-L/2,L/2,5),np.linspace(0,HEIGHT,4))
    anchors=np.asarray([[x,y,HEIGHT] for x,y in anchors],float)
    hist=[]
    if not len(anchors):return mesh,hist
    for level in range(max_levels):
        xyz=mesh.p[:,mesh.t].transpose(2,1,0)
        center=xyz.mean(axis=1)
        distances=np.linalg.norm(center[:,None,:]-anchors[None,:,:],axis=2).min(axis=1)
        h=np.max(np.linalg.norm(xyz[:,:,None,:]-xyz[:,None,:,:],axis=-1),axis=(1,2))
        goal=np.minimum(.009,hmin+.65*np.maximum(distances-core,0.))
        marked=np.flatnonzero(h>1.18*goal)
        hist.append(dict(level=level,tets=mesh.nelements,marked=len(marked),target=hmin))
        if not len(marked):break
        mesh=mesh.refined(marked)
    else:raise RuntimeError('Mesh refinement did not terminate')
    return mesh,hist


class BoxModel:
    def __init__(self,mesh,name='model'):
        self.name=name;self.mesh=mesh
        start=perf_counter()
        self.scalar_basis=Basis(mesh,ElementTetP2(),intorder=2)
        self.basis=Basis(mesh,ElementVector(ElementTetP2()),intorder=2)
        self.N=self.basis.N
        self.lam=E*NU/((1+NU)*(1-2*NU));self.mu=E/(2*(1+NU))
        self.K=asm(linear_elasticity(self.lam,self.mu),self.basis).tocsr()
        self.fixed=np.flatnonzero(np.repeat(np.isclose(self.scalar_basis.doflocs[2],0.,atol=1e-14),3))
        self.free=np.setdiff1d(np.arange(self.N),self.fixed)
        self.assembly_seconds=perf_counter()-start
        start=perf_counter()
        A=self.K[self.free][:,self.free].tocsc()
        self.scale=1/np.sqrt(A.diagonal())
        self.factor=splu((sparse.diags(self.scale)@A@sparse.diags(self.scale)).tocsc(),permc_spec='MMD_AT_PLUS_A')
        self.factor_seconds=perf_counter()-start
        x=self.mesh.p[:,self.mesh.t].transpose(2,1,0)
        self.origins=x[:,0]
        self.inverse=np.linalg.inv(np.stack((x[:,1]-x[:,0],x[:,2]-x[:,0],x[:,3]-x[:,0]),axis=-1))
        self.bary_grad=np.concatenate((-self.inverse.sum(axis=1)[:,None,:],self.inverse),axis=1)
        self.edges=np.array([[0,1],[1,2],[0,2],[0,3],[1,3],[2,3]])
        self.finder=self.mesh.element_finder()

    def solve(self,force):
        f=np.asarray(force,dtype=float)
        q=np.zeros_like(f)
        d=self.scale if f.ndim==1 else self.scale[:,None]
        q[self.free]=d*self.factor.solve(np.asfortranarray(d*f[self.free]))
        return q

    def residual(self,q,f):
        r=(self.K@q-f)[self.free]
        return float(np.linalg.norm(r)/max(np.linalg.norm(f[self.free]),1e-30))

    def locate(self,points):
        # scikit-fem's finder tests a cross-product of ALL queries with the
        # union of their candidate elements, with an all-element fallback.
        # Bounded query chunks preserve the exact search while avoiding
        # quadratic scratch on a global probe bank.
        return np.concatenate([self.finder(*points[:,i:i+64]) for i in range(0,points.shape[1],64)])

    def load(self,contact,order=24):
        start=perf_counter();n=int(order)
        x,w=roots_jacobi(n,.5,0.);rr=contact.radius*np.sqrt((x+1)/2)
        theta=2*np.pi*(np.arange(4*n)+.5)/(4*n)
        pos=np.stack((contact.center[0]+rr[:,None]*np.cos(theta),contact.center[1]+rr[:,None]*np.sin(theta),np.full((n,4*n),HEIGHT)),axis=0).reshape(3,-1)
        if np.any(np.abs(pos[:2])>=L/2):raise ValueError('Footprint outside authored top face')
        weights=np.broadcast_to((w/w.sum())[:,None]/(4*n),(n,4*n)).ravel()
        elem=self.locate(pos)
        local=np.einsum('pij,pj->pi',self.inverse[elem],pos.T-self.origins[elem])
        bary=np.column_stack((1-local.sum(axis=1),local))
        shape=np.empty((len(elem),10));shape[:,:4]=bary*(2*bary-1)
        for k,(a,b) in enumerate(self.edges):shape[:,4+k]=4*bary[:,a]*bary[:,b]
        nodal=np.zeros(self.scalar_basis.N)
        np.add.at(nodal,self.scalar_basis.element_dofs[:,elem].T.ravel(),(shape*weights[:,None]).ravel())
        load=(nodal[:,None]*np.asarray(contact.force)).ravel()
        force=load.reshape(-1,3).sum(axis=0)
        moment=np.cross(self.scalar_basis.doflocs.T,load.reshape(-1,3)).sum(axis=0)
        target=np.cross([*contact.center,HEIGHT],contact.force)
        return load,dict(quadrature_order=n,samples=len(weights),force_relative=float(np.linalg.norm(force-contact.force)/np.linalg.norm(contact.force)),moment_normalized=float(np.linalg.norm(moment-target)/(np.linalg.norm(contact.force)*contact.radius)),seconds=perf_counter()-start)

    def stress_map(self,points):
        """Sparse map to per-point [xx,yy,zz,xy,xz,yz], actual Cauchy components."""
        points=np.asarray(points,dtype=float)
        assert points.shape[0]==3
        elem=self.locate(points)
        local=np.einsum('pij,pj->pi',self.inverse[elem],points.T-self.origins[elem])
        bary=np.column_stack((1-local.sum(axis=1),local))
        grad=np.empty((len(elem),10,3))
        grad[:,:4]=(4*bary-1)[:,:,None]*self.bary_grad[elem]
        for k,(a,b) in enumerate(self.edges):
            grad[:,4+k]=4*(bary[:,a,None]*self.bary_grad[elem,b]+bary[:,b,None]*self.bary_grad[elem,a])
        dofs=self.scalar_basis.element_dofs[:,elem].T
        coeff=np.zeros((len(elem),6,10,3))
        for k,(i,j) in enumerate(((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))):
            if i==j:coeff[:,k]+=self.lam*grad
            coeff[:,k,:,i]+=self.mu*grad[:,:,j]
            coeff[:,k,:,j]+=self.mu*grad[:,:,i]
        rows=np.broadcast_to(np.arange(6*len(elem)).reshape(-1,6,1,1),coeff.shape).ravel()
        cols=np.broadcast_to((3*dofs[:,:,None]+np.arange(3))[...,None].transpose(0,3,1,2),coeff.shape).ravel()
        return sparse.coo_matrix((coeff.ravel(),(rows,cols)),shape=(6*len(elem),self.N)).tocsr()

    def storage(self):
        return dict(dofs=self.N,tets=self.mesh.nelements,vertices=self.mesh.nvertices,
                    stiffness_bytes=self.K.data.nbytes+self.K.indices.nbytes+self.K.indptr.nbytes,
                    lu_nonzeros=self.factor.L.nnz+self.factor.U.nnz,
                    approximate_lu_bytes=sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (self.factor.L,self.factor.U)),
                    assembly_seconds=self.assembly_seconds,factor_seconds=self.factor_seconds)


def tensor(values):
    s=np.asarray(values).reshape(-1,6)
    t=np.zeros((len(s),3,3));t[:,0,0]=s[:,0];t[:,1,1]=s[:,1];t[:,2,2]=s[:,2]
    t[:,0,1]=t[:,1,0]=s[:,3];t[:,0,2]=t[:,2,0]=s[:,4];t[:,1,2]=t[:,2,1]=s[:,5]
    return t


def vm(values):
    t=tensor(values);dev=t-np.trace(t,axis1=1,axis2=2)[:,None,None]/3*np.eye(3)
    return np.sqrt(1.5*np.sum(dev*dev,axis=(1,2)))
