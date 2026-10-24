"""CPU prototype: fixed-reference, linear MITC4 shell stress recovery.

The runtime problem is K q = f in the rigid link frame.  This module does
not integrate deformable dynamics or solve contact.  Quad nodes have global
translation and infinitesimal rotation (six unknowns); director increments
are theta cross director.  MITC edge-midpoint covariant shear interpolation
avoids the elementary full-integration Mindlin shear-locking mechanism.

This research implementation uses a bilinear reference surface, interpolated
nodal directors, and the first derivative through thickness of the physical
displacement gradient.  The inverse-reference-Jacobian derivative is included
so that all six rigid motions remain exact even for warped quads.  A weak
consistent drilling term removes unphysical independent director-spin modes.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu


def quad_shape(xi, eta):
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    n = .25 * (1 + signs[:, 0] * xi) * (1 + signs[:, 1] * eta)
    dn = .25 * np.column_stack((signs[:, 0] * (1 + signs[:, 1] * eta),
                                signs[:, 1] * (1 + signs[:, 0] * xi)))
    return n, dn


def _node_directors(vertices, quads):
    d = np.zeros_like(vertices)
    for ids in quads:
        x = vertices[ids]
        normal = np.cross(x[1] - x[0], x[2] - x[0]) + np.cross(x[2] - x[0], x[3] - x[0])
        d[ids] += normal
    norms = np.linalg.norm(d, axis=1)
    if np.any(norms < 1e-20):
        raise ValueError('Degenerate surface or inconsistent quad orientation.')
    return d / norms[:, None]


def _geometry(x, directors, xi, eta):
    n, dn = quad_shape(xi, eta)
    a = x.T @ dn
    normal = np.cross(a[:, 0], a[:, 1])
    jac = np.linalg.norm(normal)
    if jac <= 1e-20:
        raise ValueError('Degenerate quad Jacobian.')
    normal /= jac
    e1 = a[:, 0] / np.linalg.norm(a[:, 0])
    axes = np.column_stack((e1, np.cross(normal, e1), normal))
    d = n @ directors
    dd = directors.T @ dn
    a0 = np.column_stack((a, d))
    if np.linalg.det(a0) <= 0:
        raise ValueError('Director orientation inconsistent with quad geometry.')
    a1 = np.column_stack((dd, np.zeros(3)))
    return n, dn, a, d, a0, a1, axes, jac


def _kinematics(x, directors, xi, eta):
    n, dn, a, d, a0, a1, axes, jac = _geometry(x, directors, xi, eta)
    d0 = np.zeros((24, 3, 3))
    d1 = np.zeros_like(d0)
    for i in range(4):
        for j in range(3):
            d0[6*i+j, j, :2] = dn[i]
            delta = np.cross(np.eye(3)[j], directors[i])
            d0[6*i+3+j, :, 2] = n[i] * delta
            d1[6*i+3+j, :, :2] = delta[:, None] * dn[i][None, :]
    inverse = np.linalg.inv(a0)
    g0 = d0 @ inverse
    g1 = (d1 - g0 @ a1) @ inverse
    local0 = np.einsum('ai,nab,bj->nij', axes, g0, axes)
    local1 = np.einsum('ai,nab,bj->nij', axes, g1, axes)
    bm = np.array([local0[:, 0, 0], local0[:, 1, 1], local0[:, 0, 1]+local0[:, 1, 0]])
    bb = np.array([local1[:, 0, 0], local1[:, 1, 1], local1[:, 0, 1]+local1[:, 1, 0]])
    # Spin normal to the material surface agrees with continuum infinitesimal spin.
    bd = -.5 * (local0[:, 1, 0] - local0[:, 0, 1])
    for i in range(4):
        bd[6*i+3:6*i+6] += n[i] * axes[:, 2]
    return bm, bb, bd, axes, jac, (n, dn, a, d)


def _covariant_shear(x, directors, xi, eta):
    n, dn, a, d, *_ = _geometry(x, directors, xi, eta)
    bs = np.zeros((2, 24))
    for i in range(4):
        for j in range(3):
            bs[:, 6*i+j] = d[j] * dn[i]
            delta = n[i] * np.cross(np.eye(3)[j], directors[i])
            bs[:, 6*i+3+j] = a.T @ delta
    return bs


def element_operators(x, directors, xi, eta):
    bm, bb, bd, axes, jac, geom = _kinematics(x, directors, xi, eta)
    _, _, a, d = geom
    cov = np.empty((2, 24))
    cov[0] = .5*((1-eta)*_covariant_shear(x, directors, 0., -1.)[0]
                 +(1+eta)*_covariant_shear(x, directors, 0., 1.)[0])
    cov[1] = .5*((1-xi)*_covariant_shear(x, directors, -1., 0.)[1]
                 +(1+xi)*_covariant_shear(x, directors, 1., 0.)[1])
    # Transform E_{alpha,3} to local engineering transverse shears.  The
    # interpolated director need not be exactly normal to a warped surface.
    jtan = a.T @ axes[:, :2]
    dt = axes[:, :2].T @ d
    normal_director = axes[:, 2] @ d
    eps_dt = np.array([bm[0]*dt[0] + .5*bm[2]*dt[1],
                       .5*bm[2]*dt[0] + bm[1]*dt[1]])
    bs = np.linalg.solve(jtan, cov - 2*jtan @ eps_dt) / normal_director
    return bm, bb, bs, bd, axes, jac


@dataclass
class Factor:
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    lu: object
    seconds: float
    free_body: bool


class ShellModel:
    def __init__(self, vertices, quads, E=1e10, nu=.3, h=.0005,
                 density=1000., directors=None, drilling_factor=1e-3):
        start = time.perf_counter()
        self.vertices = np.asarray(vertices, dtype=float)
        self.quads = np.asarray(quads, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all():
            raise ValueError('Expected finite midsurface coordinates (N,3).')
        if self.quads.ndim != 2 or self.quads.shape[1] != 4:
            raise ValueError('Expected a four-node quadrilateral midsurface.')
        if not len(self.quads) or self.quads.min() < 0 or self.quads.max() >= len(self.vertices):
            raise ValueError('Invalid or empty quad connectivity.')
        if E <= 0 or h <= 0 or density <= 0 or not (-1 < nu < .5):
            raise ValueError('Require E,h,density > 0 and -1 < nu < 0.5.')
        if drilling_factor <= 0:
            raise ValueError('Require positive, explicitly declared drilling stabilization.')
        self.E, self.nu, self.h, self.density = float(E), float(nu), float(h), float(density)
        self.drilling_factor = float(drilling_factor)
        self.directors = (_node_directors(self.vertices, self.quads) if directors is None
                          else np.asarray(directors, dtype=float))
        self.directors = self.directors / np.linalg.norm(self.directors, axis=1)[:, None]
        self.ndof = 6*len(self.vertices)
        self.element_dofs = (6*self.quads[:, :, None] + np.arange(6)).reshape((-1, 24))
        self.C = E/(1-nu*nu)*np.array([[1.,nu,0.], [nu,1.,0.], [0.,0.,(1-nu)/2]])
        self.G = E/(2*(1+nu))
        self.A = h*self.C
        self.D = h**3/12*self.C
        self.S = 5./6*self.G*h*np.eye(2)
        ne = len(self.quads)
        self.B_membrane = np.empty((ne, 4, 3, 24))
        self.B_bending = np.empty_like(self.B_membrane)
        self.B_shear = np.empty((ne, 4, 2, 24))
        self.axes = np.empty((ne, 4, 3, 3))
        self.gauss_weights = np.empty((ne, 4))
        self.centers = self.vertices[self.quads].mean(axis=1)
        self.nodal_mass = np.zeros(len(self.vertices))
        g = 1/np.sqrt(3.)
        self.gauss_points = np.array([[-g,-g],[g,-g],[g,g],[-g,g]])
        blocks = np.zeros((ne,24,24))
        for e, ids in enumerate(self.quads):
            x, d = self.vertices[ids], self.directors[ids]
            for k, (xi, eta) in enumerate(self.gauss_points):
                bm, bb, bs, bd, axes, weight = element_operators(x,d,xi,eta)
                self.B_membrane[e,k], self.B_bending[e,k], self.B_shear[e,k] = bm,bb,bs
                self.axes[e,k], self.gauss_weights[e,k] = axes,weight
                blocks[e] += weight*(bm.T@self.A@bm + bb.T@self.D@bb + bs.T@self.S@bs
                                     + drilling_factor*self.G*h*np.outer(bd,bd))
                n, _ = quad_shape(xi,eta)
                self.nodal_mass[ids] += density*h*weight*n
        rows = np.broadcast_to(self.element_dofs[:,:,None],blocks.shape).ravel()
        cols = np.broadcast_to(self.element_dofs[:,None,:],blocks.shape).ravel()
        self.K = sparse.coo_matrix((blocks.ravel(),(rows,cols)),shape=(self.ndof,self.ndof)).tocsr()
        self.K = .5*(self.K+self.K.T)
        self.center_of_mass = np.average(self.vertices,axis=0,weights=self.nodal_mass)
        self.rigid_modes = np.zeros((self.ndof,6))
        for i,x in enumerate(self.vertices-self.center_of_mass):
            self.rigid_modes[6*i:6*i+3,:3] = np.eye(3)
            for j in range(3):
                self.rigid_modes[6*i:6*i+3,3+j] = np.cross(np.eye(3)[j],x)
                self.rigid_modes[6*i+3:6*i+6,3+j] = np.eye(3)[j]
        self.mass_diagonal = np.repeat(self.nodal_mass,6)
        self.mass_diagonal.reshape((-1,6))[:,3:] *= h*h/12
        # Director rotation has no independent rotary inertia around its own
        # director.  Retain the actual transverse 3x3 rotary blocks in relief;
        # do not give the numerical drilling stabilization a physical mass.
        self.rotary_mass = (self.nodal_mass*h*h/12)[:,None,None] * (
            np.eye(3)[None,:,:]-self.directors[:,:,None]*self.directors[:,None,:])
        self.mass_diagonal.reshape((-1,6))[:,3:] = np.diagonal(self.rotary_mass,axis1=1,axis2=2)
        self.mass_modes = self.mass_diagonal[:,None]*self.rigid_modes
        for i in range(len(self.vertices)):
            self.mass_modes[6*i+3:6*i+6] = self.rotary_mass[i] @ self.rigid_modes[6*i+3:6*i+6]
        self.mass_gram = self.rigid_modes.T@self.mass_modes
        _, _, pivots = linalg.qr(self.rigid_modes.T, pivoting=True, mode='economic')
        self.gauge_dofs = np.sort(pivots[:6])
        self._factors = {}
        self.last_relief = None
        self.assembly_seconds = time.perf_counter()-start

    def factorize(self, fixed_dofs=None):
        free_body = fixed_dofs is None
        fixed = self.gauge_dofs if free_body else np.unique(np.asarray(fixed_dofs,dtype=int))
        key = (free_body,tuple(fixed))
        if key not in self._factors:
            start = time.perf_counter()
            free = np.setdiff1d(np.arange(self.ndof),fixed)
            lu = splu(self.K[free][:,free].tocsc(),permc_spec='MMD_AT_PLUS_A')
            self._factors[key] = Factor(free,fixed,lu,time.perf_counter()-start,free_body)
        return self._factors[key]

    def balance_load(self, force):
        f = np.asarray(force,dtype=float)
        coeff = np.linalg.solve(self.mass_gram,self.rigid_modes.T@f)
        return f-self.mass_modes@coeff, coeff

    def solve(self, force, fixed_dofs=None, balance=True):
        f = np.asarray(force,dtype=float)
        if f.shape[0] != self.ndof:
            raise ValueError('Load leading dimension must be ndof.')
        factor = self.factorize(fixed_dofs)
        if factor.free_body and balance:
            f, self.last_relief = self.balance_load(f)
        else:
            self.last_relief = np.zeros((6,)+f.shape[1:])
        q = np.zeros_like(f)
        q[factor.free_dofs] = factor.lu.solve(np.asfortranarray(f[factor.free_dofs]))
        return q

    def stress(self, q):
        """Stress at four in-plane points, top/bottom; tensor axes are global.

        For batched q(ndof,B), B is the final axis of every returned array.
        N/M are membrane force and bending moment per unit length, in each
        quadrature point's local tangent basis.  Q is transverse shear resultant.
        Surface tensors returned here contain in-plane membrane+bending stress;
        they do NOT claim to include 3D Hertz near-contact normal stress.
        """
        q = np.asarray(q,dtype=float)
        u = q[self.element_dofs]
        eps = np.einsum('egij,ej...->egi...',self.B_membrane,u)
        kap = np.einsum('egij,ej...->egi...',self.B_bending,u)
        shear = np.einsum('egij,ej...->egi...',self.B_shear,u)
        n = np.einsum('ij,egj...->egi...',self.A,eps)
        m = np.einsum('ij,egj...->egi...',self.D,kap)
        qq = np.einsum('ij,egj...->egi...',self.S,shear)
        top = n/self.h + 6*m/self.h**2
        bottom = n/self.h - 6*m/self.h**2
        e1,e2 = self.axes[:,:,:,0],self.axes[:,:,:,1]
        tensor_basis = np.stack((e1[:,:,:,None]*e1[:,:,None,:],
                                 e2[:,:,:,None]*e2[:,:,None,:],
                                 e1[:,:,:,None]*e2[:,:,None,:]+e2[:,:,:,None]*e1[:,:,None,:]),axis=2)
        sigma_top = np.einsum('egkij,egk...->egij...',tensor_basis,top)
        sigma_bottom = np.einsum('egkij,egk...->egij...',tensor_basis,bottom)
        return dict(N=n,M=m,Q=qq,epsilon=eps,kappa=kap,local_top=top,local_bottom=bottom,
                    sigma_top=sigma_top,sigma_bottom=sigma_bottom)

    def stress_matrix(self,kind='local'):
        """Sparse exact operator q -> stacked top/bottom in-plane stresses."""
        if kind!='local':
            raise ValueError('Only local in-plane stress map is currently exposed.')
        maps = np.stack((np.einsum('ij,egjk->egik',self.C,self.B_membrane+.5*self.h*self.B_bending),
                         np.einsum('ij,egjk->egik',self.C,self.B_membrane-.5*self.h*self.B_bending)))
        nrow = maps.shape[0]*maps.shape[1]*maps.shape[2]*maps.shape[3]
        row = np.repeat(np.arange(nrow),24)
        col = np.broadcast_to(self.element_dofs[None,:,None,None,:],maps.shape).ravel()
        return sparse.coo_matrix((maps.ravel(),(row,col)),shape=(nrow,self.ndof)).tocsr()

    def residual(self,q,force,fixed_dofs=None,balance=True):
        f=np.asarray(force,dtype=float)
        if fixed_dofs is None and balance:
            f,_=self.balance_load(f)
        r=self.K@q-f
        if fixed_dofs is not None:
            r[np.asarray(fixed_dofs,dtype=int)]=0
        return np.linalg.norm(r)/max(np.linalg.norm(f),1e-30)

