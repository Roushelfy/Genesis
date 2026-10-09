"""Fixed-geometry spatial indexing for arbitrary moving finite contact patches.

Only surface quadrature geometry is cached. Centers, sizes, force directions
and the number of contacts are runtime inputs; no stress/load basis is used.
Positions and vectors use the cached reference geometry's material frame.
"""
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree


class IndexedContacts:
    def __init__(self,geometry):
        start=perf_counter();self.geometry=geometry
        self.coords=np.ascontiguousarray(geometry.coords.reshape(-1,3))
        self.weights=np.ascontiguousarray(geometry.integration_weights.ravel())
        self.tree=cKDTree(self.coords)
        self.nq=geometry.shape.shape[0]
        self.metadata={'construction_s':perf_counter()-start,'quadrature_points':len(self.coords),
            'coordinates_bytes':self.coords.nbytes,'integration_weights_bytes':self.weights.nbytes,
            'scope':'All finite patches on stored surface geometry; centers, radius and force change each query'}

    def load(self,patches):
        """Map supplied smooth patches, conserving their specified resultant.

        Torque is the integrated pressure centroid crossed with the force;
        it is not generally center_m crossed with force_N. The profile is a
        test/load-model choice, not a consequence of rigid contact output.
        """
        g=self.geometry;raw=np.zeros((len(g.f.xyz),3))
        for patch in patches:
            center=np.asarray(patch['center_m'],float)
            radius=float(patch['radius_m']);sigma=float(patch['sigma_m'])
            F=np.asarray(patch['force_N'],float)
            if center.shape!=(3,) or F.shape!=(3,) or radius<=0 or sigma<=0 or not np.isfinite(np.r_[center,F,radius,sigma]).all():
                raise ValueError('finite center/force and positive finite radius/sigma required')
            ids=np.asarray(self.tree.query_ball_point(center,radius,return_sorted=True),np.int64)
            if not len(ids):raise ValueError('No quadrature sample lies in this patch; refine integration')
            distance2=np.sum((self.coords[ids]-center)**2,axis=1)
            w=np.exp(-.5*distance2/sigma**2)*np.maximum(0.,1-distance2/radius**2)**2*self.weights[ids]
            integral=w.sum()
            if integral<=0:raise ValueError('Contact support is not resolved by surface quadrature')
            w/=integral;faces=ids//self.nq;qs=ids%self.nq
            node_weights=w[:,None]*g.shape[qs]
            contribution=(node_weights[:,:,None]*F).reshape(-1,3)
            np.add.at(raw,g.nodes[faces].ravel(),contribution)
        return raw.ravel()

    def load_tractions(self,traction_N_per_m2,quadrature_ids=None):
        """Map arbitrary vector traction samples; no Gaussian pressure law.

        Supply every stored outer-surface quadrature point, or a sparse subset.
        Positions/vectors use the material frame; duplicate IDs add together.
        Integration accuracy remains the caller's discretization responsibility.
        """
        g=self.geometry
        ids=np.arange(len(self.coords)) if quadrature_ids is None else np.asarray(quadrature_ids,np.int64)
        traction=np.asarray(traction_N_per_m2,np.float64)
        if ids.ndim!=1 or traction.shape!=(len(ids),3) or not np.isfinite(traction).all() or np.any(ids<0) or np.any(ids>=len(self.coords)):
            raise ValueError('valid quadrature IDs and finite [samples,3] tractions required')
        raw=np.zeros((len(g.f.xyz),3))
        nodes=g.nodes[ids//self.nq]
        weights=self.weights[ids,None]*g.shape[ids%self.nq]
        contribution=(weights[:,:,None]*traction[:,None,:]).reshape(-1,3)
        np.add.at(raw,nodes.ravel(),contribution)
        return raw.ravel()
