#!/usr/bin/env python3
"""Separate recovery error, load quadrature error and solid-shell FEM error.

Run next to eggshell_gripper_cpu.py, rigid_stress_reference.py and
rigid_stress_temporal_cpu_test.py. NumPy/SciPy/threadpoolctl are required;
--plot additionally uses matplotlib. Examples:
  python eggshell_convergence_cpu.py --jobs 1:3:4 1:4:4 2:3:1 2:4:1 --plot
  python eggshell_convergence_cpu.py --jobs 1:5:2 2:5:1 --append --plot

Job format is interpolation order : surface refinement level : wall layers.
Prefix the order with q for the central-load quarter-symmetry diagnostic;
an optional fourth integer adds local pad-area surface refinement passes.
For example: --jobs q2:5:1 q2:6:1 q2:6:1:1 q2:6:1:2 q2:6:2.
P2 uses ten displacement nodes and straight-sided tetrahedra, so geometry is
identical to P1 for a given level/layers. Stiffness uses exact degree-two
quadrature, consistent mass is integrated analytically. Stress is evaluated
at all four element vertices: its affine P2 stress field makes this the exact
elementwise maximum of the convex von Mises norm (no nodal averaging).
Every result is checkpointed. The force is always 4 N on each finite pad.
The equilibrium residual is scaled by max(||rhs||, 1 N). The volume-weighted
RMS element-peak diagnostic is not the RMS of the pointwise stress field.
"""
from pathlib import Path
from time import perf_counter
import argparse
import csv
import gc
import itertools
import json
import math
import os
import resource
import sys

import numpy as np
from scipy import linalg,sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import eggshell_gripper_cpu as egg
import rigid_stress_reference as ref

EDGES=np.array(list(itertools.combinations(range(4),2)))
TET_QUAD=np.full((4,4),(5-np.sqrt(5))/20)
np.fill_diagonal(TET_QUAD,(5+3*np.sqrt(5))/20)


def mass_template(order):
    terms=[]
    for i in range(4):
        a=np.zeros(4,dtype=int);a[i]=1
        if order==1:terms.append([(1.,a)])
        else:terms.append([(2.,2*a),(-1.,a)])
    if order==2:
        for i,j in EDGES:
            a=np.zeros(4,dtype=int);a[i]=a[j]=1;terms.append([(4.,a)])
    out=np.zeros((len(terms),len(terms)))
    for i,p in enumerate(terms):
        for j,q in enumerate(terms):
            for a,alpha in p:
                for b,beta in q:
                    exponent=alpha+beta
                    integral=6*math.prod(math.factorial(int(k)) for k in exponent)/math.factorial(3+int(exponent.sum()))
                    out[i,j]+=a*b*integral
    assert abs(out.sum()-1)<1e-13
    assert np.linalg.eigvalsh(out).min()>0
    return out


def shape_gradients(grad_lambda,bary,order):
    if order==1:return np.broadcast_to(grad_lambda[:,None],(len(grad_lambda),len(bary),4,3))
    vertex=(4*bary[None,:,:,None]-1)*grad_lambda[:,None,:,:]
    edge=4*(bary[None,:,EDGES[:,0],None]*grad_lambda[:,None,EDGES[:,1],:]+
            bary[None,:,EDGES[:,1],None]*grad_lambda[:,None,EDGES[:,0],:])
    return np.concatenate([vertex,edge],axis=2)


class FEM:
    def __init__(self,xyz,tets,outer,order=1,young=1e10,poisson=.3,density=2000.,label='',quarter=False):
        start=perf_counter();self.order=order;self.young=young;self.poisson=poisson
        self.base_xyz=xyz;self.base_tets=tets;self.outer_faces=outer
        nbase=len(xyz);self.nbase=nbase;self.quarter=quarter
        if order==2:
            edge=np.sort(tets[:,EDGES].reshape(-1,2),axis=1)
            unique,inverse=np.unique(edge,axis=0,return_inverse=True)
            self.edge_keys=unique[:,0]*nbase+unique[:,1]
            self.xyz=np.concatenate([xyz,xyz[unique].mean(axis=1)])
            self.elements=np.column_stack([tets,nbase+inverse.reshape(-1,6)])
        else:self.xyz=xyz;self.elements=tets
        self.nloc=self.elements.shape[1];self.ndof=3*len(self.xyz);self.ne=len(tets)
        self.dofs=(3*self.elements[:,:,None]+np.arange(3)).reshape(self.ne,-1)
        points=xyz[tets];matrix=np.concatenate([np.ones((self.ne,4,1)),points],axis=2)
        self.glambda=np.linalg.inv(matrix)[:,1:,:].transpose(0,2,1)
        self.volume=np.abs(np.linalg.det(points[:,1:]-points[:,:1]))/6
        self.d=ref.isotropic_elasticity(young,poisson)
        lam=young*poisson/((1+poisson)*(1-2*poisson));mu=young/(2*(1+poisson))
        nl=self.nloc*3;values=np.empty((self.ne,nl,nl))
        bary=np.full((1,4),.25) if order==1 else TET_QUAD
        for lo in range(0,self.ne,512):
            hi=min(self.ne,lo+512);g=shape_gradients(self.glambda[lo:hi],bary,order)
            a=lam*np.einsum('eqia,eqjb->eiajb',g,g,optimize=True)
            a+=mu*np.einsum('eqib,eqja->eiajb',g,g,optimize=True)
            dot=np.einsum('eqic,eqjc->eij',g,g,optimize=True)
            for c in range(3):a[:,:,c,:,c]+=mu*dot
            values[lo:hi]=a.reshape(-1,nl,nl)*self.volume[lo:hi,None,None]/len(bary)
        row=np.repeat(self.dofs,nl,axis=1).astype(np.int32)
        col=np.tile(self.dofs,(1,nl)).astype(np.int32)
        self.k=sparse.coo_matrix((values.ravel(),(row.ravel(),col.ravel())),shape=(self.ndof,self.ndof)).tocsc()
        self.k.eliminate_zeros();del row,col,values,points,matrix;gc.collect()
        ms=mass_template(order);svalues=(density*self.volume[:,None,None]*ms).reshape(-1)
        rn=np.repeat(self.elements,self.nloc,axis=1).ravel();cn=np.tile(self.elements,(1,self.nloc)).ravel()
        scalar=sparse.coo_matrix((svalues,(rn,cn)),shape=(len(self.xyz),len(self.xyz))).tocsc()
        self.m=sparse.kron(scalar,sparse.eye(3),format='csc')
        self.m.eliminate_zeros();del scalar,rn,cn,svalues;gc.collect()
        nodal_mass=np.asarray(self.m.sum(axis=1)).ravel()[::3]
        self.mass=float(nodal_mass.sum());self.com=nodal_mass@self.xyz/self.mass
        self.r=np.empty((self.ndof,6))
        for i,p in enumerate(self.xyz-self.com):self.r[3*i:3*i+3]=np.column_stack([np.eye(3),-ref.skew(p)])
        self.mr=self.m@self.r;self.gram=self.r.T@self.mr;self.gfactor=linalg.cho_factor(self.gram,lower=True)
        if quarter:
            self.symmetry=np.r_[3*np.flatnonzero(np.abs(self.xyz[:,0])<1e-13),
                                3*np.flatnonzero(np.abs(self.xyz[:,1])<1e-13)+1]
            self.zpin=3*int(np.argmin(self.xyz[:,2]))+2
            self.pins=np.r_[self.symmetry,self.zpin]
        else:self.pins=ref.independent_gauge_rows(self.r)
        self.free=np.setdiff1d(np.arange(self.ndof),self.pins)
        self.assembly_s=perf_counter()-start
        print(json.dumps({'phase':'factor','model':label,'nodes':len(self.xyz),'elements':self.ne,
                          'K_nnz':self.k.nnz,'rss_MB':rss_mb()}),flush=True)
        start=perf_counter()
        self.factor=splu(self.k[self.free][:,self.free],permc_spec='MMD_AT_PLUS_A',
                        diag_pivot_thresh=0.,options={'SymmetricMode':True})
        self.factor_s=perf_counter()-start

    def balance(self,raw):
        if self.quarter:return raw.copy()
        return raw-self.mr@linalg.cho_solve(self.gfactor,self.r.T@raw)

    def canonical(self,u):
        if self.quarter:
            z=self.r[:,2];return u-z*float(z@(self.m@u)/(z@(self.m@z)))
        return u-self.r@linalg.cho_solve(self.gfactor,self.r.T@(self.m@u))

    def solve(self,raw,alternate_gauge=False):
        rhs=self.balance(raw);u=np.zeros(self.ndof)
        if alternate_gauge:
            pins=np.r_[self.symmetry,3*int(np.argmax(self.xyz[:,2]))+2] if self.quarter else ref.independent_gauge_rows(self.r,np.arange(self.ndof)[::-1])
            free=np.setdiff1d(np.arange(self.ndof),pins)
            factor=splu(self.k[free][:,free],permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        else:factor=self.factor;free=self.free
        u[free]=factor.solve(rhs[free])
        difference=self.k@u-rhs
        residual=float(np.linalg.norm(difference[free] if self.quarter else difference)/max(np.linalg.norm(rhs),1.))
        if self.quarter:assert abs(difference[pins[-1] if alternate_gauge else self.zpin])<1e-7
        assert residual<1e-7,residual
        return u,rhs,residual

    def strain(self,u,bary):
        for lo in range(0,self.ne,1024):
            hi=min(self.ne,lo+1024);g=shape_gradients(self.glambda[lo:hi],bary,self.order)
            v=u[self.dofs[lo:hi]].reshape(-1,self.nloc,3)
            du=np.einsum('eia,eqib->eqab',v,g,optimize=True)
            eps=np.stack([du[:,:,0,0],du[:,:,1,1],du[:,:,2,2],
                          du[:,:,0,1]+du[:,:,1,0],du[:,:,1,2]+du[:,:,2,1],
                          du[:,:,0,2]+du[:,:,2,0]],axis=-1)
            yield lo,hi,eps

    def metrics(self,u,raw,rhs,residual):
        peak=0.;hot=None;hot_element=None
        bary=np.full((1,4),.25) if self.order==1 else np.eye(4)
        vm_all=[]
        for lo,hi,eps in self.strain(u,bary):
            stress=eps@self.d.T;vm=ref.von_mises(stress)
            vm_all.append(vm.max(axis=1));e,q=np.unravel_index(vm.argmax(),vm.shape)
            if vm[e,q]>peak:
                peak=float(vm[e,q]);hot_element=int(lo+e)
                hot=bary[q]@self.base_xyz[self.base_tets[lo+e]]
        canonical=self.canonical(u).reshape(-1,3)
        compliance=float(rhs@u)
        integrated=0.;quad=np.full((1,4),.25) if self.order==1 else TET_QUAD
        for lo,hi,eps in self.strain(u,quad):
            stress=eps@self.d.T
            integrated+=float(np.sum(np.einsum('eqi,eqi->eq',stress,eps)*self.volume[lo:hi,None]/len(quad)))
        energy_error=abs(integrated-compliance)/max(abs(compliance),1e-30)
        assert energy_error<1e-7,energy_error
        return {'max_von_Mises_Pa':peak,'hotspot_m':hot.tolist(),'hotspot_element':hot_element,
                'compliance_Nm':compliance,'strain_energy_J':compliance/2,
                'max_canonical_displacement_m':float(np.linalg.norm(canonical,axis=1).max()),
                'scaled_equilibrium_residual':residual,'energy_consistency_error':energy_error,
                'volume_weighted_RMS_element_peak_Pa':float(np.sqrt(np.average(np.concatenate(vm_all)**2,weights=self.volume)))}


def rss_mb():
    p=Path('/proc/self/statm')
    return int(p.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/1e6 if p.exists() else None


def triangle_rule(n):
    if n==0:return np.full((1,3),1/3),np.ones(1)
    x,w=np.polynomial.legendre.leggauss(n);x=(x+1)/2;w=w/2
    u,v=np.meshgrid(x,x,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij')
    return np.column_stack([1-u.ravel(),(u*(1-v)).ravel(),(u*v).ravel()]),(2*u*wu*wv).ravel()


def grip_load(fem,quad=6,force=4.,radius=.005,sigma=.002,z=0.):
    bary,weights=triangle_rule(quad);raw=np.zeros((len(fem.xyz),3));metadata=[]
    faces=fem.outer_faces
    if fem.order==1:nodes=faces;shape=bary
    else:
        e=np.sort(faces[:,[[0,1],[0,2],[1,2]]],axis=2)
        key=e[:,:,0]*fem.nbase+e[:,:,1]
        mids=fem.nbase+np.searchsorted(fem.edge_keys,key)
        nodes=np.column_stack([faces,mids])
        shape=np.column_stack([bary*(2*bary-1),4*bary[:,0]*bary[:,1],
                               4*bary[:,0]*bary[:,2],4*bary[:,1]*bary[:,2]])
    for side in ([1] if fem.quarter else [-1,1]):
        nodal=np.zeros(len(fem.xyz));integral=0.;moment=0.;center=np.zeros(3)
        for lo in range(0,len(faces),512):
            hi=min(len(faces),lo+512);p=fem.base_xyz[faces[lo:hi]]
            area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2
            q=np.einsum('qi,eic->eqc',bary,p,optimize=True)
            d2=q[:,:,1]**2+(q[:,:,2]-z)**2
            pressure=np.exp(-.5*d2/sigma**2)*np.maximum(0,1-d2/radius**2)**2
            pressure*=side*p.mean(axis=1)[:,0,None]>0
            weight=pressure*area[:,None]*weights[None,:]
            element_load=weight@shape
            for a in range(nodes.shape[1]):np.add.at(nodal,nodes[lo:hi,a],element_load[:,a])
            integral+=float(weight.sum());moment+=float((weight*d2).sum())
            center+=np.einsum('eq,eqc->c',weight,q)
        assert integral>0
        raw[:,0]+=-side*force*(.5 if fem.quarter else 1.)*nodal/integral
        metadata.append({'side':side,'pressure_integral_m2':integral,
                         'continuous_pressure_RMS_radius_m':np.sqrt(moment/integral),
                         'force_centroid_m':(center/integral).tolist(),
                         'nodal_signed_second_moment_m2':float(nodal@(fem.xyz[:,1]**2+(fem.xyz[:,2]-z)**2)/integral)})
    return raw.ravel(),metadata


def element_checks():
    checks=[]
    # Pure bending polynomial: sigma_xx = -E*kappa*z, other stresses zero.
    # P2 represents this exactly; P1 introduces spurious bending strain energy.
    xyz,tet=ref.structured_cube(3);xyz*=np.array([.020,.006,.0005]);xyz[:,2]-=.00025
    outer=ref.boundary_triangles(tet)
    for order in [1,2]:
        f=FEM(xyz,tet,outer,order,label=f'patch-P{order}')
        x,y,z=f.xyz.T;nu=f.poisson;kappa=1.
        u=np.column_stack([-kappa*x*z,nu*kappa*y*z,.5*kappa*(x*x+nu*(z*z-y*y))]).ravel()
        energy=float(u@(f.k@u));exact=f.young*kappa*kappa*.020*.006*(.0005**3)/12
        squared=0.;denom=0.;shear=0.
        for lo,hi,eps in f.strain(u,TET_QUAD):
            s=eps@f.d.T
            q=np.einsum('qi,eic->eqc',TET_QUAD,xyz[tet[lo:hi]])
            true=np.zeros_like(s);true[:,:,0]=-f.young*kappa*q[:,:,2]
            squared+=float(np.sum((s-true)**2*f.volume[lo:hi,None,None]/4))
            denom+=float(np.sum(true**2*f.volume[lo:hi,None,None]/4))
            shear+=float(np.sum(s[:,:,3:]**2*f.volume[lo:hi,None,None]/4))
        # An affine field should have exactly constant prescribed strain.
        G=np.array([[.002,.003,-.001],[.0005,-.001,.002],[.004,.001,.003]])
        affine=(f.xyz@G.T).ravel()
        e=np.array([G[0,0],G[1,1],G[2,2],G[0,1]+G[1,0],G[1,2]+G[2,1],G[0,2]+G[2,0]])
        err=max(float(np.max(np.abs(eps-e))) for _,_,eps in f.strain(affine,TET_QUAD))
        rigid=f.r@np.array([.01,-.03,.02,.3,-.1,.2])
        rigid_error=np.linalg.norm(f.k@rigid)/max(np.linalg.norm(f.k.data)*np.linalg.norm(rigid),1e-30)
        assert err<1e-12 and rigid_error<1e-12
        if order==2:assert abs(energy/exact-1)<1e-8 and np.sqrt(squared/denom)<1e-9
        checks.append({'order':order,'bending_energy_ratio_to_exact':energy/exact,
                       'relative_RMS_bending_stress_error':np.sqrt(squared/denom),
                       'spurious_shear_RMS_relative':np.sqrt(shear/denom),
                       'affine_strain_max_absolute_error':err,'rigid_stiffness_error':rigid_error})
        del f;gc.collect()
    # Same P1 model assembled through the original B^T D B and vector form.
    xyz,tets,outer,_=egg.egg_mesh(1,1)
    f=FEM(xyz,tets,outer,1,label='assembly-cross-check')
    old=ref.CachedElasticRecovery(xyz,tets,young=1e10,poisson=.3,density=2000.)
    difference=f.k-old.k
    kerr=np.linalg.norm(difference.data)/np.linalg.norm(old.k.data)
    difference=f.m-old.m;merr=np.linalg.norm(difference.data)/np.linalg.norm(old.m.data)
    assert kerr<1e-12 and merr<1e-12
    return {'patch_tests':checks,'P1_stiffness_relative_difference_from_original':kerr,
            'P1_mass_relative_difference_from_original':merr}


def refine_sphere(unit,faces,selected):
    marked=set()
    for f in faces[selected]:
        for a,b in zip(f,np.roll(f,-1)):marked.add(tuple(sorted((int(a),int(b)))))
    points=unit.tolist();mid={}
    for a,b in sorted(marked):
        p=unit[a]+unit[b];p/=np.linalg.norm(p);mid[(a,b)]=len(points);points.append(p.tolist())
    result=[]
    for f in faces:
        edges=[tuple(sorted((int(a),int(b)))) for a,b in zip(f,np.roll(f,-1))]
        count=sum(e in mid for e in edges)
        if count==0:result.append(f.tolist())
        elif count==3:
            a,b,c=f;ab,bc,ca=[mid[e] for e in edges]
            result.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
        elif count==1:
            i=next(i for i,e in enumerate(edges) if e in mid)
            a,b,c=np.roll(f,-i);m=mid[edges[i]];result.extend([[a,m,c],[m,b,c]])
        else:
            missing=next(i for i,e in enumerate(edges) if e not in mid)
            a,b,c=np.roll(f,-((missing+1)%3))
            ab=mid[tuple(sorted((int(a),int(b))))];bc=mid[tuple(sorted((int(b),int(c))))]
            result.append([ab,b,bc])
            pp=np.asarray(points)
            if np.linalg.norm(pp[c]-pp[ab])<=np.linalg.norm(pp[a]-pp[bc]):result.extend([[a,ab,c],[ab,bc,c]])
            else:result.extend([[a,ab,bc],[a,bc,c]])
    return np.asarray(points),np.asarray(result,dtype=int)


def quarter_mesh(level,layers,adaptive=0):
    unit=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[0,0,-1.]])
    faces=np.array([[0,1,2],[0,3,1]])
    for _ in range(level):unit,faces=refine_sphere(unit,faces,np.ones(len(faces),dtype=bool))
    for _ in range(adaptive):
        c=unit[faces].mean(axis=1);scale=1-.18*c[:,2]
        d=np.sqrt((.022*scale*c[:,1])**2+(.030*c[:,2])**2)
        unit,faces=refine_sphere(unit,faces,d<.012)
    scale=1-.18*unit[:,2]
    outer=unit*np.column_stack([.022*scale,.022*scale,np.full(len(unit),.030)])
    gradient=np.column_stack([2*unit[:,0]/(.022*scale),2*unit[:,1]/(.022*scale),
                              2/.030*(unit[:,2]+.18*(unit[:,0]**2+unit[:,1]**2)/scale)])
    normal=gradient/np.linalg.norm(gradient,axis=1)[:,None]
    n=len(unit);xyz=np.concatenate([outer-depth*normal for depth in np.linspace(0,.0005,layers+1)])
    tets=[]
    for layer in range(layers):
        for a,b,c in np.sort(faces,axis=1):
            A,B,C=np.array([a,b,c])+layer*n;D,E,F=np.array([a,b,c])+(layer+1)*n
            tets.extend([[A,B,C,F],[A,B,E,F],[A,D,E,F]])
    return xyz,np.asarray(tets,dtype=int),faces,{'surface_refinement':level,'through_thickness_layers':layers,
                                             'thickness_m':.0005,'adaptive_refinements':adaptive,
                                             'local_refinement_radius_m':.012,'mesh_family':'octahedral quarter shell'}


def symmetry_check():
    xyz,tet,faces,meta=quarter_mesh(2,1)
    q=FEM(xyz,tet,faces,2,label='quarter-symmetry-check',quarter=True)
    raw,_=grip_load(q,10);u,rhs,res=q.solve(raw)
    qm=q.metrics(u,raw,rhs,res)['max_von_Mises_Pa'];qc=float(rhs@u)
    signs=[(1,1),(-1,1),(1,-1),(-1,-1)]
    points=np.concatenate([xyz*np.array([sx,sy,1]) for sx,sy in signs])
    full,inv=np.unique(np.round(points,15),axis=0,return_inverse=True)
    inverse=inv.reshape(4,len(xyz))
    full_tet=np.concatenate([p[tet] for p in inverse]);full_faces=np.concatenate([p[faces] for p in inverse])
    f=FEM(full,full_tet,full_faces,2,label='full-symmetry-check')
    raw,_=grip_load(f,10);u,rhs,res=f.solve(raw);fm=f.metrics(u,raw,rhs,res)['max_von_Mises_Pa']
    error=abs(fm-qm)/fm;compliance=abs(float(rhs@u)-4*qc)/float(rhs@u)
    assert error<1e-7 and compliance<1e-7,(error,compliance)
    return {'full_vs_quarter_peak_relative_error':error,'full_vs_4x_quarter_compliance_relative_error':compliance}


def run_job(order,level,layers,gauge=False,quarter=False,adaptive=0):
    label=f"{'Q' if quarter else ''}P{order}-L{level}-T{layers}"+(f'-A{adaptive}' if adaptive else '')
    xyz,tets,outer,meta=quarter_mesh(level,layers,adaptive) if quarter else egg.egg_mesh(level,layers)
    f=FEM(xyz,tets,outer,order,label=label,quarter=quarter);results={}
    for quad in [0,3,6,10]:
        raw,loadmeta=grip_load(f,quad);u,rhs,res=f.solve(raw)
        metrics=f.metrics(u,raw,rhs,res)
        results[str(quad)]={**metrics,'pressure':loadmeta}
        print(json.dumps({'model':label,'quadrature':quad,'peak_MPa':metrics['max_von_Mises_Pa']/1e6,
                          'residual':res,'compliance_Nm':metrics['compliance_Nm']}),flush=True)
        if gauge and quad==6:
            other,_,_=f.solve(raw,True)
            gauge_error=np.linalg.norm(f.canonical(other)-f.canonical(u))/max(np.linalg.norm(f.canonical(u)),1e-30)
            results[str(quad)]['alternate_gauge_displacement_relative_error']=gauge_error
            assert gauge_error<1e-7,gauge_error
    edge=np.concatenate([xyz[outer[:,0]]-xyz[outer[:,1]],xyz[outer[:,1]]-xyz[outer[:,2]],xyz[outer[:,2]]-xyz[outer[:,0]]])
    result={'label':label,'order':order,'surface_level':level,'wall_layers':layers,
            'family':'octa_quarter' if quarter else 'icosphere','adaptive_refinements':adaptive,
            'nodes':len(f.xyz),'elements':f.ne,'mean_surface_edge_mm':float(np.linalg.norm(edge,axis=1).mean()*1000),
            'minimum_surface_edge_mm':float(np.linalg.norm(edge,axis=1).min()*1000),
            'assembly_s':f.assembly_s,'factor_s':f.factor_s,'LU_nnz':f.factor.nnz,
            'rss_MB':rss_mb(),'max_rss_MB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'geometry':meta,'results':results}
    del f;gc.collect();return result


def plot(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    colors={1:'#2474a6',2:'#14806e'};styles={1:':',2:'--',4:'-'}
    for order in [1,2]:
        for layers in [1,2,4]:
            jobs=sorted([j for j in data['jobs'] if j['order']==order and j['wall_layers']==layers and j.get('family','icosphere')=='icosphere'],key=lambda j:j['surface_level'])
            if not jobs:continue
            x=[j['mean_surface_edge_mm'] for j in jobs]
            y=[j['results']['6']['max_von_Mises_Pa']/1e6 for j in jobs]
            compliance=[j['results']['6']['compliance_Nm']*1e6 for j in jobs]
            label=f'P{order}, {layers} wall layer(s)'
            axs[0,0].plot(x,y,'o',color=colors[order],ls=styles[layers],label=label)
            axs[0,1].plot(x,compliance,'o',color=colors[order],ls=styles[layers],label=label)
    for ax in axs[0]:ax.set_xscale('log');ax.invert_xaxis();ax.grid(alpha=.2);ax.set_xlabel('Mean surface edge h (mm; refinement to right)')
    axs[0,0].set_ylabel('Maximum von Mises (MPa)');axs[0,1].set_ylabel('Compliance (micro N m)')
    axs[0,0].set_title('Peak stress: smooth pad quadrature');axs[0,1].set_title('Global deformation / energy')
    axs[0,0].legend(fontsize=8,frameon=False)
    for order in [1,2]:
        jobs=sorted([j for j in data['jobs'] if j['order']==order and j['wall_layers']==1 and j.get('family','icosphere')=='icosphere'],key=lambda j:j['surface_level'])
        if jobs:
            x=[j['mean_surface_edge_mm'] for j in jobs]
            error=[100*abs(j['results']['0']['max_von_Mises_Pa']-j['results']['6']['max_von_Mises_Pa'])/j['results']['6']['max_von_Mises_Pa'] for j in jobs]
            axs[1,0].plot(x,error,'o-',color=colors[order],label=f'P{order}')
    axs[1,0].set_xscale('log');axs[1,0].invert_xaxis();axs[1,0].set_xlabel('Mean surface edge h (mm)')
    axs[1,0].set_ylabel('Centroid-load peak error vs high quadrature (%)');axs[1,0].set_title('Pressure discretization effect');axs[1,0].grid(alpha=.2);axs[1,0].legend(frameon=False)
    tests=data['checks']['patch_tests']
    axs[1,1].bar(['P1','P2'],[p['bending_energy_ratio_to_exact'] for p in tests],color=[colors[1],colors[2]])
    axs[1,1].axhline(1,color='#777777',ls='--');axs[1,1].set_yscale('log')
    axs[1,1].set_ylabel('Interpolated pure-bending energy / exact energy')
    axs[1,1].set_title('Why linear tetrahedra are too stiff in bending')
    for i,p in enumerate(tests):axs[1,1].text(i,p['bending_energy_ratio_to_exact']*1.05,f"{p['bending_energy_ratio_to_exact']:.3g}x",ha='center')
    axs[1,1].grid(axis='y',alpha=.2);axs[1,1].set_axisbelow(True)
    fig.suptitle('Hollow egg: convergence diagnosis | 4 N per finger | 0.5 mm wall\nP1 / P2 share the same straight-sided geometry at each mesh level',fontsize=13)
    fig.savefig(out/'eggshell_convergence_cpu.png',dpi=180);plt.close(fig)
    quarter=[j for j in data['jobs'] if j.get('family')=='octa_quarter']
    if quarter:
        fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        from matplotlib.ticker import FixedLocator,FuncFormatter
        for order in [1,2]:
            for layers in [1,2,4]:
                jobs=sorted([j for j in quarter if j['order']==order and j['wall_layers']==layers],key=lambda j:(j['surface_level'],j.get('adaptive_refinements',0)))
                if not jobs:continue
                x=[j['minimum_surface_edge_mm'] for j in jobs]
                color=colors[1] if order==1 else {1:colors[2],2:'#bd782a',4:'#8850a0'}[layers]
                marker={1:'o',2:'s',4:'^'}[layers] if order==2 else 'o'
                axs[0].plot(x,[j['results']['10']['max_von_Mises_Pa']/1e6 for j in jobs],marker,color=color,ls=styles[layers],label=f'P{order}, {layers} wall layer(s)')
                axs[1].plot(x,[j['results']['10']['compliance_Nm']*4e6 for j in jobs],marker,color=color,ls=styles[layers])
                for xx,j in zip(x,jobs):
                    if j.get('adaptive_refinements',0) and layers!=2:axs[0].annotate(f"local A{j['adaptive_refinements']}",(xx,j['results']['10']['max_von_Mises_Pa']/1e6),xytext=(-5,10),textcoords='offset points',ha='center',fontsize=8)
        for ax in axs:
            ax.set_xscale('log');ax.invert_xaxis();ax.set_xlabel('Minimum surface edge (mm; refinement to right)');ax.grid(alpha=.2)
            ax.xaxis.set_major_locator(FixedLocator([3.,1.5,.75,.4,.2]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        axs[0].set_ylim(top=max(j['results']['10']['max_von_Mises_Pa']/1e6 for j in quarter)+.65)
        axs[0].set_ylabel('Peak von Mises (MPa)');axs[1].set_ylabel('Full-body equivalent compliance (micro N m)')
        axs[0].legend(frameon=False,fontsize=8);axs[0].set_title('Uniform + local refinement at the pads');axs[1].set_title('Global energy / deformation convergence')
        fig.suptitle('Hollow egg: measured mesh convergence | 4 N per finger | 0.5 mm wall\nQuarter-shell symmetry; unaveraged element stress; P1 and P2 at identical geometry',fontsize=12)
        fig.savefig(out/'eggshell_convergence_refined.png',dpi=180);plt.close(fig)


def save(data,out):
    (out/'eggshell_convergence_cpu_results.json').write_text(json.dumps(data,indent=2))
    rows=[]
    for j in data['jobs']:
        for quad,m in j['results'].items():
            rows.append({**{k:j[k] for k in ['label','order','surface_level','wall_layers','nodes','elements','mean_surface_edge_mm','assembly_s','factor_s','rss_MB']},
                         'family':j.get('family','icosphere'),'adaptive_refinements':j.get('adaptive_refinements',0),
                         'surface_quadrature_order':quad,**{k:v for k,v in m.items() if np.isscalar(v)}})
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (out/'eggshell_convergence_cpu.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='+',default=['1:2:2','1:3:1','1:3:2','1:3:4','1:4:1','1:4:2','1:4:4','2:2:1','2:3:1','2:3:2','2:4:1'])
    p.add_argument('--append',action='store_true');p.add_argument('--plot',action='store_true');p.add_argument('--output-dir',type=Path,default=HERE)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);out=args.output_dir
    path=out/'eggshell_convergence_cpu_results.json'
    with threadpool_limits(limits=1):
        data=json.loads(path.read_text()) if args.append and path.exists() else {'checks':element_checks(),'jobs':[],
             'force_per_finger_N':4.,'wall_thickness_m':.0005,'assumed_young_Pa':1e10,'poisson':.3,
             'criterion':'Relative change on refinement; <5% peak and <2% compliance are practical diagnostics, not a rigorous error bound.'}
        data['metric_definitions']={
            'stress':'Maximum unaveraged element von Mises; P1 element constant, P2 maximum over four element vertices.',
            'scaled_equilibrium_residual':'||K u - f|| / max(||f||, 1 N); evaluate free DOFs in a symmetry model.',
            'compliance_Nm':'f^T u; multiply quarter-shell values by 4 for full-body-equivalent compliance.',
            'volume_weighted_RMS_element_peak_Pa':'Volume-weighted RMS of each element maximum; not pointwise stress RMS.',
            'surface_quadrature':'0: centroid; n > 0: n x n Duffy tensor Gauss points per triangle.'}
        for job in data['jobs']:
            for metrics in job['results'].values():
                for old,new in [('relative_residual','scaled_equilibrium_residual'),
                                ('volume_RMS_von_Mises_Pa','volume_weighted_RMS_element_peak_Pa')]:
                    if old in metrics:metrics[new]=metrics.pop(old)
        save(data,out)
        if any(j.startswith('q') for j in args.jobs) and 'symmetry_check' not in data['checks']:
            data['checks']['symmetry_check']=symmetry_check();save(data,out)
        for spec in args.jobs:
            parts=spec.split(':');quarter=parts[0].startswith('q')
            order=int(parts[0].lstrip('q'));level=int(parts[1]);layers=int(parts[2]);adaptive=int(parts[3]) if len(parts)>3 else 0
            if any(j['order']==order and j['surface_level']==level and j['wall_layers']==layers and j.get('family','icosphere')==('octa_quarter' if quarter else 'icosphere') and j.get('adaptive_refinements',0)==adaptive for j in data['jobs']):continue
            result=run_job(order,level,layers,gauge=(level==3 and layers==1),quarter=quarter,adaptive=adaptive)
            data['jobs'].append(result);save(data,out)
            print(json.dumps({'completed':result['label'],'rss_after_MB':rss_mb()}),flush=True)
        if args.plot:plot(data,out)
    print('Convergence run complete.',flush=True)


if __name__=='__main__':main()
