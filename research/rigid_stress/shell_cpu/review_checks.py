"""Independent analytical/invariance checks of the CPU shell prototype.

This file does not assemble the shell stiffness.  It tests the implementation
against rigid-motion identities, prescribed continuum strain energies, the
Navier thin-plate solution, and the spherical membrane solution.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy import sparse, linalg
from scipy.integrate import quad
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import eigsh


def plate_mesh(nx=4, ny=None, lx=0.1, ly=None):
    ny = nx if ny is None else ny
    ly = lx if ly is None else ly
    vertices = np.array([[x, y, 0.] for y in np.linspace(0, ly, ny+1)
                         for x in np.linspace(0, lx, nx+1)])
    quads = np.array([[j*(nx+1)+i, j*(nx+1)+i+1,
                       (j+1)*(nx+1)+i+1, (j+1)*(nx+1)+i]
                      for j in range(ny) for i in range(nx)], dtype=int)
    return vertices, quads


def sphere_mesh(n=4, radius=0.025):
    """Stitched six-face radial cube projection, independently generated."""
    vertices, faces, lookup = [], [], {}
    for axis in range(3):
        a, b = (axis+1) % 3, (axis+2) % 3
        for sign in [-1, 1]:
            ids = np.empty((n+1, n+1), dtype=int)
            for j in range(n+1):
                for i in range(n+1):
                    key = [0, 0, 0]
                    key[axis] = sign*n
                    key[a] = -n+2*i
                    key[b] = sign*(-n+2*j)
                    key = tuple(key)
                    if key not in lookup:
                        lookup[key] = len(vertices)
                        v = np.array(key, dtype=float)
                        vertices.append(radius*v/np.linalg.norm(v))
                    ids[j, i] = lookup[key]
            for j in range(n):
                for i in range(n):
                    faces.append([ids[j,i], ids[j,i+1], ids[j+1,i+1], ids[j+1,i]])
    return np.array(vertices), np.array(faces, dtype=int)


def rotation_matrix():
    axis = np.array([0.3, -0.5, 0.8]); axis /= np.linalg.norm(axis)
    angle = 0.71
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


def plane_D(E, nu):
    return E/(1-nu*nu)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])


def relative(a, b):
    return float(abs(a-b)/max(abs(b), 1e-300))


def test_invariants(ShellModel):
    v, f = plate_mesh(2)
    v[:, 2] = 0.002*np.sin(v[:,0]*np.pi/0.1)*np.sin(v[:,1]*np.pi/0.1)
    m = ShellModel(v, f, E=1e10, nu=.3, h=.0005)
    K = m.K
    maxdiag = float(np.max(np.abs(K.diagonal())))
    sym = float(np.linalg.norm((K-K.T).data)/np.linalg.norm(K.data))
    rigid = m.rigid_modes
    residuals = np.linalg.norm(K@rigid, axis=0)/(np.linalg.norm(K.data)*np.linalg.norm(rigid,axis=0))
    vals = np.linalg.eigvalsh(K.toarray())/maxdiag
    rng = np.random.default_rng(123)
    q = rng.normal(size=m.ndof)*1e-6
    R = rotation_matrix(); shift = np.array([.3,-.2,.7])
    rot = ShellModel(v@R.T+shift, f, E=1e10, nu=.3, h=.0005)
    qrot = q.reshape(-1, 2, 3)@R.T
    e = float(q@(K@q))
    er = float(qrot.ravel()@(rot.K@qrot.ravel()))
    force_rot = (K@q).reshape(-1,2,3)@R.T
    force_direct = (rot.K@qrot.ravel()).reshape(-1,2,3)
    return dict(symmetry_relative=sym, rigid_relative=residuals.tolist(),
                eigenvalues_scaled=vals[:12].tolist(),
                near_zero_eigenvalues=int(np.sum(np.abs(vals)<1e-10)),
                negative_eigenvalues=int(np.sum(vals < -1e-10)),
                rotation_energy_relative=relative(er,e),
                rotation_operator_relative=float(np.linalg.norm(force_direct-force_rot)/np.linalg.norm(force_rot)))


def test_flat_patches(ShellModel):
    E, nu, h = 1e10, .3, .0005
    v, f = plate_mesh(4); area=.1**2
    m=ShellModel(v,f,E=E,nu=nu,h=h)
    ex, ey, gamma=1e-5,-2e-6,7e-6
    q=np.zeros((len(v),6))
    q[:,0]=ex*v[:,0]+.5*gamma*v[:,1]
    q[:,1]=ey*v[:,1]+.5*gamma*v[:,0]
    strain=np.array([ex,ey,gamma])
    expected=.5*area*h*(strain@plane_D(E,nu)@strain)
    measured=.5*q.ravel()@(m.K@q.ravel())
    kx,ky,kxy=.02,-.007,.011
    qb=np.zeros_like(q)
    x,y=v[:,0],v[:,1]
    qb[:,2]=.5*kx*x*x+.5*ky*y*y+kxy*x*y
    qb[:,3]=ky*y+kxy*x
    qb[:,4]=-kx*x-kxy*y
    curvature=np.array([-kx,-ky,-2*kxy])
    expected_b=.5*area*h**3/12*(curvature@plane_D(E,nu)@curvature)
    measured_b=.5*qb.ravel()@(m.K@qb.ravel())
    return dict(membrane_energy=float(measured),membrane_analytic=float(expected),
                membrane_relative=relative(measured,expected),
                bending_energy=float(measured_b),bending_analytic=float(expected_b),
                bending_relative=relative(measured_b,expected_b))


def quad_pressure_load(vertices, quads, pressure, order=4):
    """Consistent pressure on the bilinear surface, independent shape code."""
    gp,gw=np.polynomial.legendre.leggauss(order)
    F=np.zeros((len(vertices),6))
    for face in quads:
        X=vertices[face]
        for i,xi in enumerate(gp):
            for j,eta in enumerate(gp):
                N=.25*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),
                                (1+xi)*(1+eta),(1-xi)*(1+eta)])
                dx=.25*np.array([-(1-eta),1-eta,1+eta,-(1+eta)])
                de=.25*np.array([-(1-xi),-(1+xi),1+xi,1-xi])
                df=pressure*np.cross(dx@X,de@X)*gw[i]*gw[j]
                F[face,:3] += N[:,None]*df[None,:]
    return F.ravel()


def navier_coefficient(terms=101):
    odd=np.arange(1,terms,2)
    m,n=np.meshgrid(odd,odd,indexing='ij')
    sign=(-1.)**((m-1)//2+(n-1)//2)
    return float(16/np.pi**6*np.sum(sign/(m*n*(m*m+n*n)**2)))


def test_navier(ShellModel, resolutions=(4,8,16), ratios=(.01,.001)):
    L=.1; E=1e10; nu=.3; pressure=.01
    coefficient=navier_coefficient()
    rows=[]
    for ratio in ratios:
        h=L*ratio; D=E*h**3/(12*(1-nu*nu))
        exact=coefficient*pressure*L**4/D
        for n in resolutions:
            v,f=plate_mesh(n,lx=L)
            model=ShellModel(v,f,E=E,nu=nu,h=h)
            force=quad_pressure_load(v,f,pressure)
            boundary=(np.isclose(v[:,0],0)|np.isclose(v[:,0],L)|
                      np.isclose(v[:,1],0)|np.isclose(v[:,1],L))
            fixed=np.r_[np.arange(len(v))*6, np.arange(len(v))*6+1,
                        np.arange(len(v))*6+5, np.flatnonzero(boundary)*6+2]
            q=model.solve(force,fixed_dofs=np.unique(fixed))
            if isinstance(q,tuple): q=q[0]
            center=np.argmin(np.linalg.norm(v[:,:2]-L/2,axis=1))
            measured=float(np.asarray(q).reshape(-1,6)[center,2])
            rows.append(dict(n=n,h_over_L=ratio,center_displacement=measured,
                             analytic_KL_displacement=exact,
                             relative_KL_difference=relative(measured,exact)))
    return dict(navier_coefficient=coefficient,rows=rows)


def test_sphere(ShellModel,resolutions=(2,4,8,12),h=.0005):
    E,nu,R,p=1e10,.3,.025,100.
    expected=p*R*R*(1-nu)/(2*E*h)
    expected_sigma=p*R/(2*h)
    rows=[]
    for n in resolutions:
        v,f=sphere_mesh(n,R)
        model=ShellModel(v,f,E=E,nu=nu,h=h,directors=v/R)
        force=quad_pressure_load(v,f,p)
        q=model.solve(force)
        if isinstance(q,tuple):q=q[0]
        q=np.asarray(q)
        # Remove any gauge-dependent rigid motion before spatial error metrics.
        rigid=model.rigid_modes
        q=q-rigid@np.linalg.lstsq(rigid,q,rcond=None)[0]
        radial=np.einsum('ij,ij->i',q.reshape(-1,6)[:,:3],v/R)
        stress=model.stress(q)
        membrane=stress['N']/h
        exact_membrane=np.broadcast_to([expected_sigma,expected_sigma,0.],membrane.shape)
        weights=model.gauss_weights
        tensor_error=np.sqrt(np.sum(weights[:,:,None]*(membrane-exact_membrane)**2)
                             /np.sum(weights[:,:,None]*exact_membrane**2))
        membrane_mean=np.sum(weights[:,:,None]*membrane,axis=(0,1))/weights.sum()
        rows.append(dict(n=n,nodes=len(v),elements=len(f),dofs=model.ndof,
                         radial_mean=float(radial.mean()),radial_std=float(radial.std()),
                         analytic_radial=expected,
                         relative_radial_mean=relative(radial.mean(),expected),
                         relative_radial_L2=float(np.linalg.norm(radial-expected)/np.sqrt(len(v))/abs(expected)),
                         analytic_membrane_stress=expected_sigma,
                         membrane_stress_mean=membrane_mean.tolist(),
                         relative_membrane_stress_L2=float(tensor_error),
                         total_force=np.asarray(force).reshape(-1,6)[:,:3].sum(axis=0).tolist()))
    return dict(rows=rows)


def test_sphere_thickness(ShellModel):
    rows=[]
    for ratio in (.05,.025,.01,.005):
        case=test_sphere(ShellModel,resolutions=(8,12),h=.025*ratio)
        for r in case['rows']:
            r['h_over_R']=ratio
            rows.append(r)
    return {'rows':rows,
            'scope':'Uniform membrane-pressure thickness control; does not exclude locking for all bending-dominated curved states.'}


def test_drilling_sensitivity(ShellModel):
    R=.025;E=1e10;nu=.3;h=.0005;p=100.
    v,f=sphere_mesh(4,R);force=quad_pressure_load(v,f,p)
    rows=[]
    for parameter in (1e-4,1e-3,1e-2):
        model=ShellModel(v,f,E=E,nu=nu,h=h,directors=v/R,drilling_factor=parameter)
        q=model.solve(force)
        radial=np.einsum('ij,ij->i',q.reshape(-1,6)[:,:3],v/R)
        stress=model.stress(q)['N']/h
        rows.append(dict(drilling_factor=parameter,radial_mean=float(radial.mean()),
                         stress_mean=float(np.mean(stress[:,:,:2]))))
    means=[r['radial_mean'] for r in rows]
    return {'rows':rows,'relative_radial_range':float((max(means)-min(means))/np.mean(means)),
            'scope':'Uniform pressure sphere; this single sensitivity control does not bound all asymmetric contact cases.'}


def cylinder_mesh(n=32,nz=2,R=.025,L=.05):
    theta=np.arange(n)*2*np.pi/n
    verts=np.array([[R*np.cos(t),R*np.sin(t),z]
                    for z in np.linspace(0,L,nz+1) for t in theta])
    faces=np.array([[j*n+i,j*n+(i+1)%n,(j+1)*n+(i+1)%n,(j+1)*n+i]
                    for j in range(nz) for i in range(n)])
    normals=verts.copy();normals[:,2]=0;normals/=R
    return verts,faces,normals


def test_cylinder_ovalization(ShellModel):
    E,nu,R,L,p,mode=1e10,.3,.025,.05,.01,2
    rows=[]
    for ratio in (.05,.02,.005):
        h=R*ratio;D=E*h**3/(12*(1-nu*nu))
        expected=p*R**4/(D*(mode*mode-1)**2)
        for n in (16,32,64):
            vertices,faces,directors=cylinder_mesh(n,R=R,L=L)
            model=ShellModel(vertices,faces,E=E,nu=nu,h=h,directors=directors)
            # Identical cross-sections and zero axial/tilt DOFs give a
            # cylindrical ring per axial length in plane strain.
            ir=[];ic=[]
            for i in range(len(vertices)):
                for a,b in ((0,0),(1,1),(5,2)):
                    ir.append(6*i+a);ic.append(3*(i%n)+b)
            P=sparse.coo_matrix((np.ones(len(ir)),(ir,ic)),shape=(model.ndof,3*n)).tocsr()
            Kr=(P.T@model.K@P).tocsc()
            theta=np.arange(n)*2*np.pi/n
            reduced_force=np.zeros((n,3))
            reduced_force[:,:2]=p*R*L*2*np.pi/n*np.cos(mode*theta)[:,None]*directors[:n,:2]
            rigid=np.zeros((3*n,3))
            rigid[0::3,0]=1;rigid[1::3,1]=1
            rigid[0::3,2]=-vertices[:n,1]
            rigid[1::3,2]=vertices[:n,0];rigid[2::3,2]=1
            _,_,piv=linalg.qr(rigid.T,pivoting=True,mode='economic')
            free=np.setdiff1d(np.arange(3*n),piv[:3])
            qr=np.zeros(3*n);qr[free]=splu(Kr[free][:,free]).solve(reduced_force.ravel()[free])
            radial=np.einsum('ij,ij->i',qr.reshape(n,3)[:,:2],directors[:n,:2])
            measured=float(radial@np.cos(mode*theta)/(np.cos(mode*theta)@np.cos(mode*theta)))
            q=P@qr;stress=model.stress(q)
            em=.5*np.einsum('egi,ij,egj,eg->',stress['epsilon'],model.A,stress['epsilon'],model.gauss_weights)
            eb=.5*np.einsum('egi,ij,egj,eg->',stress['kappa'],model.D,stress['kappa'],model.gauss_weights)
            es=.5*np.einsum('egi,ij,egj,eg->',stress['Q'],np.linalg.inv(model.S),stress['Q'],model.gauss_weights)
            total=.5*q@(model.K@q)
            rows.append(dict(n=n,h_over_R=ratio,measured_W=measured,analytic_W=expected,
                             relative_displacement_error=relative(measured,expected),
                             energy_membrane=float(em),energy_bending=float(eb),energy_shear=float(es),
                             energy_drilling=float(total-em-eb-es)))
    return {'rows':rows,'scope':'Independent sinusoidal pressure ring/cylinder solution; identical axial slices and plane-strain axial kinematics.'}


def independent_rectangle_weights(vertices,quads,center,radius):
    """Integrate Q4 Hertz loads with analytic y moments + adaptive x quadrature.

    Independent of the contact code's polar fan quadrature. Only axis-aligned
    rectangular faces are used in this analytical control.
    """
    cx,cy=center[:2];a=radius
    weights=np.zeros(len(vertices));scale=3/(2*np.pi*a**3)
    for face in quads:
        X=vertices[face,:2];x0,y0=X.min(axis=0);x1,y1=X.max(axis=0)
        lo,hi=max(x0,cx-a),min(x1,cx+a)
        if hi<=lo:continue
        points=[lo,hi]
        for y in (y0,y1):
            if abs(y-cy)<a:
                root=np.sqrt(a*a-(y-cy)**2)
                points.extend([x for x in (cx-root,cx+root) if lo<x<hi])
        points=sorted(set(points))
        for k,node in enumerate(face):
            sx=1 if X[k,0]>=(x0+x1)/2 else -1
            sy=1 if X[k,1]>=(y0+y1)/2 else -1
            bx=sx/(x1-x0);ax=(1 if sx<0 else 0)-bx*x0
            by=sy/(y1-y0);ay=(1 if sy<0 else 0)-by*y0
            def integrand(x):
                A=max(0.,a*a-(x-cx)**2);root=np.sqrt(A)
                yl,yh=max(y0-cy,-root),min(y1-cy,root)
                if yh<=yl or A==0:return 0.
                def primitive0(y):
                    if abs(y/root)>=1-2e-14:
                        return np.sign(y)*np.pi*A/4
                    return .5*(y*np.sqrt(max(0.,A-y*y))+A*np.arcsin(np.clip(y/root,-1,1)))
                def primitive1(y):
                    return 0. if abs(y/root)>=1-2e-14 else -max(0.,A-y*y)**1.5/3
                J0=primitive0(yh)-primitive0(yl)
                J1=primitive1(yh)-primitive1(yl)
                return scale*(ax+bx*x)*((ay+by*cy)*J0+by*J1)
            weights[node]+=sum(quad(integrand,l,r,epsabs=2e-13,epsrel=2e-12,limit=150)[0]
                               for l,r in zip(points[:-1],points[1:]))
    return weights


def test_contacts(ShellModel):
    from shell_cpu.contacts import Contact,ShellDPRTransfer,hertz_radius
    v,f=plate_mesh(4);h=.0005
    model=ShellModel(v,f,E=1e10,nu=.3,h=h)
    transfer=ShellDPRTransfer(model)
    rng=np.random.default_rng(991);q=rng.normal(size=model.ndof)*1e-5
    rows=[]
    for side in (1,-1):
        force=np.array([.2,-.1,-side*1.])
        for center,radius in (([.037,.033],.003),([.047,.054],.009),([.05,.05],.009)):
            point=np.array([*center,side*h/2]);normal=np.array([0.,0.,float(side)])
            contact=Contact(point,force,normal,(50.,80.),side=side,radius=radius,friction=.4)
            weights=independent_rectangle_weights(v,f,point,radius)
            exact=np.zeros((len(v),6));exact[:,:3]=weights[:,None]*force
            exact[:,3:]=weights[:,None]*(side*h/2*np.cross(model.directors,force))
            for order in (16,24,32,48):
                row=dict(side=side,center=center,radius=radius,order=order,
                         independent_mass_error=float(weights.sum()-1))
                try:
                    load,info=transfer.transfer(contact,order=order,adaptive=False)
                    nodal_error=float(np.linalg.norm(load-exact.ravel())/np.linalg.norm(exact))
                    work=float(q@load);exact_work=float(q@exact.ravel())
                    row.update(method=info['method'],faces=info['faces'],
                               force_error=info['force_error'],moment_error=info['moment_error'],
                               independent_nodal_load_error=nodal_error,
                               actual_order=info.get('order',order),
                               retry_failures=info.get('retry_failures',[]),
                               virtual_work_normalized=float(abs(work-exact_work)/(np.linalg.norm(q)*np.linalg.norm(exact))),
                               virtual_work_relative=relative(work,exact_work))
                except ValueError as exc:
                    row['error']=str(exc)
                rows.append(row)
    # Same physical geometry/data rotated and translated together.
    R=rotation_matrix();shift=np.array([.3,-.2,.7])
    rotated=ShellModel(v@R.T+shift,f,E=1e10,nu=.3,h=h,directors=model.directors@R.T)
    c=Contact(np.array([.047,.054,h/2]),np.array([.2,-.1,-1.]),np.array([0.,0.,1.]),(50.,80.),radius=.009)
    f0,i0=transfer.transfer(c,order=48,adaptive=False)
    cr=Contact(c.point@R.T+shift,c.force@R.T,c.normal@R.T,c.curvatures,side=c.side,radius=c.radius)
    fr,ir=ShellDPRTransfer(rotated).transfer(cr,order=48,adaptive=False)
    expected=f0.reshape(-1,2,3)@R.T
    zero,_=transfer.transfer(Contact(c.point,np.zeros(3),c.normal,c.curvatures))
    a1=hertz_radius(1.,curvatures=(50.,80.));a8=hertz_radius(8.,curvatures=(50.,80.))
    return dict(rows=rows,rigid_transform_load_relative=float(np.linalg.norm(fr-expected.ravel())/np.linalg.norm(expected)),
                release_exact_zero=bool(np.array_equal(zero,np.zeros(model.ndof))),
                cubic_force_radius_ratio=a8/a1,
                scope='Analytical one-dimensional Hertz moments and discrete virtual work; not independent contact-radius physics.')


def test_nonconvex_transfer(ShellModel):
    from shell_cpu.contacts import Contact,ShellDPRTransfer
    width,height,length=.03,.02,.04;m=4;nz=4;h=.0005
    xy=[[-width/2,height*(1-j/m)] for j in range(m+1)]
    xy += [[-width/2+width*i/m,0.] for i in range(1,m+1)]
    xy += [[width/2,height*j/m] for j in range(1,m+1)]
    ns=len(xy)
    v=np.array([[x,y,z] for z in np.linspace(0,length,nz+1) for x,y in xy])
    faces=np.array([[j*ns+i,j*ns+i+1,(j+1)*ns+i+1,(j+1)*ns+i]
                    for j in range(nz) for i in range(ns-1)])
    model=ShellModel(v,faces,E=1e10,nu=.3,h=h)
    transfer=ShellDPRTransfer(model);rows=[]
    for wall in (-1,1):
        outward=np.array([-float(wall),0.,0.])
        point=np.array([wall*(width/2-h/2),height*.7,length*.47])
        force=-outward+np.array([0.,.1,.15])
        load,info=transfer.transfer(Contact(point,force,outward,(50.,80.),side=-1,friction=.4),order=32)
        active=np.flatnonzero(np.linalg.norm(load.reshape(-1,6),axis=1)>1e-15)
        rows.append(dict(wall=wall,loaded_nodes=int(len(active)),
                         actual_wall_preserved=bool(np.all(wall*v[active,0]>.4*width)),
                         force_error=info['force_error'],moment_error=info['moment_error'],
                         radius_m=info['radius_m'],method=info['method']))
    return {'rows':rows,'scope':'Actual facing inner walls of a connected U-shaped shell; positive supplied relative curvature is a Hertz input assumption, not a contact solution.'}


def main():
    from shell_cpu.core import ShellModel
    out={"scope":"Independent analytical/invariance review; same linear shell theory, not a 3D contact reference.",
         "source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in ('shell_cpu/core.py','shell_cpu/contacts.py','shell_cpu/review_checks.py')}}
    checks=[('invariants',test_invariants),('patches',test_flat_patches),
            ('navier',test_navier),('sphere',test_sphere),
            ('sphere_thickness',test_sphere_thickness),('drilling_sensitivity',test_drilling_sensitivity),
            ('cylinder_ovalization',test_cylinder_ovalization),('contacts',test_contacts),
            ('nonconvex_transfer',test_nonconvex_transfer)]
    for name,fn in checks:
        start=time.perf_counter()
        try:out[name]=fn(ShellModel)
        except Exception as exc:
            import traceback
            out[name]={"error":repr(exc),"traceback":traceback.format_exc()}
        out[name]['seconds']=time.perf_counter()-start
        print(name,json.dumps(out[name]),flush=True)
        Path('shell_cpu/review_results/review.json').write_text(json.dumps(out,indent=2))
    checks={
        'stiffness_symmetric':out['invariants'].get('symmetry_relative',1)<1e-13,
        'six_and_only_six_rigid_null_modes':out['invariants'].get('near_zero_eigenvalues')==6,
        'no_negative_modes':out['invariants'].get('negative_eigenvalues')==0,
        'rigid_and_rotational_invariance':max(out['invariants'].get('rigid_relative',[1]))<1e-12
            and out['invariants'].get('rotation_operator_relative',1)<1e-12,
        'analytic_membrane_bending_energy':out['patches'].get('membrane_relative',1)<1e-10
            and out['patches'].get('bending_relative',1)<1e-8,
        'thin_plate_n16_below_half_percent':all(r['relative_KL_difference']<.005
            for r in out['navier'].get('rows',[]) if r['n']==16),
        'sphere_n12_mean_and_stress_below_one_percent':all(
            r['relative_radial_mean']<.01 and r['relative_membrane_stress_L2']<.01
            for r in out['sphere'].get('rows',[]) if r['n']==12),
        'cylinder_n64_below_half_percent':all(r['relative_displacement_error']<.005
            for r in out['cylinder_ovalization'].get('rows',[]) if r['n']==64),
        'drilling_uniform_sphere_sensitivity_below_point_one_percent':
            out['drilling_sensitivity'].get('relative_radial_range',1)<.001,
        'contact_high_order_independent_loads':all(
            'error' not in r and r.get('independent_nodal_load_error',1)<1e-8
            for r in out['contacts'].get('rows',[]) if r['order']>=32),
        'contact_rigid_transform_invariance':out['contacts'].get('rigid_transform_load_relative',1)<1e-11,
        'release_zero':out['contacts'].get('release_exact_zero',False),
        'nonconvex_actual_inner_sheet':all(r['actual_wall_preserved']
            for r in out['nonconvex_transfer'].get('rows',[])),
    }
    out['selected_control_checks']=checks
    out['selected_control_checks_passed']=all(checks.values())
    out['source_unchanged_during_run']=all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in out['source_sha256'].items())
    Path('shell_cpu/review_results/review.json').write_text(json.dumps(out,indent=2))
    print('selected_control_checks',json.dumps(checks),flush=True)


if __name__=='__main__':main()
