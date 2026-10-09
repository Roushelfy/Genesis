#!/usr/bin/env python3
"""Full-body P2 validation with unrestricted finite-area contact inputs.

The reference shape is fixed; material and density are known model inputs.
Contact centers, Cartesian forces, number, radii and symmetry may change.
This deliberately small mesh is NOT stress-converged. This is not a Genesis
contact-solver integration, an RL throughput measurement, or a fracture test.
"""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import eggshell_gripper_cpu as egg
import eggshell_convergence_cpu as conv


class SurfaceGeometry:
    """Consistent six-node shape integration on the straight outer faces."""
    def __init__(self,f,quad):
        self.f=f;self.quad=quad
        self.bary,self.weights=conv.triangle_rule(quad)
        faces=f.outer_faces;vertices=f.base_xyz[faces]
        edges=np.sort(faces[:,[[0,1],[0,2],[1,2]]],axis=2)
        keys=edges[:,:,0]*f.nbase+edges[:,:,1]
        mids=f.nbase+np.searchsorted(f.edge_keys,keys)
        assert np.array_equal(f.edge_keys[mids-f.nbase],keys)
        self.nodes=np.column_stack([faces,mids])
        b=self.bary
        self.shape=np.column_stack([b*(2*b-1),4*b[:,0]*b[:,1],4*b[:,0]*b[:,2],4*b[:,1]*b[:,2]])
        self.coords=np.einsum('qi,fic->fqc',b,vertices)
        self.area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
        self.integration_weights=self.area[:,None]*self.weights[None,:]
        # Signed corner weights are necessary for consistent quadratic loads.
        assert np.max(np.abs(self.shape.sum(axis=1)-1))<1e-14
        reconstructed=np.einsum('qi,fic->fqc',self.shape,f.xyz[self.nodes])
        assert np.max(np.abs(reconstructed-self.coords))<1e-14


def patch_weights(geometry,patch):
    center=np.asarray(patch['center_m']);radius=patch['radius_m'];sigma=patch['sigma_m']
    distance2=np.sum((geometry.coords-center)**2,axis=2)
    pressure=np.exp(-.5*distance2/sigma**2)*np.maximum(0.,1-distance2/radius**2)**2
    weight=pressure*geometry.integration_weights
    integral=float(weight.sum())
    assert integral>0,('No finite contact footprint sampled',patch)
    return weight/integral,distance2,integral


def check_patch(geometry,patch,weight,distance2,integral):
    f=geometry.f;F=np.asarray(patch['force_N'])
    load=weight@geometry.shape
    nodal=np.zeros(len(f.xyz))
    for a in range(6):np.add.at(nodal,geometry.nodes[:,a],load[:,a])
    centroid=np.einsum('fq,fqc->c',weight,geometry.coords)
    actual_force=nodal.sum()*F
    actual_torque=np.cross(f.xyz,nodal[:,None]*F).sum(axis=0)
    expected_torque=np.cross(centroid,F)
    force_error=float(np.linalg.norm(actual_force-F))
    torque_error=float(np.linalg.norm(actual_torque-expected_torque))
    assert force_error<1e-11*max(np.linalg.norm(F),1.)
    assert torque_error<1e-11*max(np.linalg.norm(expected_torque),1.)
    return {'quadrature_force_centroid_m':centroid.tolist(),
            'nodal_force_N':actual_force.tolist(),'nodal_torque_Nm':actual_torque.tolist(),
            'force_error_N':force_error,'torque_error_Nm':torque_error,
            'pressure_integral_m2':integral,
            'RMS_center_distance_m':float(np.sqrt(np.sum(weight*distance2))),
            'support_triangles':int(np.count_nonzero(weight.sum(axis=1))),
            'signed_nodal_weight_min':float(nodal.min())}


def load_old(f,patches,quad):
    """Straight baseline: recompute surface geometry separately per patch."""
    raw=np.zeros((len(f.xyz),3))
    for patch in patches:
        geometry=SurfaceGeometry(f,quad)
        weight,_,_=patch_weights(geometry,patch)
        element=weight@geometry.shape
        for a in range(6):
            for axis in range(3):
                np.add.at(raw[:,axis],geometry.nodes[:,a],element[:,a]*patch['force_N'][axis])
    return raw.ravel()


def load_shared(geometry,patches,validate=False):
    """Reuse all surface quadrature geometry and scatter combined tractions."""
    f=geometry.f;raw=np.zeros((len(f.xyz),3));metadata=[]
    weighted_force=np.zeros((*geometry.coords.shape[:2],3))
    for patch in patches:
        weight,distance2,integral=patch_weights(geometry,patch)
        weighted_force+=weight[:,:,None]*np.asarray(patch['force_N'])
        if validate:metadata.append(check_patch(geometry,patch,weight,distance2,integral))
    element=np.einsum('fqc,qa->fac',weighted_force,geometry.shape)
    for a in range(6):
        for axis in range(3):np.add.at(raw[:,axis],geometry.nodes[:,a],element[:,a,axis])
    return raw.ravel(),metadata


def surface_frame(f,face,bary):
    vertices=f.base_xyz[f.outer_faces[face]]
    center=np.asarray(bary)@vertices
    normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]);normal/=np.linalg.norm(normal)
    if normal@(center-f.com)<0:normal=-normal
    axis=np.eye(3)[np.argmin(np.abs(normal))]
    tangent=np.cross(normal,axis);tangent/=np.linalg.norm(tangent)
    return center,normal,tangent,np.cross(normal,tangent)


def make_patch(f,face,bary,radius,force_components):
    bary=np.asarray(bary);bary/=bary.sum()
    center,n,t1,t2=surface_frame(f,face,bary)
    Fn,Ft1,Ft2=force_components
    F=-Fn*n+Ft1*t1+Ft2*t2
    return {'face':int(face),'barycentric_center':bary.tolist(),
            'center_m':center.tolist(),'radius_m':float(radius),'sigma_m':float(.45*radius),
            'force_N':F.tolist(),'normal_tangent_components_N':list(map(float,force_components))}


def workloads(f,frames):
    rng=np.random.default_rng(20261009);faces=len(f.outer_faces);cases={}
    initial_faces=rng.choice(faces,3,replace=False)
    smooth=[]
    for i,t in enumerate(np.linspace(0,1,frames)):
        patches=[]
        for j,face in enumerate(initial_faces):
            bary=np.array([.34+.12*np.sin(2*np.pi*t+j),.33+.1*np.cos(2*np.pi*t+.3*j),0.])
            bary[2]=1-bary[:2].sum()
            patches.append(make_patch(f,face,bary,.0055+.0007*np.sin(2*np.pi*t+j),
                                      [2.+.5*np.sin(2*np.pi*t+j),.3*np.cos(3*np.pi*t+j),.2*np.sin(3*np.pi*t)]))
        smooth.append({'patches':patches,'omega_rad_s':[3*np.sin(2*np.pi*t),2.,1.]})
    cases['continuous_centers_forces_radii']=smooth
    stages=[]
    for count in [1,4,2]:
        patches=[make_patch(f,int(rng.integers(faces)),rng.dirichlet([2,2,2]),
                            rng.uniform(.004,.0075),[rng.uniform(.5,4),rng.uniform(-.8,.8),rng.uniform(-.8,.8)])
                 for _ in range(count)]
        stages.append({'patches':patches,'omega_rad_s':rng.uniform(-5,5,3).tolist()})
    cases['position_count_direction_jumps']=[stages[min(2,3*i//frames)] for i in range(frames)]
    random=[]
    for _ in range(frames):
        patches=[make_patch(f,int(rng.integers(faces)),rng.dirichlet([1.5,1.5,1.5]),
                            rng.uniform(.004,.0075),[rng.uniform(.2,4.5),rng.uniform(-1.2,1.2),rng.uniform(-1.2,1.2)])
                 for _ in range(int(rng.integers(1,5)))]
        random.append({'patches':patches,'omega_rad_s':rng.uniform(-8,8,3).tolist()})
    cases['random_unrestricted_patches']=random
    cases['zero_unilateral_and_spin']=[
        {'patches':[],'omega_rad_s':[0.,0.,0.]},
        {'patches':[make_patch(f,int(initial_faces[0]),[.21,.37,.42],.006,[3.,1.,-.6])],
         'omega_rad_s':[0.,4.,-2.]},
        {'patches':[],'omega_rad_s':[0.,0.,20.]},
        {'patches':[],'omega_rad_s':[0.,0.,0.]},
    ]
    return cases


def peak(f,u):
    maximum=0.
    for _,_,eps in f.strain(u,np.eye(4)):
        maximum=max(maximum,float(conv.ref.von_mises(eps@f.d.T).max()))
    return maximum


def centrifugal_inertia(f,omega):
    r=f.xyz-f.com
    acceleration=np.cross(omega,np.cross(omega,r))
    return f.m@acceleration.ravel()


def validate(f,u,rhs,raw,reference_peak):
    difference=f.k@u-rhs;normrhs=float(np.linalg.norm(rhs))
    scaled=float(np.linalg.norm(difference)/max(normrhs,1.))
    relative=float(np.linalg.norm(difference)/normrhs) if normrhs>1e-16 else None
    assert scaled<1e-7,scaled
    if relative is not None:assert relative<3e-8,relative
    wrench=f.r.T@rhs
    assert np.linalg.norm(wrench)<1e-10*max(normrhs,1.)
    if normrhs==0.:
        assert np.all(u==0.)
        metrics={'max_von_Mises_Pa':0.,'energy_consistency_error':0.,'hotspot_m':None}
    else:
        metrics=f.metrics(u,raw,rhs,scaled)
    error=abs(metrics['max_von_Mises_Pa']-reference_peak)/max(reference_peak,1.)
    assert error<1e-7,error
    return {'full_equilibrium_residual_1N_floor':scaled,'full_equilibrium_relative_nonzero':relative,
            'balanced_six_mode_wrench':wrench.tolist(),'peak_relative_error_1Pa_floor':error,
            'energy_consistency_error':metrics['energy_consistency_error'],
            'peak_Pa':metrics['max_von_Mises_Pa'],'hotspot_m':metrics['hotspot_m']}


def stats(times):
    t=np.asarray(times)
    return {'p50_ms':float(np.median(t)*1e3),'p95_ms':float(np.percentile(t,95)*1e3),
            'mean_ms':float(t.mean()*1e3),'queries_per_second':float(len(t)/t.sum())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,default=16)
    parser.add_argument('--quad',type=int,default=10);parser.add_argument('--level',type=int,default=2)
    parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--output',type=Path,default=HERE/'arbitrary_contact_results.json')
    args=parser.parse_args();assert args.frames>=3
    data={'scope':'Full-body unrestricted finite-area contact algorithm validation on a deliberately coarse, unconverged mesh.',
          'contact_assumptions':'Finite footprint is a per-frame input; no fixed centers, force directions, contact count, or symmetry. Known same-body material E=10GPa,nu=0.3,rho=2000kg/m3; varying material would require operator updates.',
          'CPU':egg.cpu_name(),'platform':platform.platform(),'blas_threads':1,'GPU_tested':False,
          'straight_geometry_P2':True,'centrifugal_inertia':'raw=f_contact-M[omega cross (omega cross r)]; then six-mode inertia relief',
          'reference_timing_excluded':True,'cases':{},'reverse_gauge_checks':[]}
    data['validation_interpretation']={
        'sequence':'Changing-input algebraic snapshots; acceleration inferred independently from each load, not a temporally consistent rigid-body or elastodynamic trajectory.',
        'frame':'All contact centers, forces and angular velocities use the reference geometry material frame.',
        'profile':'Smooth finite patches are test inputs. Their moment uses the actual integrated force centroid, not necessarily the nominal patch center.',
        'reference':'Direct and incremental solve paths share a numerical factor. Additional independent checks include full equilibrium residual, six-mode wrench, alternate gauge and energy.',
        'surface':'Outer surface only in this prototype. The arbitrary traction interface maps any vector samples on this stored outer quadrature, not exact continuous traction integration.'}
    with threadpool_limits(limits=1):
        xyz,tet,outer,mesh=egg.egg_mesh(args.level,args.layers)
        f=conv.FEM(xyz,tet,outer,order=2,quarter=False,label='arbitrary-contact-full-P2')
        start=perf_counter();geometry=SurfaceGeometry(f,args.quad);geometry_setup=perf_counter()-start
        data['mesh']={**mesh,'nodes_P2':len(f.xyz),'DOFs':f.ndof,'tetrahedra':f.ne,'corner_stress_samples':4*f.ne,
                      'assembly_s':f.assembly_s,'factor_s':f.factor_s,'cached_surface_geometry_s':geometry_setup,
                      'gauge_scalar_rows':f.pins.tolist(),'surface_quadrature_order':args.quad,'stress_converged':False}
        descriptors=workloads(f,args.frames);reverse_candidates=[]
        for name,sequence in descriptors.items():
            previous_rhs=np.zeros(f.ndof);previous_u=np.zeros(f.ndof)
            timings={k:[] for k in ['contact_mapping_old','contact_mapping_shared','cached_factor_and_peak','incremental_factor_and_peak']}
            rows=[]
            for frame,descriptor in enumerate(sequence):
                patches=descriptor['patches'];omega=np.asarray(descriptor['omega_rad_s'])
                start=perf_counter();old=load_old(f,patches,args.quad);timings['contact_mapping_old'].append(perf_counter()-start)
                start=perf_counter();contact,_=load_shared(geometry,patches);timings['contact_mapping_shared'].append(perf_counter()-start)
                np.testing.assert_allclose(contact,old,rtol=1e-12,atol=1e-13)
                _,loadchecks=load_shared(geometry,patches,validate=True)
                raw=contact-centrifugal_inertia(f,omega)
                rhs=f.balance(raw)
                # Independent call of the original full-body solver is an oracle.
                reference,reference_rhs,_=f.solve(raw)
                np.testing.assert_allclose(reference_rhs,rhs,rtol=0,atol=0)
                reference_peak=peak(f,reference)
                start=perf_counter();direct=np.zeros(f.ndof)
                direct[f.free]=f.factor.solve(rhs[f.free]);direct_peak=peak(f,direct)
                timings['cached_factor_and_peak'].append(perf_counter()-start)
                start=perf_counter()
                if not np.any(rhs):incremental=np.zeros(f.ndof)
                else:
                    incremental=previous_u.copy()
                    incremental[f.free]+=f.factor.solve((rhs-previous_rhs)[f.free])
                incremental_peak=peak(f,incremental)
                timings['incremental_factor_and_peak'].append(perf_counter()-start)
                checks={'direct':validate(f,direct,rhs,raw,reference_peak),
                        'incremental':validate(f,incremental,rhs,raw,reference_peak)}
                assert abs(direct_peak-reference_peak)/max(reference_peak,1.)<1e-7
                assert abs(incremental_peak-reference_peak)/max(reference_peak,1.)<1e-7
                rows.append({'frame':frame,'descriptor':descriptor,'contact_integration_checks':loadchecks,
                             'raw_unbalanced_wrench':(f.r.T@raw).tolist(),
                             'centrifugal_inertia_norm_N':float(np.linalg.norm(centrifugal_inertia(f,omega))),
                             'reference_peak_Pa':reference_peak,'checks':checks})
                previous_u=incremental;previous_rhs=rhs
                if name in ['random_unrestricted_patches','zero_unilateral_and_spin'] and patches:
                    reverse_candidates.append((name,frame,raw,reference))
            data['cases'][name]={'frames':len(sequence),'timings':{k:stats(v) for k,v in timings.items()},'rows':rows}
            print(json.dumps({'case':name,'frames':len(sequence),'timings':data['cases'][name]['timings']}),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2))
        # Pick a random asymmetric multi-contact frame and the unilateral frame.
        selected=[next(v for v in reverse_candidates if v[0]=='random_unrestricted_patches'),
                  next(v for v in reverse_candidates if v[0]=='zero_unilateral_and_spin')]
        for name,frame,raw,reference in selected:
            alternate,rhs,residual=f.solve(raw,alternate_gauge=True)
            canonical_error=float(np.linalg.norm(f.canonical(alternate)-f.canonical(reference))/max(np.linalg.norm(f.canonical(reference)),1e-30))
            peak_error=abs(peak(f,alternate)-peak(f,reference))/max(peak(f,reference),1.)
            assert canonical_error<1e-7 and peak_error<1e-7
            data['reverse_gauge_checks'].append({'case':name,'frame':frame,'canonical_displacement_relative_error':canonical_error,
                                                 'peak_relative_error_1Pa_floor':peak_error,'full_equilibrium_residual_1N_floor':residual,
                                                 'six_mode_balanced_wrench':(f.r.T@rhs).tolist()})
        # A modest load quadrature comparison does not establish mesh convergence.
        geometry_finer=SurfaceGeometry(f,args.quad+4);data['load_quadrature_checks']=[]
        for name,frame,_,_ in selected:
            descriptor=descriptors[name][frame]
            raw_low,_=load_shared(geometry,descriptor['patches'])
            raw_high,_=load_shared(geometry_finer,descriptor['patches'])
            inertia=centrifugal_inertia(f,np.asarray(descriptor['omega_rad_s']))
            ulow,_,_=f.solve(raw_low-inertia);uhigh,_,_=f.solve(raw_high-inertia)
            low=peak(f,ulow);high=peak(f,uhigh)
            data['load_quadrature_checks'].append({'case':name,'frame':frame,'orders':[args.quad,args.quad+4],
                                                   'peak_change_relative':abs(low-high)/max(high,1.),
                                                   'nodal_contact_load_change_relative':float(np.linalg.norm(raw_low-raw_high)/max(np.linalg.norm(raw_high),1e-30))})
    args.output.write_text(json.dumps(data,indent=2))
    print(json.dumps({'complete':str(args.output),'reverse_gauge_checks':data['reverse_gauge_checks']}),flush=True)


if __name__=='__main__':main()
