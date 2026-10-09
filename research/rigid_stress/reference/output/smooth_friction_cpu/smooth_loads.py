#!/usr/bin/env python3
"""Synthetic smooth friction-admissible snapshots on a fixed full egg shell.

API:
    cases = prepare_cases(f, g, frames=96)
    traction, checks, phase = assemble_descriptor(g, cases[name][frame])

``f`` is full-body eggshell_convergence_cpu.FEM(..., order=2, quarter=False).
``g`` is arbitrary_contact_test.SurfaceGeometry(f, quad), with quadrature
coordinates shaped (faces, quadrature_points, 3). Returned traction is in
N/m^2 and has shape (faces * quadrature_points, 3), in that same order.
Use g.integration_weights to integrate or consistently assemble it.

These are prescribed friction-cone-admissible traction snapshots, NOT a
contact/Coulomb dynamics solve or a measured physically consistent motion.
Velocity declarations only check tangential direction/dissipation and cone
boundary examples. Root code adds gravity/inertia and does nodal assembly.
There is no fixed load/stress basis, contact symmetry, or selected hotspot.
"""
import numpy as np


def quintic_ramp(t, begin, end):
    """C2 step, exactly zero before begin and exactly one after end."""
    x=float(np.clip((t-begin)/(end-begin),0.,1.))
    return x*x*x*(10.+x*(-15.+6.*x))


def _outward_normals(g):
    if not hasattr(g,'_smooth_friction_outward_normals'):
        f=g.f
        assert f.order==2 and not f.quarter
        vertices=f.base_xyz[f.outer_faces]
        normal=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        reverse=np.einsum('fi,fi->f',normal,vertices.mean(axis=1)-f.com)<0
        normal[reverse]*=-1
        assert np.max(np.abs(np.linalg.norm(normal,axis=1)-1))<1e-13
        g._smooth_friction_outward_normals=normal[:,None,:]
    return g._smooth_friction_outward_normals


def _egg_center_frame(longitude,latitude,twist=0.):
    """Analytic outer egg curve; finite pressure intersects the fixed faceted FEM skin."""
    unit=np.array([np.cos(latitude)*np.cos(longitude),np.cos(latitude)*np.sin(longitude),np.sin(latitude)])
    scale=1-.18*unit[2]
    center=unit*np.array([.022*scale,.022*scale,.030])
    normal=np.array([2*unit[0]/(.022*scale),2*unit[1]/(.022*scale),
                     2/.030*(unit[2]+.18*(unit[0]**2+unit[1]**2)/scale)])
    normal/=np.linalg.norm(normal)
    azimuth=np.array([-np.sin(longitude),np.cos(longitude),0.])
    azimuth-=normal*(azimuth@normal);azimuth/=np.linalg.norm(azimuth)
    meridian=np.cross(normal,azimuth)
    direction=np.cos(twist)*azimuth+np.sin(twist)*meridian
    return center,direction


def _phase(t):
    if t<.08:return 'approach'
    if t<.28:return 'clamp'
    if t<.48:return 'lift'
    if t<.78:return 'hold'
    return 'release'


def _smooth_descriptor(t,slide_case=False):
    settings=[(.73,.25,.08,.20,.85,.98,4.),
              (-2.05,-.35,.14,.26,.82,.95,3.5),
              (2.30,.92,.38,.50,.70,.82,1.3)]
    slide_blend=quintic_ramp(t,.46,.56)*(1.-quintic_ramp(t,.67,.76)) if slide_case else 0.
    lift=quintic_ramp(t,.28,.48)
    release_signal=quintic_ramp(t,.78,.96)
    contacts=[]
    for j,(lon0,lat0,begin,clamped,release,released,force) in enumerate(settings):
        envelope=quintic_ramp(t,begin,clamped)*(1.-quintic_ramp(t,release,released))
        longitude=lon0+.13*np.sin(2*np.pi*t+.7*j)
        latitude=lat0+.08*np.sin(2*np.pi*t+.3+.9*j)
        # The lift phase gradually adds an upward tangential traction tendency;
        # this is a load prescription, without imposing body force balance.
        center,direction=_egg_center_frame(longitude,latitude,.15*j+.12*np.sin(2*np.pi*t)-.9*lift)
        interior=.10+(.24+.03*np.sin(2*np.pi*t+.4*j))*lift*(1-release_signal)
        fraction=(1-slide_blend)*interior+slide_blend
        speed=slide_blend*(.004+.001*np.sin(2*np.pi*t+j))
        normal_force=force*envelope*(1.+.08*np.sin(2*np.pi*t+.5*j))
        contacts.append({'id':j,'center_m':center.tolist(),
                         'radius_m':float(.006+.0008*np.sin(2*np.pi*t+.8*j)),
                         'normal_pressure_integral_N':float(normal_force),
                         'tangent_direction_material':direction.tolist(),
                         'friction_fraction':float(fraction),
                         'declared_relative_velocity_m_s':(speed*direction).tolist(),
                         'friction_regime':'cone_boundary_slide' if slide_blend==1. else
                                           ('stick_cone_interior' if slide_blend==0. else 'smooth_interpolation')})
    return {'time_parameter':float(t),'phase':_phase(t),'mu_f':.6,'contacts':contacts,
            'phase_signals':{'lift_quintic':lift,'release_quintic':release_signal,'slide_blend_quintic':slide_blend},
            'active_contact_count':sum(c['normal_pressure_integral_N']>0. for c in contacts),
            'coordinate_frame':'body material/reference frame',
            'velocity_scope':'Declared admissible snapshots only; not a Coulomb or physical trajectory solve.',
            'center_geometry':'Analytic egg centers; C-infinity compact pressure restricted to fixed faceted outer surface.'}


def prepare_cases(f,g,frames=96):
    """Return compact deterministic descriptors; no dense-frame storage or solves."""
    assert g.f is f and frames>=8
    _outward_normals(g)
    t=np.linspace(0.,1.,frames)
    cases={'smooth_stick':[_smooth_descriptor(v,False) for v in t],
           'smooth_stick_slide_release':[_smooth_descriptor(v,True) for v in t]}
    rng=np.random.default_rng(20261009);abrupt=[]
    for i in range(16):
        contacts=[]
        for j in range(int(rng.integers(1,4))):
            longitude=rng.uniform(-np.pi,np.pi);latitude=np.arcsin(rng.uniform(-.85,.85))
            center,direction=_egg_center_frame(longitude,latitude,rng.uniform(-np.pi,np.pi))
            fraction=1. if (i+j)%3==0 else rng.uniform(.05,.8)
            speed=.005 if fraction==1. else 0.
            contacts.append({'id':j,'center_m':center.tolist(),'radius_m':float(rng.uniform(.0045,.0075)),
                             'normal_pressure_integral_N':float(rng.uniform(.3,5.)),
                             'tangent_direction_material':direction.tolist(),'friction_fraction':float(fraction),
                             'declared_relative_velocity_m_s':(speed*direction).tolist(),
                             'friction_regime':'cone_boundary_slide' if fraction==1. else 'stick_cone_interior'})
        abrupt.append({'time_parameter':i/15.,'phase':'abrupt_control','mu_f':.6,'contacts':contacts,
                       'active_contact_count':len(contacts),'coordinate_frame':'body material/reference frame',
                       'velocity_scope':'Declared cone-admissible snapshots only; not a dynamics solve.'})
    cases['abrupt_control']=abrupt
    return cases


def assemble_descriptor(g,descriptor):
    """Return (flat traction [Pa], pointwise/integral checks, phase).

    Normal compression is -p*n. Each tangential direction is projected into
    the actual fixed face tangent plane. All patches share mu, hence their
    sum also obeys ||tau_total|| <= mu * p_total, even when they overlap.
    The integrals below let the nodal mapper verify both force and torque.
    """
    normals=_outward_normals(g);mu=float(descriptor['mu_f']);assert mu>=0
    coords=g.coords;integration=g.integration_weights
    traction=np.zeros_like(coords);total_p=np.zeros(coords.shape[:2]);patch_checks=[]
    dissipation=0.
    for contact in descriptor['contacts']:
        force=float(contact['normal_pressure_integral_N']);radius=float(contact['radius_m'])
        fraction=float(contact['friction_fraction']);assert force>=0 and radius>0 and 0<=fraction<=1
        if force==0:continue
        center=np.asarray(contact['center_m']);d2=np.sum((coords-center)**2,axis=2)
        s=d2/radius**2;inside=s<1.
        kernel=np.zeros_like(s)
        # C-infinity compact bump: every derivative vanishes at its support edge.
        kernel[inside]=np.exp(-.5*d2[inside]/(.45*radius)**2-s[inside]/(1.-s[inside]))
        integral=float(np.sum(kernel*integration));assert integral>0,('Unsampled finite footprint',contact)
        pressure=force*kernel/integral
        direction=np.asarray(contact['tangent_direction_material'])
        tangent=direction-normals*np.sum(normals*direction,axis=2,keepdims=True)
        length=np.linalg.norm(tangent,axis=2,keepdims=True)
        tangent=np.divide(tangent,length,out=np.zeros_like(tangent),where=length>1e-14)
        tau=-mu*fraction*pressure[:,:,None]*tangent
        patch_traction=-pressure[:,:,None]*normals+tau
        traction+=patch_traction;total_p+=pressure
        velocity=np.asarray(contact['declared_relative_velocity_m_s'])
        tangential_velocity=velocity-normals*np.sum(normals*velocity,axis=2,keepdims=True)
        work=np.sum(tau*tangential_velocity,axis=2)
        assert float(work.max())<=1e-10*max(float(pressure.max()),1.)
        dissipation+=float(np.sum(work*integration))
        F=np.einsum('fq,fqc->c',integration,patch_traction)
        torque=np.einsum('fq,fqc->c',integration,np.cross(coords,patch_traction))
        pressure_force=float(np.sum(pressure*integration))
        assert abs(pressure_force-force)<1e-11*max(force,1.)
        patch_checks.append({'id':contact['id'],'pressure_integral_N':pressure_force,
                             'integrated_force_N':F.tolist(),'integrated_torque_Nm':torque.tolist(),
                             'friction_fraction':fraction,'friction_regime':contact['friction_regime'],
                             'support_quadrature_points':int(np.count_nonzero(pressure)),
                             'support_faces':int(np.count_nonzero(pressure.sum(axis=1))),
                             'pressure_centroid_m':(np.einsum('fq,fqc->c',pressure*integration,coords)/pressure_force).tolist()})
    tau_total=traction+total_p[:,:,None]*normals
    tangency=float(np.max(np.abs(np.sum(tau_total*normals,axis=2))))
    violation=float(np.max(np.linalg.norm(tau_total,axis=2)-mu*total_p))
    scale=max(float(total_p.max()),1.)
    assert np.isfinite(traction).all() and total_p.min()>=0
    assert tangency<1e-10*scale and violation<1e-10*scale,(tangency,violation,scale)
    F=np.einsum('fq,fqc->c',integration,traction)
    torque=np.einsum('fq,fqc->c',integration,np.cross(coords,traction))
    checks={'pressure_min_Pa':float(total_p.min()),'pressure_max_Pa':float(total_p.max()),
            'tangency_max_absolute_Pa':tangency,'tangency_relative_to_peak_pressure':tangency/scale,
            'cone_max_positive_violation_Pa':max(0.,violation),
            'cone_violation_relative_to_peak_pressure':max(0.,violation)/scale,
            'declared_friction_work_rate_W':dissipation,
            'integrated_force_N':F.tolist(),'integrated_torque_Nm':torque.tolist(),
            'active_contact_count':len(patch_checks),'patches':patch_checks,
            'normal_compression_sign':'traction_normal=-p*outward_normal',
            'snapshot_scope':'Friction-cone/dissipation-admissible prescription; not a solved Coulomb trajectory.'}
    return np.ascontiguousarray(traction.reshape(-1,3)),checks,descriptor['phase']
