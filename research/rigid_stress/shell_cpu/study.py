"""Egg/torus CPU study of automatic-radius directional shell recovery.

This tests prescribed rigid contact loads; it is not a coupled contact solver.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shell_cpu.study.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .core import ShellModel, quad_shape, element_operators
from .geometry import egg_mesh, egg_geometry
from .contacts import Contact, ShellDPRTransfer

AXES = (.022, .022, .030)
ASYM = -.18
H = .0005


def egg_point(direction):
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    scale = 1+ASYM*unit[2]
    return unit*np.array([AXES[0]*scale, AXES[1]*scale, AXES[2]])


def make_contact(direction, force_n=.05, shear_fraction=0., radius=None):
    point = egg_point(direction)
    normal, curvature = egg_geometry(point, axes=AXES, asym=ASYM)
    normal, curvature = normal[0], curvature[0]
    tangent = np.array([0.,0.,1.])-normal[2]*normal
    tangent /= np.linalg.norm(tangent)
    force = -force_n*normal+shear_fraction*force_n*tangent
    return Contact(point, force, normal, curvature, radius=radius, friction=.6)


def contact_cases():
    # Geometric extremum, so diametric compressive normals are collinear.
    z=(-1+np.sqrt(1+8*ASYM*ASYM))/(4*ASYM)
    directions=[(np.sqrt(1-z*z),0,z),(-np.sqrt(1-z*z),0,z)]
    return {
        'automatic_normal': [make_contact(d) for d in directions],
        'automatic_friction': [make_contact(d,shear_fraction=.25) for d in directions],
        'manual_2mm_normal': [make_contact(d,radius=.002) for d in directions],
        'automatic_asymmetric': [make_contact(d, fn, sh) for d,fn,sh in
            [((1,.2,.1),.035,.2),((-1,.1,-.15),.05,-.15),((.2,-1,.25),.02,.3)]],
    }


def field_metrics(model,q):
    s=model.stress(q)
    local=np.stack((s['local_top'],s['local_bottom']))
    x,y,t=local[...,0],local[...,1],local[...,2]
    vm=np.sqrt(np.maximum(x*x-x*y+y*y+3*t*t,0.))
    tensile=.5*(x+y)+np.sqrt(.25*(x-y)**2+t*t)
    tensors=np.stack((s['sigma_top'],s['sigma_bottom']))
    return s,tensors,vm,tensile


def qp_positions(model):
    return np.stack([np.einsum('i,eij->ej',quad_shape(*g)[0],model.vertices[model.quads])
                     for g in model.gauss_points],axis=1)


def evaluate_chart_probes(model,q,directions):
    """Evaluate strain operators at common cube-chart coordinates, no nearest-QP sampling."""
    n=int(round(np.sqrt(len(model.quads)/6)))
    frames=[([1,0,0],[0,1,0],[0,0,1]),([-1,0,0],[0,-1,0],[0,0,1]),
            ([0,1,0],[-1,0,0],[0,0,1]),([0,-1,0],[1,0,0],[0,0,1]),
            ([0,0,1],[1,0,0],[0,1,0]),([0,0,-1],[-1,0,0],[0,1,0])]
    result=[]
    for direction in directions:
        axis=int(np.argmax(np.abs(direction)))
        face=2*axis+int(direction[axis]<0)
        normal,first,second=map(np.asarray,frames[face])
        ratio=direction@normal
        uv=np.array([direction@first,direction@second])/ratio
        xy=(uv+1)*n/2
        ij=np.minimum(np.floor(xy).astype(int),n-1)
        xi,eta=2*(xy-ij)-1
        e=face*n*n+ij[1]*n+ij[0]
        ids=model.quads[e]
        bm,bb,_bs,_bd,axes,_jac=element_operators(model.vertices[ids],model.directors[ids],xi,eta)
        qe=q[model.element_dofs[e]]
        pair=[]
        for side in (1,-1):
            sig=model.C@(bm+side*.5*model.h*bb)@qe
            local=np.array([[sig[0],sig[2],0],[sig[2],sig[1],0],[0,0,0]])
            pair.append(axes@local@axes.T)
        result.append(pair)
    return np.array(result).transpose(1,0,2,3)


def run_egg(resolutions=(3,5,8,12,16), output=Path('shell_cpu/results/egg')):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    previous={}
    cases=contact_cases()
    probe_directions=np.array([[np.cos(t)*np.sin(p),np.sin(t)*np.sin(p),np.cos(p)]
        for p in np.linspace(.25,np.pi-.25,9) for t in np.linspace(0,2*np.pi,24,endpoint=False)])
    probes=np.array([egg_point(d) for d in probe_directions])
    for n in resolutions:
        outer,quads=egg_mesh(n,axes=AXES,asym=ASYM)
        directors,_=egg_geometry(outer,axes=AXES,asym=ASYM)
        start=perf_counter()
        model=ShellModel(outer-.5*H*directors,quads,E=1e10,nu=.3,h=H,directors=directors)
        assemble=perf_counter()-start
        model.factorize()
        transfer=ShellDPRTransfer(model)
        positions=qp_positions(model)
        tree=cKDTree(positions.reshape(-1,3))
        _,probe_indices=tree.query(probes)
        for name,contacts in cases.items():
            row=dict(n=n,case=name,nodes=len(outer),elements=len(quads),dofs=model.ndof,
                     assembly_s=assemble,factor_s=model.factorize().seconds)
            try:
                start=perf_counter()
                force,info=transfer.transfer_many(contacts,order=16)
                row['transfer_s']=perf_counter()-start
                start=perf_counter()
                q=model.solve(force)
                row['solve_s']=perf_counter()-start
                start=perf_counter()
                stress,tensors,vm,tensile=field_metrics(model,q)
                row['stress_s']=perf_counter()-start
                row.update(valid=True,residual=model.residual(q,force),
                    peak_vm_pa=float(vm.max()),peak_tensile_pa=float(tensile.max()),
                    max_displacement_m=float(np.linalg.norm(q.reshape(-1,6)[:,:3],axis=1).max()),
                    max_rotation=float(np.linalg.norm(q.reshape(-1,6)[:,3:],axis=1).max()),
                    strain_energy_j=float(.5*q@(model.K@q)),
                    radii_m=[i['radius_m'] for i in info],
                    max_force_error=max(i['force_error'] for i in info),
                    max_normalized_moment_error=max(i['moment_error'] for i in info),
                    a_over_h=[i['a_over_h'] for i in info],
                    generalized_raw_resultant=(model.rigid_modes.T@force).tolist(),
                    relief_acceleration=model.last_relief.tolist(),contacts=info)
                samples=evaluate_chart_probes(model,q,probe_directions)
                if name in previous:
                    old,oldpeak,oldenergy=previous[name]
                    row['adjacent_probe_tensor_change']=float(np.linalg.norm(samples-old)/max(np.linalg.norm(samples),1e-30))
                    row['adjacent_peak_change']=float(abs(row['peak_vm_pa']-oldpeak)/max(row['peak_vm_pa'],1e-30))
                    row['adjacent_energy_change']=float(abs(row['strain_energy_j']-oldenergy)/max(row['strain_energy_j'],1e-30))
                previous[name]=(samples.copy(),row['peak_vm_pa'],row['strain_energy_j'])
                np.savez_compressed(output/f'{name}_n{n}.npz',vertices=model.vertices,quads=quads,
                    directors=directors,displacement=q,force=force,positions=positions,
                    sigma_top=stress['sigma_top'],sigma_bottom=stress['sigma_bottom'],vm=vm,
                    tensile=tensile,N=stress['N'],M=stress['M'],Q=stress['Q'])
            except Exception as error:
                row.update(valid=False,error=f'{type(error).__name__}: {error}')
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('contacts',)},default=lambda x:x.tolist()),flush=True)
        (output/'summary.json').write_text(json.dumps(dict(
            scope='Fixed linear shell under supplied contact loads, not an independent contact solution or 3D stress qualification.',
            shell_geometry='Midsurface is half-thickness normal offset from authored outer egg.',
            axes_m=AXES,taper=ASYM,thickness_m=H,E_pa=1e10,nu=.3,
            transfer='Projected Hertz disk along resultant force; differs from native surface-ball DPR on curved surfaces.',
            stress='Top/bottom in-plane membrane+bending stress at four Gauss points per quad; transverse Q reported separately.',
            convergence='Probe differences evaluate each mesh at common cube-chart coordinates; geometry approximation remains mesh-dependent. Peak is four-point sampled, not continuum maximum.',
            rows=rows),indent=2,default=lambda x:x.tolist()))
    return rows


def run_torus(output=Path('shell_cpu/results/egg')):
    R,r=.025,.008
    nu,nv=24,12
    nodes=[]
    directors=[]
    for j in range(nv):
        p=2*np.pi*j/nv
        for i in range(nu):
            t=2*np.pi*i/nu
            nodes.append([(R+r*np.cos(p))*np.cos(t),(R+r*np.cos(p))*np.sin(t),r*np.sin(p)])
            directors.append([np.cos(p)*np.cos(t),np.cos(p)*np.sin(t),np.sin(p)])
    nodes=np.array(nodes);directors=np.array(directors)
    quads=np.array([[j*nu+i,j*nu+(i+1)%nu,((j+1)%nv)*nu+(i+1)%nu,((j+1)%nv)*nu+i]
                    for j in range(nv) for i in range(nu)])
    model=ShellModel(nodes,quads,h=H,directors=directors)
    transfer=ShellDPRTransfer(model)
    # Concave inner ring of torus against a convex 5-mm-radius probe.
    normal=np.array([-1.,0,0]);point=np.array([R-r-.5*H,0,0])
    combined=np.array([1/r,-1/(R-r)])+1/.005
    contact=Contact(point,np.array([.03,0,.006]),normal,combined,friction=.6)
    force,record=transfer.transfer(contact,order=16)
    q=model.solve(force)
    stress,tensors,vm,tensile=field_metrics(model,q)
    unsupported=False
    try:
        transfer.transfer(replace(contact,curvatures=np.array([1/r,-1/(R-r)])))
    except ValueError:
        unsupported=True
    result=dict(valid=True,geometry='Nonconvex torus inner ring',dofs=model.ndof,
        combined_curvatures_per_m=combined.tolist(),contact=record,
        negative_relative_curvature_rejected=unsupported,residual=model.residual(q,force),
        peak_vm_pa=float(vm.max()),peak_tensile_pa=float(tensile.max()))
    np.savez_compressed(output/'nonconvex_torus.npz',vertices=nodes,quads=quads,positions=qp_positions(model),
                        sigma_top=stress['sigma_top'],sigma_bottom=stress['sigma_bottom'],vm=vm,tensile=tensile)
    (output/'nonconvex_torus.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolutions',default='3,5,8,12,16')
    parser.add_argument('--out',default='shell_cpu/results/egg')
    parser.add_argument('--torus',action='store_true')
    args=parser.parse_args()
    run_egg(tuple(map(int,args.resolutions.split(','))),Path(args.out))
    if args.torus:
        run_torus(Path(args.out))
