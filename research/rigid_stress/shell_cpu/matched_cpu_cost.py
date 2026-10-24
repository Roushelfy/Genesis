"""Matched authored-surface CPU cost, cached shell versus P2 volume solve.

The broad 5 mm disk is a declared footprint control, not Hertz auto-sizing.
The two discretizations and their errors are not equivalent. This isolates
CPU sparse-direct solve + identical count of common surface in-plane samples.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import numpy as np
from scipy import sparse
from .core import ShellModel
from .geometry import sphere_mesh
from .contacts import Contact, ShellDPRTransfer
from .solid_comparison import P2Shell, extrude, solid_disk_load, surface_sampling_gradients, surface_stress_at_shell_points


def solid_stress_operator(solid, shell, layers=1):
    elem, grad = surface_sampling_gradients(solid, shell.quads, layers, len(shell.vertices))
    axes = np.broadcast_to(shell.axes,(2,)+shell.axes.shape).reshape(-1,3,3)
    mu=solid.young/(2*(1+solid.poisson))
    lam=solid.young*solid.poisson/((1+solid.poisson)*(1-2*solid.poisson))
    pieces=[]
    for a,b in [(0,0),(1,1),(0,1)]:
        ea,eb=axes[:,:,a],axes[:,:,b]
        ega=np.einsum('ei,eni->en',ea,grad)
        egb=np.einsum('ei,eni->en',eb,grad)
        value=mu*(ea[:,None,:]*egb[:,:,None]+eb[:,None,:]*ega[:,:,None])
        if a==b:value+=lam*grad
        pieces.append(value)
    values=np.stack(pieces,axis=1).reshape(-1,30)
    dofs=solid.dofs[elem]
    rows=np.repeat(np.arange(len(values)),30)
    cols=np.broadcast_to(dofs[:,None,:],(len(dofs),3,30)).ravel()
    return sparse.coo_matrix((values.ravel(),(rows,cols)),shape=(len(values),solid.ndof)).tocsr()


def benchmark(n=4,batches=(1,32),repeats=5):
    v,q=sphere_mesh(n,.02)
    t=time.perf_counter()
    shell=ShellModel(v,q,E=1e10,nu=.3,h=.0005,directors=v/.02)
    sf=shell.factorize()
    shell_setup=time.perf_counter()-t
    xyz,tets,tri=extrude(v,q,.0005,1)
    t=time.perf_counter()
    solid=P2Shell(xyz,tets,tri,1e10,.3,1000.,1)
    solid_setup=time.perf_counter()-t
    t=time.perf_counter()
    Sm=shell.stress_matrix()
    Pm=solid_stress_operator(solid,shell)
    maps_seconds=time.perf_counter()-t
    direction=np.array([.93,.31,.19]);direction/=np.linalg.norm(direction)
    transfer=ShellDPRTransfer(shell)
    fs=np.zeros(shell.ndof);fp=np.zeros(solid.ndof)
    for sign in (1,-1):
        point=sign*.02025*direction
        force=-sign*.05*direction
        contact=Contact(point,force,sign*direction,(1/.02025,)*2,radius=.005)
        f,_=transfer.transfer(contact,order=24)
        fs+=f
        f,_=solid_disk_load(solid,tri,point,sign*direction,force,.005)
        fp+=f
    # Validate the solid cached stress operator against separate direct tensor
    # contraction before timing; do not use an operator check as physical GT.
    qp,_,rp=solid.solve(fp)
    direct=surface_stress_at_shell_points(solid,qp,q,1,len(v))
    local=np.einsum('egai,segab,egbj->segij',shell.axes,direct,shell.axes)
    direct_vec=local[:,:,:,[0,1,0],[0,1,1]].ravel()
    op_error=np.linalg.norm(Pm@qp-direct_vec)/np.linalg.norm(direct_vec)
    qs=shell.solve(fs)
    shellop=shell.stress(qs)
    shellvec=np.stack((shellop['local_top'],shellop['local_bottom'])).ravel()
    sm_error=np.linalg.norm(Sm@qs-shellvec)/np.linalg.norm(shellvec)
    rows=[]
    for B in batches:
        scales=np.linspace(.75,1.25,B)
        Fs=np.asfortranarray(fs[:,None]*scales)
        Fp=np.asfortranarray(fp[:,None]*scales)
        def ss():
            return shell.solve(Fs)
        def ps():
            rhs=solid.balance(Fp)
            out=np.zeros_like(rhs)
            out[solid.free]=solid.factor.solve(np.asfortranarray(rhs[solid.free]))
            return out
        def sa():return Sm@ss()
        def pa():return Pm@ps()
        # Warm caches, then alternate old/new order on every repeat.
        for fun in (ss,ps,sa,pa):fun()
        result={name:[] for name in ['shell_solve','solid_solve','shell_solve_stress','solid_solve_stress']}
        funcs=[('shell_solve',ss),('solid_solve',ps),('shell_solve_stress',sa),('solid_solve_stress',pa)]
        for repeat in range(repeats):
            for name,fun in funcs[::1 if repeat%2==0 else -1]:
                count=10 if B==1 else 3
                start=time.perf_counter()
                for _ in range(count):out=fun()
                result[name].append((time.perf_counter()-start)/count*1000)
        med={name:float(np.median(times)) for name,times in result.items()}
        rows.append(dict(batch=B,median_ms=med,repeats_ms=result,
                         solve_stress_ratio_solid_over_shell=med['solid_solve_stress']/med['shell_solve_stress'],
                         shell_rhs_shape=list(Fs.shape),solid_rhs_shape=list(Fp.shape)))
    return dict(n=n,shell_dofs=shell.ndof,solid_dofs=solid.ndof,quads=len(q),solid_tets=len(tets),
                common_inplane_stress_components=Sm.shape[0],shell_setup_factor_seconds=shell_setup,
                solid_setup_factor_seconds=solid_setup,stress_map_setup_seconds=maps_seconds,
                numerical_checks=dict(solid_stress_map_relative=op_error,shell_stress_map_relative=sm_error,
                                      solid_residual=rp,shell_residual=shell.residual(qs,fs)),rows=rows,
                scope='CPU single thread, same authored spherical surface/normal-offset wall; different Q4 shell and P2 affine solid geometry. Cached sparse LU + cached common-sample in-plane stress maps. No transfer, assembly/factor, 3D full-volume output or GPU/Franka timing included; RHS amplitudes vary at fixed authored patch locations.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',nargs='+',type=int,default=[4,8])
    p.add_argument('--output',type=Path,default=Path('shell_cpu/solid_results/matched_cpu_cost.json'))
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in args.n:
        row=benchmark(n)
        rows.append(row)
        cpu=platform.processor()
        if not cpu and Path('/proc/cpuinfo').exists():
            cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), 'unknown')
        args.output.write_text(json.dumps(dict(cpu=cpu,threads=1,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows),indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
