"""Independent CPU axisymmetric elastic-halfspace unilateral contact control.

The contact support and pressure are unknown. A ring BEM solves prescribed
force against a rigid paraboloid, enforcing nonnegative pressure and gap.
Hertz enters only after the solve, as an analytical comparison. This is an
offline halfspace control; it does not establish thin-shell contact accuracy.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import ellipk
from scipy.linalg import solve


def compliance_matrix(n, domain, modulus, order=24):
    """Midpoint collocation and constant-pressure rings, SI units.

    Gauss integration splits each singular diagonal ring at its collocation
    point. No singular point is sampled; this independently integrates the
    logarithmic kernel rather than inserting a Hertz-derived diagonal.
    """
    edges=np.linspace(0.,domain,n+1)
    radius=.5*(edges[:-1]+edges[1:])
    x,w=np.polynomial.legendre.leggauss(order)
    s=edges[:-1,None]+.5*(x[None,:]+1.)*np.diff(edges)[:,None]
    weights=.5*np.diff(edges)[:,None]*w
    r=radius[:,None,None]
    argument=4*r*s[None,:,:]/(r+s[None,:,:])**2
    kernel=s[None,:,:]/(r+s[None,:,:])*ellipk(np.minimum(argument,1.-np.finfo(float).eps))
    matrix=np.einsum('ijq,jq->ij',kernel,weights)
    # Redo diagonal by split quadrature. The logarithmic endpoint lies outside
    # the open Gauss nodes on both subintervals.
    for i,ri in enumerate(radius):
        value=0.
        for low,high in ((edges[i],ri),(ri,edges[i+1])):
            sj=low+.5*(x+1)*(high-low)
            arg=4*ri*sj/(ri+sj)**2
            # Remove the known local logarithmic singularity. Its analytical
            # integral is independent of any contact-pressure solution.
            singular=.5*np.log(8*ri/np.abs(sj-ri))
            regular=sj/(ri+sj)*ellipk(np.minimum(arg,1.-np.finfo(float).eps))-singular
            length=high-low
            value+=.5*length*np.dot(w,regular)
            value+=.5*length*(np.log(8*ri/length)+1.)
        matrix[i,i]=value
    matrix*=4/(np.pi*modulus)
    area=np.pi*(edges[1:]**2-edges[:-1]**2)
    return radius,edges,area,matrix


def solve_contact(force, radius, area, matrix, indenter_radius,
                  max_iterations=100):
    """Primal/dual active set for p>=0, gap>=0 and area.p=force.

    The contact set is allowed to vary over all rings. A contiguous initial
    set uses a fixed fraction of the independently prescribed domain; no
    Hertz support, analytical pressure, or fitted radius is supplied.
    Scaling keeps the saddle solve reasonably conditioned in SI units.
    """
    n=len(radius)
    gap0=radius**2/(2*indenter_radius)
    a_scale=area.sum()
    p_scale=force/a_scale
    u_scale=np.max(matrix)*p_scale*n
    aa=matrix*p_scale/u_scale
    gg=gap0/u_scale
    areas=area/a_scale
    active=np.arange(max(2,n//4))
    visited=[]
    pressure=np.zeros(n)
    delta=0.
    for iteration in range(max_iterations):
        m=len(active)
        system=np.empty((m+1,m+1))
        system[:m,:m]=aa[np.ix_(active,active)]
        system[:m,m]=-1.
        system[m,:m]=areas[active]
        system[m,m]=0.
        rhs=np.r_[-gg[active],1.]
        answer=solve(system,rhs,assume_a='gen',check_finite=False)
        pressure[:]=0.
        pressure[active]=answer[:m]
        delta=answer[m]
        gap=gg+aa@pressure-delta
        negative=active[pressure[active] < -1e-9]
        inactive=np.setdiff1d(np.arange(n),active,assume_unique=True)
        violated=inactive[gap[inactive] < -1e-10]
        visited.append(dict(iteration=iteration,active=m,
                            min_pressure_scaled=float(pressure.min()),
                            min_gap_scaled=float(gap.min())))
        if not len(negative) and not len(violated):
            break
        if len(negative):
            # Remove negative primal variables before admitting new dual
            # violations. The domain is not assumed to have a known radius.
            active=np.setdiff1d(active,negative,assume_unique=True)
            if not len(active):
                active=np.array([int(np.argmin(gap0))])
        else:
            active=np.union1d(active,violated)
    else:
        raise RuntimeError('Active set did not converge: '+json.dumps(visited[-8:]))
    p=pressure*p_scale
    delta*=u_scale
    displacement=matrix@p
    gap=gap0+displacement-delta
    return p,delta,gap,visited


def displacement_at_points(query, edges, pressure, modulus, order=48):
    """Independent between-collocation displacement check, loaded rings only."""
    x,w=np.polynomial.legendre.leggauss(order)
    loaded=np.flatnonzero(pressure>0)
    out=[]
    for ri in query:
        if ri==0:
            out.append(2/modulus*np.dot(np.diff(edges),pressure))
            continue
        value=0.
        for j in loaded:
            low,high=edges[j],edges[j+1]
            intervals=[(low,high)]
            singular=low <= ri <= high
            if low < ri < high:
                intervals=[(low,ri),(ri,high)]
            integral=0.
            for a,b in intervals:
                sj=a+.5*(x+1)*(b-a)
                arg=4*ri*sj/(ri+sj)**2
                kernel=sj/(ri+sj)*ellipk(np.minimum(arg,1.-np.finfo(float).eps))
                if singular:
                    kernel-=.5*np.log(8*ri/np.abs(sj-ri))
                integral+=.5*(b-a)*np.dot(w,kernel)
                if singular:
                    # Integral of .5log(8r/|s-r|), on one side of r.
                    da,db=abs(a-ri),abs(b-ri)
                    primitive=lambda z: 0. if z==0 else .5*z*(np.log(8*ri/z)+1.)
                    integral+=abs(primitive(db)-primitive(da))
            value+=pressure[j]*integral
        out.append(4/(np.pi*modulus)*value)
    return np.asarray(out)


def run_case(n=320, force=.05, domain=.0003, E=1e10, nu=.3,
             indenter_radius=.02, order=24):
    start=time.perf_counter()
    modulus=E/(1-nu*nu)
    r,edges,area,matrix=compliance_matrix(n,domain,modulus,order)
    assembly=time.perf_counter()-start
    then=time.perf_counter()
    p,delta,gap,history=solve_contact(force,r,area,matrix,indenter_radius)
    solve_seconds=time.perf_counter()-then
    active=np.flatnonzero(p>max(p.max(),1.)*1e-10)
    numerical_radius=edges[active[-1]+1]
    # Analytical comparison is deliberately performed after the independent
    # contact solve, and never participates in its active-set initialization.
    hertz_radius=(3*force*indenter_radius/(4*modulus))**(1/3)
    hertz_delta=hertz_radius**2/indenter_radius
    hertz_p=3*force/(2*np.pi*hertz_radius**2)*np.sqrt(np.maximum(0.,1-r*r/hertz_radius**2))
    pressure_error=np.sqrt(np.dot(area,(p-hertz_p)**2)/np.dot(area,hertz_p**2))
    # A second error measure integrates the discontinuous ring pressure against
    # the continuous analytical pressure, including within-ring variation.
    p0=3*force/(2*np.pi*hertz_radius**2)
    low=np.minimum(edges[:-1],hertz_radius)
    high=np.minimum(edges[1:],hertz_radius)
    hertz_force_rings=2*np.pi*p0*hertz_radius**2/3 * (
        np.maximum(0.,1-low**2/hertz_radius**2)**1.5
        -np.maximum(0.,1-high**2/hertz_radius**2)**1.5)
    hertz_squared_rings=np.pi*p0*p0*((high**2-low**2)
                          -(high**4-low**4)/(2*hertz_radius**2))
    continuous_squared=np.sum(p*p*area-2*p*hertz_force_rings+hertz_squared_rings)
    continuous_error=np.sqrt(max(0.,continuous_squared)/(.5*np.pi*p0*p0*hertz_radius**2))
    relative_force=abs(area@p-force)/force
    max_p=max(p.max(),1.)
    max_u=max(delta,1e-30)
    comp=float(np.max(np.abs(p*gap))/(max_p*max_u))
    # This separate fine sample exposes the finite-ring discretization error;
    # it is not used to modify the converged collocation solution.
    query=np.linspace(0.,min(domain,2*numerical_radius),2*len(active)+3)
    query_gap=query**2/(2*indenter_radius)+displacement_at_points(query,edges,p,modulus)-delta
    result=dict(n=n,force_N=force,domain_m=domain,indenter_radius_m=indenter_radius,
                E_Pa=E,nu=nu,reduced_modulus_Pa=modulus,quadrature_order=order,
                contact_radius_m=float(numerical_radius),hertz_radius_m=float(hertz_radius),
                radius_relative_difference=float(abs(numerical_radius-hertz_radius)/hertz_radius),
                indentation_m=float(delta),hertz_indentation_m=float(hertz_delta),
                indentation_relative_difference=float(abs(delta-hertz_delta)/hertz_delta),
                pressure_relative_area_L2=float(pressure_error),
                pressure_relative_continuous_L2=float(continuous_error),
                force_relative_residual=float(relative_force),
                minimum_pressure_Pa=float(p.min()),minimum_gap_m=float(gap.min()),
                between_nodes_minimum_gap_m=float(query_gap.min()),
                between_nodes_penetration_over_indentation=float(max(0.,-query_gap.min())/max_u),
                complementarity_scaled=comp,active_rings=int(len(active)),
                outer_domain_loaded=bool(active[-1]==n-1),
                iterations=len(history),assembly_seconds=assembly,solve_seconds=solve_seconds,
                passed_numerical=bool(relative_force<1e-9 and p.min()>-1e-6*max_p
                                  and gap.min()>-1e-8*max_u and comp<1e-8
                                  and active[-1]<n-1),history=history)
    return result,dict(radius=r,edges=edges,area=area,pressure=p,gap=gap,hertz_pressure=hertz_p)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='shell_cpu/contact_reference_results')
    parser.add_argument('--n',nargs='+',type=int,default=[160,320,640])
    parser.add_argument('--forces',nargs='+',type=float,default=[.005,.05,.5])
    parser.add_argument('--domain',type=float,default=.0003)
    parser.add_argument('--order',type=int,default=24)
    args=parser.parse_args()
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Independent axisymmetric halfspace contact; no imposed Hertz pressure/support. '
                      'This does not validate a thin-shell contact radius.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=[])
    for force in args.forces:
        for n in args.n:
            row,arrays=run_case(n,force,domain=args.domain,order=args.order)
            report['rows'].append(row)
            np.savez_compressed(folder/f'force_{force:g}_n{n}.npz',**arrays)
            (folder/'results.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    print('complete',flush=True)


if __name__=='__main__':
    main()
