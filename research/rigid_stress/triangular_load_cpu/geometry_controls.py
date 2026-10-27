"""Check faceted loading on a fixed nonconvex torus shell."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.rigid_stress.shell_cpu.core import ShellModel

from .proxy import TriangularLoadTransfer
from .reference import Q4LoadTransfer
from .study import field_errors, physical_contact


def torus_model():
    """Build a low-resolution closed torus with analytic material directors."""
    major_radius, minor_radius = 0.025, 0.008
    around_count, section_count = 16, 8
    vertices, directors = [], []
    for section in range(section_count):
        polar = 2 * np.pi * section / section_count
        for around in range(around_count):
            azimuth = 2 * np.pi * around / around_count
            vertices.append(((major_radius + minor_radius*np.cos(polar))*np.cos(azimuth),
                             (major_radius + minor_radius*np.cos(polar))*np.sin(azimuth),
                             minor_radius*np.sin(polar)))
            directors.append((np.cos(polar)*np.cos(azimuth),np.cos(polar)*np.sin(azimuth),np.sin(polar)))
    quads = np.array([[section*around_count+around,section*around_count+(around+1)%around_count,
                       ((section+1)%section_count)*around_count+(around+1)%around_count,
                       ((section+1)%section_count)*around_count+around]
                      for section in range(section_count) for around in range(around_count)])
    return ShellModel(np.array(vertices), quads, directors=np.array(directors), E=1e10, nu=.3,
                      h=.0005, density=2000., gauge='translation')


def run_geometry_controls(output_dir):
    """Save same-mother field comparisons for prescribed finite nonconvex footprints."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model = torus_model()
    model.factorize()
    reference = Q4LoadTransfer(model)
    proxies = {count: TriangularLoadTransfer(model, count) for count in (4, 8, 16)}
    # The inner-ring faces have negative azimuthal curvature. All radii are prescribed.
    definitions = [
        ('inner-outside-normal', 4*16+1, (.23,-.17), 1, 0., .0005),
        ('inner-outside-friction', 4*16+1, (.23,-.17), 1, .3, .001),
        ('inner-inside-friction', 4*16+6, (-.31,.23), -1, -.25, .001),
        ('inner-material-edge', 4*16+9, (1.,.07), 1, .2, .0005),
        ('inner-triangle-seam', 4*16+12, (.07,.07), 1, -.2, .001),
        ('outer-outside-friction', 2, (.19,-.27), 1, .3, .0005),
        ('outer-inside-friction', 8, (-.31,.23), -1, .3, .001),
        ('top-friction', 2*16+4, (.13,-.29), 1, -.2, .001),
    ]
    cases, rows = [], []
    for name, face, coordinate, side, shear, radius in definitions:
        contact = physical_contact(model, face, coordinate, side=side, shear=shear, radius=radius)
        cases.append(dict(name=name,face=int(face),coordinate=list(coordinate),side=side,radius_m=radius,
                          point=contact.point.tolist(),force=contact.force.tolist(),normal=contact.normal.tolist()))
        try:
            reference_force, reference_info = reference.transfer(contact, order=48)
            lower_force, lower_info = reference.transfer(contact, order=24)
        except ValueError as error:
            rows.append(dict(case=name,method='Q4',status='reference_rejected',error=str(error)))
            continue
        reference_displacement = model.solve(reference_force)
        reference_field = model.stress(reference_displacement)
        lower_field = model.stress(model.solve(lower_force))
        rows.append(dict(case=name,method='Q4',status='valid',diagnostics=reference_info,
                         integration_uncertainty=field_errors(model,lower_field,reference_field),
                         nodal_load_q24_vs_q48=float(np.linalg.norm(lower_force-reference_force)/np.linalg.norm(reference_force)),
                         residual=float(model.residual(reference_displacement,reference_force))))
        for subdivisions, proxy in proxies.items():
            try:
                force, diagnostics = proxy.transfer(contact,order=24,analytic=True,anchor_mode='fixed')
                displacement = model.solve(force)
                field = model.stress(displacement)
                rows.append(dict(case=name,method=f'triangle-k{subdivisions}',status='valid',
                                 diagnostics=diagnostics,errors=field_errors(model,field,reference_field),
                                 nodal_load_relative=float(np.linalg.norm(force-reference_force)/np.linalg.norm(reference_force)),
                                 residual=float(model.residual(displacement,force))))
            except ValueError as error:
                rows.append(dict(case=name,method=f'triangle-k{subdivisions}',status='guarded_rejection',error=str(error)))
    summary = []
    for subdivisions in proxies:
        selected = [row for row in rows if row['method'] == f'triangle-k{subdivisions}']
        admitted = [row for row in selected if row['status'] == 'valid']
        summary.append(dict(subdivisions=subdivisions,cases=len(selected),valid=len(admitted),
                            rejected=len(selected)-len(admitted),
                            maximum_field_error=max((row['errors']['field'] for row in admitted),default=None),
                            maximum_peak_error=max((row['errors']['peak'] for row in admitted),default=None),
                            maximum_Q_error=max((row['errors']['Q'] for row in admitted),default=None),
                            maximum_force_error=max((row['diagnostics']['force_error'] for row in admitted),default=None),
                            maximum_moment_error=max((row['diagnostics']['moment_error'] for row in admitted),default=None)))
    result = dict(scope='Bounded same-Q4-mother nonconvex loading diagnostic. Prescribed 0.5/1mm radii, not a concave Hertz contact reference or arbitrary nonconvex qualification.',
                  geometry=dict(name='torus',major_radius_m=.025,minor_radius_m=.008,vertices=len(model.vertices),
                                quads=len(model.quads),raw_dofs=model.ndof,thickness_m=model.h),
                  quadrature=dict(reference=48,candidate=24,reference_uncertainty='Q4 q24 vs q48'),
                  source_center='Original actual Q4 physical surface point, unchanged for all candidates.',
                  common_model='Exactly the same K, mass, inertia relief, translation gauge and stress samples.',
                  cases=cases,summary=summary,rows=rows)
    (destination/'geometry_controls.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
