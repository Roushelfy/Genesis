"""Bounded compatibility control for native surface-ball Hertz DPR on a shell.

Imports the repository's unchanged PadPressureFit CPU oracle. The support and
base density are evaluated as sqrt(1-|x-anchor|²/a²) on the ACTUAL selected
offset Q4 surface. Positive affine pressure fitting preserves the original
point wrench before consistent translational/rotational shell scatter.

This extends the native law to a specified shell face-side. It does NOT run
the production Quadrants/tetrahedral transfer, nor claim parity with a volume
boundary that includes both inner and outer surfaces. Numerical quadrature
uncertainty is measured, not hidden by the successful wrench solve.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from .contacts import Contact, ShellDPRTransfer, _basis, _shape, hertz_radius
from .core import ShellModel
from .geometry import plate_mesh, egg_mesh, egg_geometry


def _oracle(repository):
    path = Path(repository)/"research/rigid_stress/wrench.py"
    spec = importlib.util.spec_from_file_location("native_dpr_wrench_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _default_repository():
    for parent in Path(__file__).resolve().parents:
        if (parent/"research/rigid_stress/wrench.py").is_file():
            return parent
    return Path(__file__).resolve().parents[1]/"genesis-shell-study"


def native_shell_transfer(transfer, contact, oracle, order=48):
    model = transfer.model
    point = np.asarray(contact.point, dtype=float)
    force = np.asarray(contact.force, dtype=float)
    normal = np.asarray(contact.normal, dtype=float)
    normal = normal/np.linalg.norm(normal)
    magnitude = np.linalg.norm(force)
    radius = contact.radius
    if radius is None:
        radius = hertz_radius(-force@normal, model.E, model.nu, contact.curvatures,
                              contact_modulus=transfer.contact_modulus)
    W = transfer.faces[contact.side][0]
    plane = _basis(normal)
    direction = force/magnitude
    candidates = transfer._candidates(point, radius, contact.side)
    anchors = []
    for face in candidates:
        poly = transfer._projected_polygon(W[face], point, normal, direction, plane)
        N, xy, good = transfer._inverse(poly, [[0., 0.]])
        if good[0]:
            _, dxi, deta = _shape(*xy[0])
            outward = contact.side*np.cross(dxi@W[face], deta@W[face])
            outward /= np.linalg.norm(outward)
            if outward@(-direction) > 1e-12:
                hit = N[0]@W[face]
                anchors.append((np.linalg.norm(hit-point), int(face), hit, outward))
    if not anchors:
        raise ValueError("native anchor missing")
    _, seed, center, anchor_normal = min(anchors, key=lambda row: row[0])
    plane = _basis(anchor_normal)
    candidates = transfer._candidates(center, radius, contact.side)
    positions, base_weights, shapes, face_ids = [], [], [], []
    for face in candidates:
        # Orthogonal projection into the anchor tangent plane is only an
        # integration parameterization. The pressure/support below use x in3D.
        poly = (W[face]-center)@plane
        samples, projected_weights = transfer._face_polar_samples(poly, radius, order)
        if not len(projected_weights):
            continue
        N, xy, good = transfer._inverse(poly, samples)
        if not good.all():
            raise ValueError("native Q4 sample inversion failed")
        physical = N@W[face]
        r2 = np.einsum("ij,ij->i", samples, samples)/(radius*radius)
        distance2 = np.einsum("ij,ij->i", physical-center, physical-center)/(radius*radius)
        # Correct the tangent-plane Hertz integration rule into exactly the
        # native 3D surface-ball density, including true surface area.
        _, dxi, deta = _shape(xy[:, 0], xy[:, 1])
        jx, jy = dxi@W[face], deta@W[face]
        px, py = dxi@poly, deta@poly
        jac = np.linalg.norm(np.cross(jx, jy), axis=1)/np.abs(px[:, 0]*py[:, 1]-px[:, 1]*py[:, 0])
        weights = projected_weights*np.sqrt(np.maximum(0., 1-distance2))/np.sqrt(np.maximum(1e-300, 1-r2))*jac
        keep = weights > 0
        if keep.any():
            positions.append(physical[keep]); base_weights.append(weights[keep]);shapes.append(N[keep])
            face_ids.append(np.full(np.count_nonzero(keep), face, dtype=int))
    if not base_weights:
        raise ValueError("native support empty")
    positions = np.vstack(positions)
    weights = np.concatenate(base_weights)
    shapes = np.vstack(shapes)
    face_ids = np.concatenate(face_ids)
    friction = 1. if contact.friction is None else contact.friction
    patch = oracle.WrenchPatch(point, force, radius, friction, np.zeros(3), -normal)
    sample_force, diagnostic = oracle.PadPressureFit(positions, weights, patch).solve()
    load = np.zeros((len(model.vertices), 6))
    for face in np.unique(face_ids):
        selection = face_ids == face
        nodal = shapes[selection].T@sample_force[selection]
        nodes = model.quads[face]
        np.add.at(load[:, :3], nodes, nodal)
        np.add.at(load[:, 3:], nodes, contact.side*.5*model.h*np.cross(model.directors[nodes], nodal))
    total = load[:, :3].sum(axis=0)
    moment = np.cross(model.vertices, load[:, :3]).sum(axis=0)+load[:, 3:].sum(axis=0)
    force_error = np.linalg.norm(total-force)/magnitude
    moment_error = np.linalg.norm(moment-np.cross(point, force))/(magnitude*radius)
    if force_error > 1e-8 or moment_error > 1e-8:
        raise ValueError("native scatter wrench failure")
    return load.ravel(), dict(radius_m=float(radius), source_anchor_distance_m=float(np.linalg.norm(center-point)),
        seed_face=seed, samples=len(positions), native_base_mass=float(weights.sum()),
        force_relative_error=float(force_error), moment_normalized_error=float(moment_error),
        pressure_fit_evaluations=diagnostic.constrained_evaluations, quadrature_order=order)


def _relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def helper_tests(oracle):
    """Guard helper options independently of the twelve comparison fixtures."""
    vertices, quads = plate_mesh(1,1,.04,.04)
    model=ShellModel(vertices,quads,E=1e10,nu=.3,h=.0005,gauge="translation")
    transfer=ShellDPRTransfer(model)
    point=np.array([0.,0.,.5*model.h])
    normal=np.array([0.,0.,1.])
    contact=Contact(point,np.array([0.,0.,-.05]),normal,(50.,70.),friction=0.)
    _,base=native_shell_transfer(transfer,contact,oracle,order=24)
    override=2*model.E/(1-model.nu**2)
    _,changed=native_shell_transfer(ShellDPRTransfer(model,contact_modulus=override),contact,oracle,order=24)
    measured=changed["radius_m"]/base["radius_m"]
    expected=2**(-1/3)
    assert abs(measured-expected)<1e-14
    rejected=False
    error=None
    try:
        invalid=Contact(point,np.array([.005,0.,-.05]),normal,(50.,70.),friction=0.)
        native_shell_transfer(transfer,invalid,oracle,order=24)
    except ValueError as exc:
        rejected=True
        error=str(exc)
    assert rejected, "An explicitly frictionless contact must reject nonzero shear"
    return dict(passed=True,explicit_zero_friction_normal_load_passed=True,
        explicit_zero_friction_shear_rejected=rejected,zero_friction_rejection=error,
        contact_modulus_override_pa=override,measured_radius_ratio=measured,expected_radius_ratio=expected)


def run(repository, output):
    oracle, oracle_path = _oracle(repository)
    option_checks=helper_tests(oracle)
    rows = []
    for name, n in [("plane", 4), ("egg", 4), ("egg", 8)]:
        axes, beta = (.022, .022, .030), .18
        if name == "plane":
            v, q = plate_mesh(n,n,.04,.04)
            directors = np.tile([0.,0.,1.], (len(v),1))
        else:
            v, q = egg_mesh(n, axes, beta)
            directors = egg_geometry(v,axes,beta)[0]
        model = ShellModel(v,q,E=1e10,nu=.3,h=.0005,directors=directors,gauge="translation")
        transfer = ShellDPRTransfer(model)
        face = len(q)//3+1
        nodes = q[face]
        point = np.mean(v[nodes]+.5*model.h*directors[nodes],axis=0)
        normal = directors[nodes].mean(axis=0);normal/=np.linalg.norm(normal)
        tangent = _basis(normal)[:,0]
        for shear_ratio in (0., .3):
            for radius in (None, .002):
                contact = Contact(point,-.05*normal+shear_ratio*.05*tangent,normal,(50.,70.),radius=radius,friction=.5,face_hint=face)
                row=dict(geometry=name,n=n,shear_ratio=shear_ratio,radius_mode="automatic" if radius is None else "manual2mm")
                try:
                    native, info = native_shell_transfer(transfer,contact,oracle,order=48)
                    lower, low_info = native_shell_transfer(transfer,contact,oracle,order=24)
                    projected, projected_info = transfer.transfer(contact,method="fast",order=24)
                    row.update(native_valid=True, native=info,
                        projected_method=projected_info["method"],
                        projected_vs_native_load_relative=_relative(projected,native),
                        native_order24_vs48_load_relative=_relative(lower,native))
                    # Use exactly the same shell K, gauge and inertia relief.
                    displacement=model.solve(np.column_stack((projected,native,lower)))
                    stress=model.stress(displacement)
                    tensor=np.concatenate((stress["sigma_top"],stress["sigma_bottom"]),axis=1)
                    row["projected_vs_native_stress_relative"]=_relative(tensor[...,0],tensor[...,1])
                    row["native_order24_vs48_stress_relative"]=_relative(tensor[...,2],tensor[...,1])
                except Exception as exc:
                    row.update(native_valid=False,error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                print(row,flush=True)
    result=dict(scope="Native surface-ball Hertz pressure LAW extended to the selected shell-side offset Q4 surface; actual unchanged CPU PadPressureFit oracle. This is not production Quadrants/tet parity or independent contact validation.",
        oracle_path=str(oracle_path),oracle_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_tests=option_checks,
        quadrature_note="24/48 control integrates pointwise native support/density; changes bound numerical uncertainty, not continuum certification.",rows=rows)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repository",default=str(_default_repository()))
    parser.add_argument("--output",default=str(Path(__file__).resolve().parent/"native_dpr_results"/"control.json"))
    args=parser.parse_args()
    run(args.repository,args.output)
