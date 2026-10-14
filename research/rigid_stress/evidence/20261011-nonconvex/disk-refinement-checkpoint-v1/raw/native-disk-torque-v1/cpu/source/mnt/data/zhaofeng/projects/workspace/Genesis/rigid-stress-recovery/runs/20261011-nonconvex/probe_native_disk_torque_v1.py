"""Audit finite positive-area legal free moments against the current verified-disk integration."""

import json
import sys
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
    kernel_refine_failed_disks,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info,
    create_traction_samples,
    kernel_fit_tractions,
    kernel_traction_samples,
)
from genesis.utils.misc import qd_to_numpy

backend, destination = sys.argv[1], Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=False)
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
mesh = Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")
with np.load(mesh) as saved:
    vertices, tets, faces = (saved[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
triangle = vertices[faces[0]]
anchor = triangle.mean(axis=0)
owner = np.flatnonzero(np.isin(tets, faces[0]).sum(axis=1) == 3)[0]
opposite = vertices[tets[owner][~np.isin(tets[owner], faces[0])][0]]
normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
normal /= np.linalg.norm(normal)
normal *= np.sign(normal @ (opposite - triangle[0]))
radius = 1e-5
fractions = np.tile([0.99, 0.999], 3)
frictions = np.repeat([0.5, 0.85, 1.2], 2)
inner = fractions + 0.5 * (1.0 - fractions)
outer = fractions + 0.75 * (1.0 - fractions)
mean_radius = 2.0 / 3.0 * (outer**2 + outer * inner + inner**2) / (outer + inner)
swirl_ratio = frictions * fractions / mean_radius
assert (swirl_ratio < frictions).all() and ((outer < 1) & (inner < outer)).all()
# Uniform compression and constant circumferential traction on each finite annulus have resultant n and this torque.
options = gs.options.RigidStressOptions(mesh=mesh, method="direct")
model = StressModel(options)
surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
contacts, sources = create_contacts(1, len(fractions), radius), create_sources(1, len(fractions))
footprints = create_footprints(paths.info.positions.shape[0], 1, len(fractions))
tractions, info = create_surface_fan_tractions(surface.info, 1, len(fractions)), create_traction_info(surface.info)
contacts.valid.fill(True)
contacts.position.from_numpy(np.broadcast_to(anchor, (1, len(fractions), 3)).copy())
contacts.force.from_numpy(np.broadcast_to(normal, (1, len(fractions), 3)).copy())
contacts.friction.from_numpy(frictions[None])
sources.geom.fill(7)
sources.pair.fill(11)
sources.witness.from_numpy(np.broadcast_to(anchor, (1, len(fractions), 3)).copy())
sources.free_moment.from_numpy((radius * fractions * frictions)[:, None] * normal[None, :][None])
epsilon = float(np.finfo(gs.np_float).eps)
kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
assert (qd_to_numpy(sources.status, copy=True) == 0).all()
kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
old_status = qd_to_numpy(tractions.status, transpose=True, copy=True)[:, 0]
kernel_refine_failed_disks(contacts, sources, footprints, tractions, surface.info, info, epsilon)
kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128, retry_mode=2)
samples = create_traction_samples(tractions)
kernel_traction_samples(samples, tractions, surface.info, info)
positions = qd_to_numpy(samples.positions, transpose=True, copy=True)[:, :, 0]
counts = qd_to_numpy(tractions.sample_count, transpose=True, copy=True)[:, 0]
status = qd_to_numpy(tractions.status, transpose=True, copy=True)[:, 0]
evaluations = qd_to_numpy(tractions.evaluations, transpose=True, copy=True)[:, 0]
rows = []
for i_b in range(len(fractions)):
    maximum_radius = float(np.linalg.norm(positions[i_b, : counts[i_b]] - anchor, axis=1).max() / radius)
    rows.append(
        {
            "torque_fraction_of_mu_R_Fn": float(fractions[i_b]),
            "actual_mu": float(frictions[i_b]),
            "annulus_inner_R": float(inner[i_b]),
            "annulus_outer_R": float(outer[i_b]),
            "annulus_uniform_swirl_ratio": float(swirl_ratio[i_b]),
            "baseline_status": int(old_status[i_b]),
            "refined_status": int(status[i_b]),
            "evaluations_including_failure": int(evaluations[i_b]),
            "samples": int(counts[i_b]),
            "maximum_sample_radius_R": maximum_radius,
            "discrete_normal_torque_capacity_gap": float(fractions[i_b] - maximum_radius),
        }
    )
(destination / "summary.json").write_text(
    json.dumps(
        {
            "backend": backend,
            "cases": rows,
            "scope": "Independent finite-annulus physical feasibility witness and discrete torque capacity diagnosis. No native or throughput acceptance claimed.",
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(rows), flush=True)
