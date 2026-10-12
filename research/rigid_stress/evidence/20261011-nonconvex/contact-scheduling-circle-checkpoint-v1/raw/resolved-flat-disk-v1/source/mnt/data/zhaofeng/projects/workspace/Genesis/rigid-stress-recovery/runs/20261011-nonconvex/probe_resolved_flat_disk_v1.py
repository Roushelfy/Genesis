"""Diagnose sampling of a legal finite-area planar load on the unchanged low hollow shell."""

import json
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.compatibility import (
    create_wrench_search,
    kernel_certify_continuous_force,
    kernel_certify_continuous_wrench,
    kernel_search_wrench_certificate,
)
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info,
    create_traction_samples,
    kernel_fit_tractions,
    kernel_traction_samples,
)
from genesis.utils.misc import qd_to_numpy

root = Path(__file__).parent
output = root / "resolved-flat-disk-v1/results"
output.mkdir(parents=True, exist_ok=False)
gs.init(backend=gs.cpu, precision="64", logging_level="warning")
asset = Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")
model = StressModel(gs.options.RigidStressOptions(mesh=asset, method="pcg"))
surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
with np.load(asset) as mesh:
    triangle = mesh["vertices"][mesh["surface_triangles"][0]]
anchor = triangle.mean(axis=0)
normal = -qd_to_numpy(surface.info.face_normal, copy=True)[0]
tangent = triangle[1] - triangle[0]
tangent -= (tangent @ normal) * normal
tangent /= np.linalg.norm(tangent)
radius, friction, tangent_ratio = 1e-5, 0.5, 0.4
force = normal + tangent_ratio * tangent
contacts, sources = create_contacts(1, 1, radius), create_sources(1, 1)
contacts.valid.fill(True)
contacts.center.from_numpy(anchor[None, None])
contacts.position.from_numpy(anchor[None, None])
contacts.force.from_numpy(force[None, None])
contacts.anchor_face.fill(0)
contacts.friction.fill(friction)
sources.status.fill(0)
sources.free_moment.fill(0)
sources.anchor_moment.fill(0)
footprints = create_footprints(paths.info.positions.shape[0], 1, 1)
tractions = create_surface_fan_tractions(surface.info, 1, 1)
info, search = create_traction_info(surface.info), create_wrench_search(1, 1, 8)
epsilon = float(np.finfo(gs.np_float).eps)
kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info)
samples = create_traction_samples(tractions)
kernel_traction_samples(samples, tractions, surface.info, info)
count = int(qd_to_numpy(tractions.sample_count, copy=True)[0, 0])
assert count > 2
positions = qd_to_numpy(samples.positions, transpose=True, copy=True)[0, :count, 0]
normals = qd_to_numpy(samples.normals, transpose=True, copy=True)[0, :count, 0]
np.testing.assert_allclose(normals, np.broadcast_to(normal, normals.shape), rtol=0, atol=64 * epsilon)
coordinates = (positions - anchor) @ tangent
sample_minimum = float(coordinates.min())
assert -radius < sample_minimum < 0.0
centroid_coordinate = 0.5 * (sample_minimum - radius)
height = -centroid_coordinate / tangent_ratio
point = anchor + height * normal
contacts.position.from_numpy(point[None, None])
moment = np.cross(point - anchor, force)
sources.anchor_moment.from_numpy(moment[None, None])

# A uniform vector traction on this positive-area disk has exactly the declared
# force and moment. It lies wholly on the actual material face and physical disk;
# its tangential/normal ratio is strictly below the actual input friction.
patch_radius = 0.25 * (radius - abs(centroid_coordinate))
centroid = anchor + centroid_coordinate * tangent
edge_distances = []
for first, second in zip(triangle, np.roll(triangle, -1, axis=0)):
    direction = second - first
    edge_distances.append(np.linalg.norm(np.cross(anchor - first, direction)) / np.linalg.norm(direction))
assert min(edge_distances) > radius
assert abs(centroid_coordinate) + patch_radius < radius and patch_radius > 0
assert tangent_ratio < friction
np.testing.assert_allclose(np.cross(centroid - point, force), np.zeros(3), rtol=0, atol=512 * epsilon * radius)
kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
fit_status = int(qd_to_numpy(tractions.status, copy=True)[0, 0])
kernel_certify_continuous_force(contacts, tractions, surface.info, epsilon, 1e-8)
kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, 1e-8)
kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 8, 16384, 1e-8)
status = int(qd_to_numpy(tractions.status, copy=True)[0, 0])
assert status in (3, 4), status
gap = sample_minimum - centroid_coordinate
assert gap > 1e-8 * radius
np.savez_compressed(
    output / "input.npz",
    anchor=anchor,
    point=point,
    force=force,
    radius=radius,
    friction=friction,
    normal=normal,
    tangent=tangent,
    samples=positions,
    target_moment=moment,
)
summary = {
    "radius_m": radius,
    "friction": friction,
    "tangent_ratio": tangent_ratio,
    "sample_count": count,
    "sample_minimum_coordinate_m": sample_minimum,
    "continuous_feasible_centroid_coordinate_m": centroid_coordinate,
    "positive_area_patch_radius_m": patch_radius,
    "sampled_hull_gap_m": gap,
    "fit_status": fit_status,
    "diagnostic_status": status,
    "force_error": float(qd_to_numpy(tractions.force_error, copy=True)[0, 0]),
    "moment_error": float(qd_to_numpy(tractions.moment_error, copy=True)[0, 0]),
    "scope": "A manufactured legal small-radius contact on the unchanged low shell proves finite-area continuum feasibility while the fixed Q10 cloud misses its wrench. This is a contact-regression diagnosis, not physical mesh/integration convergence or a throughput measurement.",
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
