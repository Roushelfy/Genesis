"""Saved actual policy contact, clipped finite-area Q10 charts and independent CPU audit."""

import json
import sys
from pathlib import Path

import numpy as np
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
)
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info,
    create_traction_samples,
    kernel_fit_tractions,
    kernel_scatter_tractions,
    kernel_traction_samples,
)
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import WrenchPatch
from scipy.sparse.linalg import splu
from tests.rigid.test_stress_footprint import cpu_intrinsic_distances
from tests.rigid.test_stress_shared_support import kernel_shared_traction_queries
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit
from triangle_disk_clip_candidate_v1 import install, kernel_clipped_profiles

import genesis as gs
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy

backend, destination = sys.argv[1], Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=False)
install()
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
root = Path(__file__).parent
with np.load(root / "audit-policy-failure-inputs-v1/results/original-environment-24.npz") as saved:
    original = {key: saved[key].copy() for key in saved.files}
asset = Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")
options = gs.options.RigidStressOptions(mesh=asset, contact_model="intrinsic", intrinsic_support="pair_union")
model = StressModel(options)
surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
info = create_traction_info(surface.info)
contacts, sources = create_contacts(2, 1, 0.006), create_sources(2, 1)
for name in ("position", "center", "anchor_face", "force", "normal", "friction", "radius", "valid"):
    contacts.__dict__[name].from_numpy(original["contacts." + name][:2, None].copy())
for name in ("geom", "pair", "status", "free_moment", "anchor_moment", "barycentric"):
    sources.__dict__[name].from_numpy(original["raw_sources." + name][:2, None].copy())
footprints = create_footprints(paths.info.positions.shape[0], 2, 1)
tractions = create_surface_fan_tractions(surface.info, 2, 1, shared_support=True)
epsilon = float(np.finfo(gs.np_float).eps)
kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
kernel_clipped_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
kernel_bind_traction_support(contacts, sources, footprints, tractions)
kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
summary = {
    "tick": 876,
    "environment": 24,
    "backend": backend,
    "variant": "triangle_disk_radial_clip_q10_ports8",
    "original_radius_m": original["contacts.radius"][:2].tolist(),
    "actual_friction": original["contacts.friction"][:2].tolist(),
}
for name in (
    "status",
    "sample_count",
    "evaluations",
    "force_error",
    "moment_error",
    "certificate_margin",
    "certificate",
):
    summary[name] = qd_to_numpy(tractions.__dict__[name], transpose=True, copy=True)[0].tolist()
(destination / "preliminary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
samples = create_traction_samples(tractions)
kernel_traction_samples(samples, tractions, surface.info, info)
count = qd_to_numpy(tractions.sample_count, transpose=True, copy=True)[0]
indices = qd_to_numpy(tractions.sample_indices, transpose=True, copy=True)[0]
positions = qd_to_numpy(samples.positions, transpose=True, copy=True)[0]
normals = qd_to_numpy(samples.normals, transpose=True, copy=True)[0]
profiles = qd_to_numpy(tractions.profile, transpose=True, copy=True)[0]
basis_native = qd_to_numpy(samples.shape, transpose=True, copy=True)[0]
faces_native = qd_to_numpy(samples.faces, transpose=True, copy=True)[0]
with np.load(asset) as saved:
    vertices, tets, faces = (saved[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
triangles = vertices[faces]
cpu_clouds = []
for i_c in range(2):
    anchor = original["contacts.center"][i_c]
    radius = original["contacts.radius"][i_c]
    # Solve closest-triangle barycentric coordinates and build the clipped chart independently.
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    uv = np.einsum("fij,fj->fi", np.linalg.pinv(edges), anchor - triangles[:, 0])
    inside = (uv >= 0).all(axis=1) & (uv.sum(axis=1) <= 1)
    center_bary = np.column_stack((1 - uv.sum(axis=1), uv))
    first, second = triangles[:, [0, 1, 2]], triangles[:, [1, 2, 0]]
    boundary_edges = second - first
    fraction = np.einsum("fei,fei->fe", anchor - first, boundary_edges)
    fraction /= np.einsum("fei,fei->fe", boundary_edges, boundary_edges)
    fraction = np.clip(fraction, 0, 1)
    boundary_points = first + fraction[..., None] * boundary_edges
    closest = np.argmin(np.linalg.norm(boundary_points - anchor, axis=2), axis=1)
    boundary_bary = np.zeros((len(faces), 3))
    boundary_bary[np.arange(len(faces)), closest] = 1 - fraction[np.arange(len(faces)), closest]
    boundary_bary[np.arange(len(faces)), (closest + 1) % 3] = fraction[np.arange(len(faces)), closest]
    center_bary[~inside] = boundary_bary[~inside]
    center = np.einsum("fa,fai->fi", center_bary, triangles)
    gauss, weights = np.polynomial.legendre.leggauss(10)
    gauss, weights = (gauss + 1) / 2, weights / 2
    u, v = np.repeat(gauss, 10), np.tile(gauss, 10)
    edge = (1 - v**2)[None, None, :, None] * first[:, :, None] + v[None, None, :, None] ** 2 * second[:, :, None]
    direction = edge - center[:, None, None]
    delta = center - anchor
    a = np.einsum("fkqi,fkqi->fkq", direction, direction)
    b = np.einsum("fi,fkqi->fkq", delta, direction)
    c = np.einsum("fi,fi->f", delta, delta) - radius**2
    root = np.sqrt(np.maximum(0, b * b - a * c[:, None, None]))
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(b >= 0, -c[:, None, None] / (b + root), (-b + root) / a)
    cap = np.where((a > 0) & (c[:, None, None] < 0), np.clip(cap, 0, 1), 0)
    s = cap * u**2
    edge_bary = (1 - v**2)[None, None, :, None] * np.eye(3)[None, :, None]
    edge_bary += v[None, None, :, None] ** 2 * np.eye(3)[[1, 2, 0]][None, :, None]
    bary = (1 - s[..., None]) * center_bary[:, None, None] + s[..., None] * edge_bary
    cpu_positions = np.einsum("fkqa,fai->fkqi", bary, triangles).reshape(-1, 3)
    area = np.linalg.norm(np.cross(boundary_edges[:, 0], -boundary_edges[:, 2]), axis=1)
    area = area[:, None] * center_bary[:, [2, 0, 1]]
    area_weights = (area[..., None] * 4 * u**3 * v * cap**2 * np.repeat(weights, 10) * np.tile(weights, 10)).reshape(-1)
    cpu_faces = np.repeat(np.arange(len(faces)), 300)
    bary = bary.reshape(-1, 3)
    basis = np.column_stack((bary * (2 * bary - 1), 4 * bary[:, [0, 0, 1]] * bary[:, [1, 2, 2]]))
    candidates = (area_weights > 0) & (np.linalg.norm(cpu_positions - anchor, axis=1) < radius)
    distance = cpu_intrinsic_distances(
        anchor[None],
        original["contacts.anchor_face"][i_c : i_c + 1],
        cpu_positions[candidates],
        cpu_faces[candidates],
        vertices,
        faces,
        qd_to_numpy(paths.info.positions, transpose=True),
        qd_to_numpy(paths.info.face_ports, transpose=True).T,
        qd_to_numpy(paths.info.starts, transpose=True),
        qd_to_numpy(paths.info.neighbors, transpose=True),
    )[0]
    cpu_profiles = np.zeros(len(area_weights))
    cpu_profiles[candidates] = area_weights[candidates] * np.exp(-0.5 * (distance / (0.45 * radius)) ** 2)
    cpu_profiles[candidates] *= np.maximum(1 - (distance / radius) ** 2, 0) ** 2
    idx = indices[: count[i_c], i_c]
    np.testing.assert_allclose(positions[: count[i_c], i_c], cpu_positions[idx], rtol=2e-12, atol=2e-17)
    np.testing.assert_array_equal(faces_native[: count[i_c], i_c], cpu_faces[idx])
    np.testing.assert_allclose(basis_native[: count[i_c], i_c], basis[idx], rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(profiles[: count[i_c], i_c], cpu_profiles[idx], rtol=2e-10, atol=2e-20)
    selected = cpu_profiles > 0
    assert selected.sum() == count[i_c]
    cpu_clouds.append(
        (
            cpu_positions[selected],
            cpu_profiles[selected],
            cpu_faces[selected],
            basis[selected],
            -qd_to_numpy(surface.info.face_normal, transpose=True)[cpu_faces[selected]],
        )
    )
summary["independent_geometry_profile_passed"] = True
np.savez_compressed(
    destination / "samples.npz",
    positions=positions,
    profiles=profiles,
    normals=normals,
    sample_indices=indices,
    count=count,
    **original,
)
if any(summary["status"]):
    summary["acceptance"] = "unresolved_fit"
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.exit(1)
quad_force = V_VEC(3, dtype=gs.qd_float, shape=(tractions.profile.shape[0], 2, 2, 1))
kernel_shared_traction_queries(quad_force, contacts, tractions, info)
actual_force = qd_to_numpy(quad_force, transpose=True, copy=True)[0]
xyz, w, face_ids, basis, n = [np.concatenate([cloud[i] for cloud in cpu_clouds]) for i in range(5)]
mu = np.concatenate([np.full(len(cloud[1]), original["contacts.friction"][i]) for i, cloud in enumerate(cpu_clouds)])
surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T
nodal = np.zeros((model.info.vertices.shape[0], 3))
summary["cpu_oracles"] = []
for i_c in range(2):
    F, M, p, anchor, R = (
        original[key][i_c]
        for key in (
            "contacts.force",
            "raw_sources.anchor_moment",
            "contacts.position",
            "contacts.center",
            "contacts.radius",
        )
    )
    reference = cpu_resolved_wrench_fit(xyz, w, n, WrenchPatch(anchor, F, R, mu, M))
    native = np.concatenate([actual_force[: count[s], s, i_c] for s in range(2)])
    np.testing.assert_allclose(native, reference, rtol=1e-4, atol=1e-10)
    error_f = np.linalg.norm(native.sum(axis=0) - F) / np.linalg.norm(F)
    error_m = np.linalg.norm(np.cross(xyz - p, native).sum(axis=0) - original["raw_sources.free_moment"][i_c]) / (
        R * np.linalg.norm(F)
    )
    compression = np.einsum("qi,qi->q", native, n)
    cone_excess = np.linalg.norm(native - compression[:, None] * n, axis=1) - mu * compression
    assert max(error_f, error_m) <= 1e-8 and compression.min() >= -1e-12 and cone_excess.max() <= 1e-12
    np.add.at(nodal, surface_nodes[face_ids].reshape(-1), (basis[:, :, None] * reference[:, None]).reshape(-1, 3))
    summary["cpu_oracles"].append(
        {"component": i_c, "force_error": float(error_f), "original_moment_error": float(error_m)}
    )
state, omega, errno = (
    model.create_state(1, "full"),
    V_VEC(3, dtype=gs.qd_float, shape=(1,)),
    V(dtype=gs.qd_int, shape=(1,)),
)
omega.fill(0)
kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
assert not qd_to_numpy(errno).any()
model.recover(omega, state)
assert qd_to_numpy(state.valid).all()
oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
rhs = oracle.balance(nodal.reshape(oracle.ndof, 1))
free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
displacement = np.zeros_like(rhs)
displacement[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
actual = qd_to_numpy(state.displacement, transpose=True)[0].reshape(oracle.ndof, 1)
residual = float(np.linalg.norm(oracle.k @ actual - rhs))
assert residual <= max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
tensor, vm = stress_field(oracle.glambda, oracle.elements, displacement[:, 0], options.young, options.poisson)
np.testing.assert_allclose(qd_to_numpy(state.stress_tensor, transpose=True)[0], tensor, rtol=1e-4, atol=1e-3)
np.testing.assert_allclose(qd_to_numpy(state.von_mises, transpose=True)[0, :, :, 0], vm, rtol=1e-4, atol=1e-3)
peak = float(qd_to_numpy(state.peak)[0])
np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
summary.update(acceptance="passed", full_gauge_residual_N=residual, peak_native_Pa=peak, peak_cpu_Pa=float(vm.max()))
(destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
