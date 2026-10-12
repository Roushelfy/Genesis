"""Check native disk integration, fitting, scatter and full equilibrium against independent CPU FP64."""

import json
import sys
from pathlib import Path

import numpy as np
from circle_disk_candidate_v2 import kernel_build_single_face_disks
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info,
    create_tractions,
    kernel_fit_tractions,
    kernel_scatter_tractions,
)
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import WrenchPatch
from scipy.sparse.linalg import splu
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit
from tests.rigid.test_stress_traction import kernel_traction_queries

import genesis as gs
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy

root, backend = Path(__file__).parent, sys.argv[1]
output = root / f"circle-disk-candidate-v2/{backend}/results"
output.mkdir(parents=True, exist_ok=False)
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
with np.load(root / "resolved-flat-disk-v1/results/input.npz") as saved:
    anchor, point, force = (saved[key] for key in ("anchor", "point", "force"))
    radius, friction = float(saved["radius"]), float(saved["friction"])
    moment = saved["target_moment"]
asset = Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")
options = gs.options.RigidStressOptions(mesh=asset, method="direct")
model, surface = StressModel(options), None
surface = StressSurface(10, model.info)
contacts, sources, tractions = (
    create_contacts(1, 1, radius),
    create_sources(1, 1),
    create_tractions(100, 1, 1, local_samples=True),
)
info = create_traction_info(surface.info)
contacts.valid.fill(True)
contacts.center.from_numpy(anchor[None, None])
contacts.position.from_numpy(point[None, None])
contacts.force.from_numpy(force[None, None])
contacts.anchor_face.fill(0)
contacts.friction.fill(friction)
sources.status.fill(0)
sources.free_moment.fill(0)
sources.anchor_moment.from_numpy(moment[None, None])
epsilon = float(np.finfo(gs.np_float).eps)
kernel_build_single_face_disks(contacts, tractions, surface.info, epsilon)
assert int(qd_to_numpy(tractions.sample_count, copy=True)[0, 0]) == 100
kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
assert int(qd_to_numpy(tractions.status, copy=True)[0, 0]) == 0
query = V_VEC(3, dtype=gs.qd_float, shape=(100, 1, 1))
kernel_traction_queries(query, contacts, tractions, info)
native_force = qd_to_numpy(query, transpose=True, copy=True)[0, :, 0]
native_positions = qd_to_numpy(tractions.sample_positions, transpose=True, copy=True)[0, :, 0]
native_profiles = qd_to_numpy(tractions.profile, transpose=True, copy=True)[0, :, 0]
with np.load(asset) as mesh:
    vertices, tets, faces = (mesh[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
triangle = vertices[faces[0]]
owners = np.flatnonzero(np.isin(tets, faces[0]).sum(axis=1) == 3)
opposite = vertices[tets[owners[0]][~np.isin(tets[owners[0]], faces[0])][0]]
normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
normal /= np.linalg.norm(normal)
normal *= np.sign(normal @ (opposite - triangle[0]))
tangent = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
second = np.cross(normal, tangent)
gauss, weights = np.polynomial.legendre.leggauss(10)
gauss, weights = (gauss + 1.0) / 2.0, weights / 2.0
u, v = np.repeat(gauss, 10), np.tile(gauss, 10)
positions = anchor + radius * u[:, None] ** 2 * (
    np.cos(2 * np.pi * v)[:, None] * tangent + np.sin(2 * np.pi * v)[:, None] * second
)
profiles = 4 * np.pi * radius**2 * u**3 * np.repeat(weights, 10) * np.tile(weights, 10)
profiles *= np.exp(-0.5 * (u**2 / 0.45) ** 2) * (1 - u**4) ** 2
np.testing.assert_allclose(native_positions, positions, rtol=0, atol=128 * epsilon * np.linalg.norm(anchor))
np.testing.assert_allclose(native_profiles, profiles, rtol=1e-12, atol=0)
reference_force = cpu_resolved_wrench_fit(
    positions, profiles, np.broadcast_to(normal, positions.shape), WrenchPatch(anchor, force, radius, friction, moment)
)
np.testing.assert_allclose(native_force, reference_force, rtol=1e-4, atol=1e-10)
assert np.linalg.norm(native_force.sum(axis=0) - force) <= 1e-8 * np.linalg.norm(force)
assert np.linalg.norm(np.cross(native_positions - point, native_force).sum(axis=0)) <= 1e-8 * radius * np.linalg.norm(
    force
)
local = np.linalg.lstsq((triangle[1:] - triangle[0]).T, (positions - triangle[0]).T, rcond=None)[0].T
bary = np.column_stack((1 - local.sum(axis=1), local))
basis = np.column_stack(
    (bary * (2 * bary - 1), 4 * bary[:, 0] * bary[:, 1], 4 * bary[:, 0] * bary[:, 2], 4 * bary[:, 1] * bary[:, 2])
)
nodal = np.zeros((model.info.vertices.shape[0], 3))
indices = qd_to_numpy(model.info.surface_nodes, copy=True)[0]
np.add.at(
    nodal, np.broadcast_to(indices, (100, 6)).reshape(-1), (basis[:, :, None] * reference_force[:, None]).reshape(-1, 3)
)
state, omega, errno = (
    model.create_state(1, "full"),
    V_VEC(3, dtype=gs.qd_float, shape=(1,)),
    V(dtype=gs.qd_int, shape=(1,)),
)
omega.fill(0)
kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
assert int(qd_to_numpy(errno, copy=True)[0]) == 0
np.testing.assert_allclose(qd_to_numpy(state.force, transpose=True, copy=True)[0], nodal, rtol=1e-4, atol=1e-10)
oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
rhs = oracle.balance(nodal.reshape(oracle.ndof, 1))
free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, copy=True))
reference = np.zeros_like(rhs)
reference[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
model.recover(omega, state)
assert qd_to_numpy(state.valid, copy=True).all()
displacement = qd_to_numpy(state.displacement, transpose=True, copy=True).reshape(oracle.ndof, 1)
residual = float(np.linalg.norm(oracle.k @ displacement - rhs))
assert residual <= max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
tensor, vm = stress_field(oracle.glambda, oracle.elements, reference[:, 0], options.young, options.poisson)
np.testing.assert_allclose(qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0], tensor, rtol=1e-4, atol=1e-3)
np.testing.assert_allclose(
    qd_to_numpy(state.von_mises, transpose=True, copy=True)[0, :, :, 0], vm, rtol=1e-4, atol=1e-3
)
peak = float(qd_to_numpy(state.peak, copy=True)[0])
np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
summary = {
    "backend": backend,
    "sample_count": 100,
    "evaluations": int(qd_to_numpy(tractions.evaluations, copy=True)[0, 0]),
    "force_error": float(qd_to_numpy(tractions.force_error, copy=True)[0, 0]),
    "moment_error": float(qd_to_numpy(tractions.moment_error, copy=True)[0, 0]),
    "full_residual_N": residual,
    "native_peak_Pa": peak,
    "cpu_peak_Pa": float(vm.max()),
    "scope": "Native verified one-face disk quadrature candidate, unchanged legal input and budgets, independent CPU geometry/vector fit/P2 loads/full equilibrium/tensor/peak. Not integrated or a physical convergence study.",
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
