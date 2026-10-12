"""Classify retained multi-source numerical failure using independent LP/convex dual and native cold fits."""
import json
import math
from pathlib import Path

import numpy as np
from cvxopt import matrix, spmatrix, solvers
from scipy.optimize import linprog, minimize, root
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import create_surface_fan_tractions
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_traction_info, create_traction_samples, kernel_traction_samples, kernel_fit_tractions, kernel_scatter_tractions
from genesis.utils.array_class import iter_data, V, V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.wrench import project_coulomb
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.field_cpu import stress_field

directory = Path(__file__).parent
capture = directory / "native-certified-failure-capture-v1/b4096-623103/failure/before-rigid-integration.npz"
output = directory / "native-group-failure-audit-v3"
with np.load(capture) as asset:
    saved = {key: asset[key].copy() for key in asset.files}
gs.init(backend=gs.cpu, precision="64", logging_level="warning")
options = gs.options.RigidStressOptions(mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz", method="direct", contact_model="intrinsic", intrinsic_load_partition="body_source_prior")
model = StressModel(options)
surface = StressSurface(10, model.info)
info = create_traction_info(surface.info)
source_count = saved["fit_contacts.valid"].shape[1]
contacts, sources = create_contacts(source_count, 1, 0.006), create_sources(source_count, 1)
tractions = create_surface_fan_tractions(surface.info, source_count, 1, shared_support=True, source_priors=True)
for name, state in (("fit_contacts", contacts), ("fit_sources", sources), ("tractions", tractions)):
    for item in iter_data(state, name):
        if ".frontier." in item.name or math.prod(item.value.shape) == 0:
            continue
        array = saved[item.name]
        if item.value.shape:
            array = np.moveaxis(array, 0, len(item.value.shape) - 1)
        item.value.from_numpy(np.ascontiguousarray(array) if array.ndim else array)
samples = create_traction_samples(tractions)
kernel_traction_samples(samples, tractions, surface.info, info)
counts = qd_to_numpy(tractions.sample_count, transpose=True)[0]
valid = qd_to_numpy(contacts.valid, transpose=True)[0]
indices = np.flatnonzero(valid)
assert len(indices) >= 1 and indices[0] == 0
positions_all = qd_to_numpy(samples.positions, transpose=True)[0]
normals_all = qd_to_numpy(samples.normals, transpose=True)[0]
profiles_all = qd_to_numpy(tractions.profile, transpose=True)[0]
faces_all = qd_to_numpy(samples.faces, transpose=True)[0]
friction_all = qd_to_numpy(contacts.friction, transpose=True)[0]
clouds = []
for index in indices:
    count = counts[index]
    eligible = profiles_all[:count, index] > 0
    clouds.append((positions_all[:count, index][eligible], normals_all[:count, index][eligible], profiles_all[:count, index][eligible], faces_all[:count, index][eligible], np.full(eligible.sum(), friction_all[index]), np.full(eligible.sum(), index, dtype=int)))
positions, normals, profiles, faces, friction, source = [np.concatenate([cloud[dimension] for cloud in clouds]) for dimension in range(6)]
original_forces = qd_to_numpy(tractions.priors.force, transpose=True)[0]
original_moments = qd_to_numpy(tractions.priors.anchor_moment, transpose=True)[0]
centers = qd_to_numpy(contacts.center, transpose=True)[0]
points = qd_to_numpy(contacts.position, transpose=True)[0]
radii = qd_to_numpy(contacts.radius, transpose=True)[0]
anchor, point, radius = centers[0], points[0], float(radii[0])
force = qd_to_numpy(contacts.force, transpose=True)[0, 0]
moment = qd_to_numpy(sources.anchor_moment, transpose=True)[0, 0]
scale = float(qd_to_numpy(tractions.priors.group_scale, transpose=True)[0, 0])
budget = np.linalg.norm(force) or scale
target = np.concatenate((force, moment / radius)) / scale
relative = (positions - anchor) / radius
axes = np.eye(3)[np.argmin(abs(normals), axis=1)]
tangent = np.cross(normals, axes)
tangent /= np.linalg.norm(tangent, axis=1)[:, None]
other = np.cross(normals, tangent)
reports = []
angles = np.arange(256) * (2 * np.pi / 256)
for kind, dilation in (("inner", 1.0), ("outer", 1 / np.cos(np.pi / 256))):
    rays = normals[:, None] + (friction * dilation)[:, None, None] * (np.cos(angles)[None, :, None] * tangent[:, None] + np.sin(angles)[None, :, None] * other[:, None])
    columns = np.concatenate((rays, np.cross(relative[:, None], rays)), axis=-1).reshape(-1, 6).T
    solved = linprog(np.zeros(columns.shape[1]), A_eq=columns, b_eq=target, bounds=(0, None), options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    record = {"polygon": kind, "success": bool(solved.success), "status": int(solved.status), "message": str(solved.message)}
    if solved.success:
        recovered = np.einsum("qk,qki->qi", solved.x.reshape(len(positions), 256), rays)
        actual = np.concatenate((recovered.sum(axis=0), np.cross(relative, recovered).sum(axis=0)))
        compression = np.einsum("qi,qi->q", recovered, normals)
        record.update(normalized_wrench_error=float(np.linalg.norm(actual - target)), physical_cone_excess=float(np.max(np.linalg.norm(recovered - compression[:, None] * normals, axis=1) - friction * compression)))
        np.savez_compressed(output / f"{kind}-primal.npz", force=recovered * scale, positions=positions, normals=normals)
    else:
        dual = linprog(target, A_ub=-columns.T, b_ub=np.zeros(columns.shape[1]), bounds=[(-1, 1)] * 6, options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
        record["dual_success"] = bool(dual.success)
        if dual.success:
            ray = -dual.x / np.linalg.norm(dual.x)
            values = ray[:3] + np.cross(ray[3:], relative)
            compression = np.einsum("qi,qi->q", values, normals)
            record.update(ray=ray.tolist(), target_margin=float(target @ ray), true_sampled_polar_upper_bound=float(np.max(compression + friction * np.linalg.norm(values - compression[:, None] * normals, axis=1))))
    reports.append(record)
    print(json.dumps(record), flush=True)
totals = np.array([profiles[source == index].sum() for index in range(source_count)])
weights = profiles / profiles.sum()
density = original_forces[source] * (profiles.sum() / totals[source] / scale)[:, None]
operator = np.zeros((len(positions), 6, 3))
operator[:, :3] = np.eye(3)
x, y, z = relative.T
operator[:, 3, 1], operator[:, 3, 2] = -z, y
operator[:, 4, 0], operator[:, 4, 2] = z, -x
operator[:, 5, 0], operator[:, 5, 1] = -y, x

def evaluate(multiplier):
    values = density - np.einsum("qij,i->qj", operator, multiplier)
    projected, jacobian = project_coulomb(values, normals, friction)
    energy = target @ multiplier + 0.5 * np.einsum("q,qi,qi->", weights, projected, projected)
    gradient = target - np.einsum("q,qij,qj->i", weights, operator, projected)
    hessian = np.einsum("q,qik,qkl,qjl->ij", weights, operator, jacobian, operator)
    return energy, gradient, hessian, weights[:, None] * scale * projected

oracle = None
if any(report["success"] for report in reports):
    fitted = minimize(lambda value: evaluate(value)[:2], np.zeros(6), jac=True, hess=lambda value: evaluate(value)[2], method="trust-exact", options={"gtol": 1e-13, "maxiter": 2000})
    corrected = root(lambda value: evaluate(value)[1], fitted.x, jac=lambda value: evaluate(value)[2], options={"xtol": 1e-12})
    multiplier = min((fitted.x, corrected.x), key=lambda value: np.linalg.norm(evaluate(value)[1]))
    energy, gradient, hessian, recovered = evaluate(multiplier)
    oracle = {"iterations": int(fitted.nit), "optimizer_success": bool(fitted.success), "message": str(fitted.message), "energy": float(energy), "dual": multiplier.tolist(), "force_error": float(np.linalg.norm(gradient[:3]) * scale / budget), "moment_error": float(np.linalg.norm(gradient[3:] - np.cross((point - anchor) / radius, gradient[:3])) * scale / budget), "hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist()}
    print(json.dumps(oracle), flush=True)
    np.savez_compressed(output / "independent-optimum.npz", force=recovered, dual=multiplier)
conic_reference = None
if oracle is not None and max(oracle["force_error"], oracle["moment_error"]) > 1e-8:
    # Independent primal SOCP,scaled by sqrt(area weights),uses neither native duals nor native forces.
    count = len(positions)
    size = 3 * count
    roots = np.sqrt(weights)
    local_cones = -np.stack((friction[:, None] * normals, tangent, other), axis=1)
    rows = np.broadcast_to(3 * np.arange(count)[:, None, None] + np.arange(3)[None, :, None], (count, 3, 3)).reshape(-1)
    cols = np.broadcast_to(3 * np.arange(count)[:, None, None] + np.arange(3)[None, None, :], (count, 3, 3)).reshape(-1)
    cones = spmatrix(local_cones.reshape(-1).tolist(), rows.tolist(), cols.tolist(), (size, size))
    equality = (operator.transpose(1, 0, 2) * roots[None, :, None]).reshape(6, size)
    equality = spmatrix(equality.reshape(-1).tolist(), np.repeat(np.arange(6), size).tolist(), np.tile(np.arange(size), 6).tolist(), (6, size))
    diagonal = spmatrix([1.0] * size, list(range(size)), list(range(size)), (size, size))
    conic = solvers.coneqp(diagonal, matrix(-(roots[:, None] * density).reshape(-1)), cones, matrix(np.zeros(size)), {"l": 0, "q": [3] * count, "s": []}, equality, matrix(target), kktsolver="chol", options={"maxiters": 200, "abstol": 1e-12, "reltol": 1e-12, "feastol": 1e-12, "show_progress": False})
    recovered = np.asarray(conic["x"]).reshape(count, 3) * roots[:, None] * scale
    wrench = np.concatenate((recovered.sum(axis=0), np.cross(relative, recovered).sum(axis=0))) / scale
    gradient = target - wrench
    conic_reference = {"status": str(conic["status"]), "iterations": int(conic["iterations"]), "gap": float(conic["gap"]), "relative_gap": None if conic["relative gap"] is None else float(conic["relative gap"]), "primal_infeasibility": float(conic["primal infeasibility"]), "dual_infeasibility": float(conic["dual infeasibility"]), "force_error": float(np.linalg.norm(gradient[:3]) * scale / budget), "moment_error": float(np.linalg.norm(gradient[3:] - np.cross((point - anchor) / radius, gradient[:3])) * scale / budget)}
    print(json.dumps(conic_reference), flush=True)
    assert conic["status"] == "optimal"
    assert max(conic_reference["force_error"], conic_reference["moment_error"]) <= 1e-8
    oracle = {"previous_dual_attempt": oracle, "primal_SOCP": conic_reference, "force_error": conic_reference["force_error"], "moment_error": conic_reference["moment_error"]}
    np.savez_compressed(output / "independent-primal-optimum.npz", force=recovered)
native = []
for evaluations in (128,):
    kernel_fit_tractions(contacts, sources, tractions, info, float(np.finfo(np.float64).eps), 9e-9, evaluations, cooperative=False, warm_start=False)
    row = {"budget": evaluations, "status": int(qd_to_numpy(tractions.status, transpose=True)[0, 0]), "force_error": float(qd_to_numpy(tractions.force_error, transpose=True)[0, 0]), "moment_error": float(qd_to_numpy(tractions.moment_error, transpose=True)[0, 0]), "evaluations": int(qd_to_numpy(tractions.evaluations, transpose=True)[0, 0]), "dual": qd_to_numpy(tractions.dual, transpose=True)[0, 0].tolist()}
    native.append(row)
    print(json.dumps(row), flush=True)
kernel_fit_tractions(contacts, sources, tractions, info, float(np.finfo(np.float64).eps), 9e-9, 128, retry_mode=4, cooperative=False, warm_start=False)
row = {"mode": "continue_only_failed_refined_fan", "status": int(qd_to_numpy(tractions.status, transpose=True)[0, 0]), "force_error": float(qd_to_numpy(tractions.force_error, transpose=True)[0, 0]), "moment_error": float(qd_to_numpy(tractions.moment_error, transpose=True)[0, 0]), "evaluations": int(qd_to_numpy(tractions.evaluations, transpose=True)[0, 0])}
native.append(row)
print(json.dumps(row), flush=True)
assert row["status"] == 0 and max(row["force_error"], row["moment_error"]) <= 1e-8
field = None
if oracle is not None and max(oracle["force_error"], oracle["moment_error"]) <= 1e-8:
    state = model.create_state(1, "full")
    errno, omega = V(dtype=gs.qd_int, shape=(1,)), V_VEC(3, dtype=gs.qd_float, shape=(1,))
    errno.fill(0)
    omega.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    assert qd_to_numpy(errno)[0] == 0
    basis_all = qd_to_numpy(samples.shape, transpose=True)[0]
    basis = np.concatenate([basis_all[:counts[index], index][profiles_all[:counts[index], index] > 0] for index in indices])
    surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T
    nodal = np.zeros((model.info.vertices.shape[0], 3))
    np.add.at(nodal, surface_nodes[faces].reshape(-1), (basis[:, :, None] * recovered[:, None]).reshape(-1, 3))
    native_nodal = qd_to_numpy(state.force, transpose=True)[0]
    np.testing.assert_allclose(native_nodal, nodal, rtol=1e-4, atol=1e-10)
    with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
        vertices, tets, boundary_faces = (asset[key].copy() for key in ("vertices", "tetrahedra", "surface_triangles"))
    independent = P2Shell(vertices, tets, boundary_faces, options.young, options.poisson, options.density, 2, factor_backend="none")
    rhs = independent.balance(nodal.reshape(independent.ndof, 1))
    free = np.setdiff1d(np.arange(independent.ndof), qd_to_numpy(model.info.pins, transpose=True))
    expected = np.zeros_like(rhs)
    expected[free] = splu(independent.k[free][:, free].tocsc()).solve(rhs[free])
    model.recover(omega, state)
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(1, independent.ndof).T
    complete_residual = float(np.linalg.norm(independent.k @ displacement - rhs))
    complete_budget = max(options.absolute_tolerance, options.tolerance * float(np.linalg.norm(rhs)))
    assert complete_residual <= complete_budget and qd_to_numpy(state.valid)[0]
    expected_tensor, expected_vm = stress_field(independent.glambda, independent.elements, expected[:, 0], options.young, options.poisson)
    tensor, vm, peak = qd_to_numpy(state.stress_tensor, transpose=True)[0], qd_to_numpy(state.von_mises, transpose=True)[0, :, :, 0], float(qd_to_numpy(state.peak)[0])
    np.testing.assert_allclose(tensor, expected_tensor, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(vm, expected_vm, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(peak, expected_vm.max(), rtol=1e-4, atol=1e-3)
    field = {"complete_residual_N": complete_residual, "complete_residual_budget_fraction": complete_residual / complete_budget, "native_peak_Pa": peak, "independent_peak_Pa": float(expected_vm.max()), "peak_difference_Pa": float(abs(peak - expected_vm.max())), "nodal_max_difference_N": float(abs(native_nodal - nodal).max())}
    print(json.dumps(field), flush=True)
else:
    raise AssertionError("Independent optimizer did not reach the unchanged wrench budget.")
np.savez_compressed(output / "cloud.npz", positions=positions, normals=normals, profiles=profiles, faces=faces, source=source, friction=friction, density=density, target=target, original_forces=original_forces, original_moments=original_moments, centers=centers, points=points, radii=radii, scale=scale)
summary = {"scope": __doc__, "environment": int(saved["environment"]), "sources": indices.tolist(), "sample_count": len(profiles), "minimum_normalized_weight": float(weights.min()), "maximum_normalized_weight": float(weights.max()), "sampled_LP": reports, "independent_dual": oracle, "native_cold_fits": native, "independent_full_field": field}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
