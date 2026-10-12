"""Offline CPU classification of immutable new single-contact failures; no production mutation."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize, root as solve_root

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import create_surface_fan_tractions
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info, create_traction_samples, kernel_traction_samples, kernel_fit_tractions,
)
from genesis.utils.array_class import iter_data
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.wrench import project_coulomb

root = Path(__file__).parent
label = sys.argv[1]
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
saved_path = root / f"native-body-failure-capture-v1/{label}/failure/before-rigid-integration.npz"
with np.load(saved_path) as asset:
    saved = {key: asset[key].copy() for key in asset.files}
gs.init(backend=gs.cpu, precision="64", logging_level="warning")
options = gs.options.RigidStressOptions(mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz", method="direct",
                                      contact_model="intrinsic", intrinsic_load_partition="body_source_prior")
model = StressModel(options)
surface = StressSurface(10, model.info)
info = create_traction_info(surface.info)
n_sources = saved["fit_contacts.valid"].shape[1]
contacts, sources = create_contacts(n_sources, 1, 0.006), create_sources(n_sources, 1)
tractions = create_surface_fan_tractions(surface.info, n_sources, 1, shared_support=True, source_priors=True)
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
assert valid.sum() == 1, "This audit first handles the captured single-contact cases."
i_s = int(np.flatnonzero(valid)[0])
count = int(counts[i_s])
positions = qd_to_numpy(samples.positions, transpose=True)[0, :count, i_s]
normals = qd_to_numpy(samples.normals, transpose=True)[0, :count, i_s]
faces = qd_to_numpy(samples.faces, transpose=True)[0, :count, i_s]
profiles = qd_to_numpy(tractions.profile, transpose=True)[0, :count, i_s]
packed_count = count
positive = profiles > 0
positions, normals, faces, profiles = (value[positive] for value in (positions, normals, faces, profiles))
count = len(profiles)
anchor = qd_to_numpy(contacts.center, transpose=True)[0, i_s]
point = qd_to_numpy(contacts.position, transpose=True)[0, i_s]
force = qd_to_numpy(contacts.force, transpose=True)[0, i_s]
moment = qd_to_numpy(sources.anchor_moment, transpose=True)[0, i_s]
radius = float(qd_to_numpy(contacts.radius, transpose=True)[0, i_s])
mu = float(qd_to_numpy(contacts.friction, transpose=True)[0, i_s])
scale = max(np.linalg.norm(force), np.linalg.norm(moment) / radius)
target = np.concatenate((force, moment / radius)) / scale
relative = (positions - anchor) / radius
axes = np.eye(3)[np.argmin(abs(normals), axis=1)]
tangent = np.cross(normals, axes)
tangent /= np.linalg.norm(tangent, axis=1)[:, None]
other = np.cross(normals, tangent)
angles = np.arange(64) * (2 * np.pi / 64)
reports = []
print(json.dumps({"packed_count": packed_count, "positive_count": count}), flush=True)
for kind, dilation in (("inner", 1.0), ("outer", 1 / np.cos(np.pi / 64))):
    rays = normals[:, None] + mu * dilation * (
        np.cos(angles)[None, :, None] * tangent[:, None] + np.sin(angles)[None, :, None] * other[:, None]
    )
    columns = np.concatenate((rays, np.cross(relative[:, None], rays)), axis=-1).reshape(-1, 6).T
    solved = linprog(np.zeros(columns.shape[1]), A_eq=columns, b_eq=target, bounds=(0, None),
                     options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    report = {"polygon": kind, "status": int(solved.status), "message": solved.message, "success": bool(solved.success)}
    if solved.success:
        recovered = np.einsum("qk,qki->qi", solved.x.reshape(count, 64), rays)
        resultant = np.concatenate((recovered.sum(axis=0), np.cross(relative, recovered).sum(axis=0)))
        compression = np.einsum("qi,qi->q", recovered, normals)
        report["normalized_wrench_error"] = float(np.linalg.norm(resultant - target))
        report["physical_cone_excess"] = float(np.max(np.linalg.norm(recovered - compression[:, None] * normals, axis=1) - mu * compression))
        np.savez_compressed(output / f"{kind}-primal.npz", force=recovered * scale, positions=positions, normals=normals)
    reports.append(report)
    print(json.dumps(report), flush=True)
weights = profiles / profiles.sum()
operator = np.zeros((count, 6, 3))
operator[:, :3] = np.eye(3)
x, y, z = relative.T
operator[:, 3, 1], operator[:, 3, 2] = -z, y
operator[:, 4, 0], operator[:, 4, 2] = z, -x
operator[:, 5, 0], operator[:, 5, 1] = -y, x

def evaluate(multiplier):
    values = force / scale - np.einsum("qij,i->qj", operator, multiplier)
    projected, jacobian = project_coulomb(values, normals, np.full(count, mu))
    energy = target @ multiplier + 0.5 * np.einsum("q,qi,qi->", weights, projected, projected)
    gradient = target - np.einsum("q,qij,qj->i", weights, operator, projected)
    hessian = np.einsum("q,qik,qkl,qjl->ij", weights, operator, jacobian, operator)
    return energy, gradient, hessian, weights[:, None] * scale * projected

oracle_report = None
if reports[0]["success"]:
    fitted = minimize(lambda value: evaluate(value)[:2], np.zeros(6), jac=True,
                      hess=lambda value: evaluate(value)[2], method="trust-exact",
                      options={"gtol": 1e-13, "maxiter": 2000})
    corrected = solve_root(lambda value: evaluate(value)[1], fitted.x,
                           jac=lambda value: evaluate(value)[2], options={"xtol": 1e-12})
    multiplier = min((fitted.x, corrected.x), key=lambda value: np.linalg.norm(evaluate(value)[1]))
    energy, gradient, hessian, recovered = evaluate(multiplier)
    offset = (point - anchor) / radius
    oracle_report = {"optimizer_success": bool(fitted.success), "iterations": int(fitted.nit),
                     "message": str(fitted.message), "energy": float(energy), "dual": multiplier.tolist(),
                     "force_error": float(np.linalg.norm(gradient[:3]) * scale / np.linalg.norm(force)),
                     "moment_error": float(np.linalg.norm(gradient[3:] - np.cross(offset, gradient[:3])) * scale / np.linalg.norm(force)),
                     "hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist()}
    np.savez_compressed(output / "independent-optimum.npz", force=recovered, dual=multiplier)
    print(json.dumps(oracle_report), flush=True)
iterations = []
for budget in (128, 512, 2048):
    kernel_fit_tractions(contacts, sources, tractions, info, float(np.finfo(np.float64).eps), 9e-9, budget,
                         cooperative=False, warm_start=False)
    row = {"budget": budget, "status": qd_to_numpy(tractions.status, transpose=True)[0, i_s].item(),
           "force_error": qd_to_numpy(tractions.force_error, transpose=True)[0, i_s].item(),
           "moment_error": qd_to_numpy(tractions.moment_error, transpose=True)[0, i_s].item(),
           "evaluations": qd_to_numpy(tractions.evaluations, transpose=True)[0, i_s].item()}
    iterations.append(row)
    print(json.dumps(row), flush=True)
summary = {"scope": __doc__, "case": label, "sample_count": count, "faces": np.unique(faces).tolist(),
           "radius": radius, "friction": mu, "anchor": anchor.tolist(), "point": point.tolist(),
           "force": force.tolist(), "moment": moment.tolist(), "scale": scale,
           "circle_face": qd_to_numpy(tractions.circle.face, transpose=True)[0, i_s].item(),
           "circle_refined": bool(qd_to_numpy(tractions.circle.refined, transpose=True)[0, i_s]),
           "circle_focus": qd_to_numpy(tractions.circle.focus, transpose=True)[0, i_s].tolist(),
           "polygons": reports, "native_cold_iterations": iterations,
           "positive_count": count, "packed_count": packed_count, "independent_optimum": oracle_report,
           "interpretation": "An inner primal satisfying the original cones/budget proves sampled feasibility. Outer polygon infeasibility implies sampled cone infeasibility,not continuous physical infeasibility. Iteration budgets here are diagnostic only;production128 and final budgets remain unchanged."}
np.savez_compressed(output / "cloud.npz", positions=positions, normals=normals, profiles=profiles, faces=faces,
                    anchor=anchor, point=point, force=force, moment=moment, radius=radius, friction=mu)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
