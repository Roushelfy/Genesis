"""Replay the preserved heterogeneous pressure failure and inspect its sampled convex problem."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
import scipy.optimize

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    create_contacts,
    func_contact_sample,
    func_force_frame,
    kernel_anchor,
    kernel_pressure,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json


@qd.kernel
def kernel_samples(
    contacts: StressContactState, surface: StressSurfaceInfo, coordinates: qd.Tensor, weights: qd.Tensor
):
    for i_q in range(weights.shape[0]):
        first, second = func_force_frame(contacts.force[0, 0])
        value, weight, _position, _shape, _face = func_contact_sample(
            surface.positions.shape[0] + i_q, 0, 0, first, second, contacts, surface
        )
        coordinates[i_q], weights[i_q] = value, weight


def gradient(coefficient, coordinates, weights):
    return coordinates.T @ (weights * np.maximum(coordinates @ coefficient, 0.0)) - np.array([1.0, 0.0, 0.0])


def hessian(coefficient, coordinates, weights):
    active = coordinates @ coefficient > 0.0
    return coordinates[active].T @ (weights[active, None] * coordinates[active])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, precision="64", logging_level="warning")
    mesh = Path(gs.__file__).parent.parent / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    options = gs.options.RigidStressOptions.dpr(mesh=mesh, method="pcg", quadrature=16)
    model = StressModel(options)
    surface = StressSurface(16, model.load_info)
    saved = np.load(args.failure)
    env, slot = saved["failed_selection"][0]
    contacts = create_contacts(1, 1, float(saved["radius"][env, slot]), hertz_profile=True)
    for name, array in (("position", contacts.position), ("force", contacts.force), ("normal", contacts.normal)):
        array.from_numpy(saved[name][env, slot][None, None])
    contacts.friction.fill(float(saved["friction"][env, slot]))
    contacts.valid.fill(True)
    kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
    kernel_pressure(contacts, surface.info, cooperative=args.backend == "gpu")
    result = dict(
        identity=runtime_identity(require_gpu_inventory=args.backend == "gpu"),
        failure=file_id(args.failure),
        saved_status=int(saved["status"][env, slot]),
        replay_status=int(qd_to_numpy(contacts.status)[0, 0]),
        replay_evaluations=int(qd_to_numpy(contacts.evaluations)[0, 0]),
        center_difference=float(np.linalg.norm(qd_to_numpy(contacts.center)[0, 0] - saved["center"][env, slot])),
        radius=float(saved["radius"][env, slot]),
        complete=False,
    )
    write_json(args.output / "result.json", result)
    assert bool(qd_to_numpy(contacts.is_refined)[0, 0])
    n_samples = surface.info.face_origin.shape[0] * 3 * surface.info.shape.shape[0]
    points = V_VEC(3, dtype=gs.qd_float, shape=(n_samples,))
    weights = V(dtype=gs.qd_float, shape=(n_samples,))
    kernel_samples(contacts, surface.info, points, weights)
    x, w = qd_to_numpy(points), qd_to_numpy(weights)
    active = w > 0.0
    x, w = x[active], w[active]
    total = w.sum()
    w /= total
    gram = x.T @ (w[:, None] * x)
    initial = np.linalg.solve(gram, np.array([1.0, 0.0, 0.0]))
    starts = dict(
        initial=initial,
        saved=saved["coefficient"][env, slot] * total,
        replay=qd_to_numpy(contacts.coefficient)[0, 0] * total,
    )
    feasible = scipy.optimize.linprog(
        np.zeros(len(w)), A_eq=x.T, b_eq=[1.0, 0.0, 0.0], bounds=(0, None), method="highs"
    )
    result.update(
        samples=len(w),
        gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
        feasibility=dict(success=bool(feasible.success), status=int(feasible.status), message=feasible.message),
        fits={},
    )
    for name, start in starts.items():
        fit = scipy.optimize.least_squares(
            gradient, start, jac=hessian, args=(x, w), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=1000, x_scale="jac"
        )
        result["fits"][name] = dict(
            initial_gradient=float(np.linalg.norm(gradient(start, x, w))),
            final_gradient=float(np.linalg.norm(gradient(fit.x, x, w))),
            coefficient=fit.x.tolist(),
            evaluations=int(fit.nfev),
            success=bool(fit.success),
            message=fit.message,
            active_hessian_eigenvalues=np.linalg.eigvalsh(hessian(fit.x, x, w)).tolist(),
        )
    np.savez_compressed(args.output / "sampled-problem.npz", coordinates=x, weights=w, **starts)
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
